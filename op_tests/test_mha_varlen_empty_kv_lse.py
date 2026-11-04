# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A varlen sequence with no keys must report lse = -inf, or sink[h] with a sink.

Callers that split attention over several calls (chunked context, context
parallel) merge the partials by exp(lse). An empty partial that reports lse = 0
gets weight 1 and drags the merged output toward zero, so the value stored for
those rows is part of the op contract, not an implementation detail.
"""

import itertools

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

SUPPORTED_GFX = ["gfx942", "gfx950"]

# hdim 64 routes to CK; bf16 hdim 128 qualifies for the fmha_v3_varlen_fwd ASM
# kernels, so the pair covers both varlen forward backends.
HEAD_DIMS = [64, 128]
SINK_VALUES = [None, 1.5]

SEQLENS_Q = [4, 4, 4]
SEQLENS_K = [8, 0, 5]
EMPTY = 1  # index of the sequence with no keys
NHEADS, NHEADS_K = 8, 8


def _cu_seqlens(seqlens, device):
    return torch.tensor(
        [0] + list(itertools.accumulate(seqlens)), dtype=torch.int32, device=device
    )


def _merge(out_a, lse_a, out_b, lse_b):
    """LSE merge of two attention partials, in fp32."""
    lse_a, lse_b = lse_a.float(), lse_b.float()
    m = torch.maximum(lse_a, lse_b)
    sa, sb = torch.exp(lse_a - m), torch.exp(lse_b - m)
    den = sa + sb
    wa = (sa / den).transpose(0, 1).unsqueeze(-1)
    wb = (sb / den).transpose(0, 1).unsqueeze(-1)
    return out_a.float() * wa + out_b.float() * wb


def _varlen_fwd(head_dim, sink_val, dtype=dtypes.bf16, device="cuda"):
    cu_q = _cu_seqlens(SEQLENS_Q, device)
    cu_k = _cu_seqlens(SEQLENS_K, device)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, NHEADS, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_k, NHEADS_K, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    sink_ptr = (
        torch.full((NHEADS,), sink_val, device=device, dtype=torch.float32)
        if sink_val is not None
        else None
    )
    out, lse = aiter.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(SEQLENS_Q),
        max(SEQLENS_K),
        causal=False,
        return_lse=True,
        sink_ptr=sink_ptr,
    )
    return out, lse, cu_q


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("sink_val", SINK_VALUES)
def test_empty_kv_sequence_lse(head_dim, sink_val):
    """A sequence with no keys reports lse = -inf (sink[h] with a sink) and out = 0."""
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"varlen forward not validated on {get_gfx()}")
    torch.manual_seed(0)
    out, lse, cu_q = _varlen_fwd(head_dim, sink_val)

    empty = slice(int(cu_q[EMPTY]), int(cu_q[EMPTY + 1]))
    expected = float("-inf") if sink_val is None else sink_val
    assert torch.all(lse[:, empty] == expected), lse[:, empty]
    assert torch.all(out[empty] == 0), out[empty]

    for i in range(len(SEQLENS_Q)):
        if i == EMPTY:
            continue
        rows = slice(int(cu_q[i]), int(cu_q[i + 1]))
        assert torch.isfinite(lse[:, rows]).all(), lse[:, rows]


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_empty_kv_sequence_is_identity_under_merge(head_dim):
    """Merging a no-key sequence's partial into a real one returns the real one."""
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"varlen forward not validated on {get_gfx()}")
    torch.manual_seed(0)
    out, lse, cu_q = _varlen_fwd(head_dim, None)

    empty = slice(int(cu_q[EMPTY]), int(cu_q[EMPTY + 1]))
    real = slice(int(cu_q[0]), int(cu_q[1]))
    n = min(empty.stop - empty.start, real.stop - real.start)
    merged = _merge(
        out[empty.start : empty.start + n],
        lse[:, empty.start : empty.start + n],
        out[real.start : real.start + n],
        lse[:, real.start : real.start + n],
    )
    torch.testing.assert_close(
        merged, out[real.start : real.start + n].float(), atol=0, rtol=0
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("varlen forward not validated on %s; skipping", get_gfx())
        return
    for head_dim, sink_val in itertools.product(HEAD_DIMS, SINK_VALUES):
        test_empty_kv_sequence_lse(head_dim, sink_val)
        aiter.logger.info("empty-kv lse ok: head_dim=%s sink=%s", head_dim, sink_val)
    for head_dim in HEAD_DIMS:
        test_empty_kv_sequence_is_identity_under_merge(head_dim)
        aiter.logger.info("empty-kv merge identity ok: head_dim=%s", head_dim)


if __name__ == "__main__":
    main()
