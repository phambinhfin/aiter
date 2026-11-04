# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The paged varlen module must hold the instance the C++ mask normalization asks for.

mha_varlen_fwd_kernels.cu rewrites `causal` to false when max_seqlen_q == 1 and
there is no alibi, then selects a kernel from the rewritten mask. The JIT module
has to be generated from the same rewritten value or mha_fwd_splitkv finds no
matching instance.

Needs ROCm/aiter#4462: without it the paged branch of cmdGenFunc_mha_varlen_fwd
cannot produce a module at all and these cases skip.
"""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

SUPPORTED_GFX = ["gfx942", "gfx950"]
PAGE_SIZE = 128  # CK requires the paged block size to be a multiple of 128
NHEADS, NHEADS_K, HEAD_DIM = 8, 2, 64


def _paged_kv(k, v, seqlen_k):
    num_blocks = (seqlen_k + PAGE_SIZE - 1) // PAGE_SIZE
    shape = (num_blocks, PAGE_SIZE, NHEADS_K, HEAD_DIM)
    k_cache = torch.zeros(shape, dtype=k.dtype, device=k.device)
    v_cache = torch.zeros(shape, dtype=v.dtype, device=v.device)
    for i in range(num_blocks):
        lo, hi = i * PAGE_SIZE, min((i + 1) * PAGE_SIZE, seqlen_k)
        k_cache[i, : hi - lo] = k[lo:hi]
        v_cache[i, : hi - lo] = v[lo:hi]
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=k.device).view(
        1, num_blocks
    )
    return k_cache, v_cache, block_table


def _paged_fwd(seqlen_q, seqlen_k, causal, window_size, device="cuda"):
    dtype = dtypes.bf16
    q = torch.randn(seqlen_q, NHEADS, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(seqlen_k, NHEADS_K, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn_like(k)
    k_cache, v_cache, block_table = _paged_kv(k, v, seqlen_k)
    try:
        return aiter.flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            torch.tensor([0, seqlen_q], dtype=torch.int32, device=device),
            torch.tensor([0, seqlen_k], dtype=torch.int32, device=device),
            seqlen_q,
            seqlen_k,
            causal=causal,
            window_size=window_size,
            return_lse=True,
            block_table=block_table,
        )
    except Exception as exc:  # noqa: BLE001
        if "filter_fwd" in str(exc):
            pytest.skip("paged codegen needs ROCm/aiter#4462")
        raise


_CASES = [
    # (seqlen_q, seqlen_k, window_size), each with causal on and off
    (1, 2048, (-1, -1)),  # the rewritten case: causal with a single query row
    (128, 2048, (-1, -1)),  # control: causal keeps its mask
    (1, 2048, (256, 0)),  # control: the window keeps the mask non-trivial
]


@pytest.mark.parametrize("seqlen_q,seqlen_k,window_size", _CASES)
def test_paged_varlen_mask_module_matches_kernel(seqlen_q, seqlen_k, window_size):
    """causal=True must dispatch, and match causal=False wherever C++ equates them."""
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"paged varlen forward not validated on {get_gfx()}")
    torch.manual_seed(0)
    out, lse = _paged_fwd(seqlen_q, seqlen_k, True, window_size)
    assert torch.isfinite(out).all()

    if seqlen_q == 1 and window_size == (-1, -1):
        torch.manual_seed(0)
        ref_out, ref_lse = _paged_fwd(seqlen_q, seqlen_k, False, window_size)
        torch.testing.assert_close(out, ref_out, atol=0, rtol=0)
        torch.testing.assert_close(lse, ref_lse, atol=0, rtol=0)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "paged varlen forward not validated on %s; skipping", get_gfx()
        )
        return
    for seqlen_q, seqlen_k, window_size in _CASES:
        test_paged_varlen_mask_module_matches_kernel(seqlen_q, seqlen_k, window_size)
        aiter.logger.info(
            "paged varlen dispatch ok: sq=%s sk=%s window=%s",
            seqlen_q,
            seqlen_k,
            window_size,
        )


if __name__ == "__main__":
    main()
