# SPDX-License-Identifier: Apache-2.0
"""Fuse projection and residual while preserving the intermediate BF16 rounding."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _projection_residual(
    A, W, R, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    PROGRAMS: tl.constexpr, WARP_SPECIALIZE: tl.constexpr,
):
    mt: tl.constexpr = triton.cdiv(M, BM)
    nt: tl.constexpr = triton.cdiv(N, BN)
    for tile in tl.range(tl.program_id(0), mt * nt, PROGRAMS,
                         flatten=True, warp_specialize=WARP_SPECIALIZE):
        row = (tile % mt) * BM
        col = (tile // mt) * BN
        acc = tl.full((BM, BN), 0, tl.float32)
        for start in range(triton.cdiv(K, BK)):
            a = A.load([row, start * BK])
            w = W.load([col, start * BK])
            acc = tl.dot(a, w.T, acc)
        projected = acc.to(tl.bfloat16).to(tl.float32)
        residual = R.load([row, col]).to(tl.float32)
        result = (projected + residual).to(tl.bfloat16)
        O.store([row, col], result)


@torch.no_grad()
def library_run(attn_output, residual, o_proj_weight, output):
    n = attn_output.shape[-1]
    projected = torch.empty_like(output)
    torch.mm(attn_output.reshape(-1, n), o_proj_weight.t(), out=projected.view(-1, n))
    torch.add(projected, residual, out=output)


def run(attn_output, residual, o_proj_weight, output):
    if not (attn_output.is_contiguous() and residual.is_contiguous() and o_proj_weight.is_contiguous()):
        return library_run(attn_output, residual, o_proj_weight, output)
    n = attn_output.shape[-1]
    m = attn_output.numel() // n
    bm, bn, bk = 128, 128, 64
    a = TensorDescriptor(attn_output, [m, n], [n, 1], [bm, bk])
    w = TensorDescriptor(o_proj_weight, [n, n], [n, 1], [bn, bk])
    r = TensorDescriptor(residual, [m, n], [n, 1], [bm, bn])
    o = TensorDescriptor(output, [m, n], [n, 1], [bm, bn])
    programs = min(296, triton.cdiv(m, bm) * triton.cdiv(n, bn))
    _projection_residual[(programs,)](
        a, w, r, o, m, n, n, bm, bn, bk, programs, False,
        num_warps=4, num_stages=3,
    )
