# SPDX-License-Identifier: Apache-2.0
"""FP16 GEMM candidates for small numbers of query rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_small(A, B, C, M: tl.constexpr, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, 2048)
    weights = tl.load(B + n[:, None] * 2048 + k[None, :])
    for row in tl.static_range(M):
        values = tl.load(A + row * 2048 + k)
        products = weights.to(tl.float32) * values[None, :].to(tl.float32)
        result = tl.sum(products, 1)
        tl.store(C + row * 5120 + n, result)


@triton.jit
def _gemm_split(
    A, B, C, M: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    reduction: tl.constexpr = 2048 // SPLIT
    k = part * reduction + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(triton.cdiv(reduction, BK)):
        col = k + start * BK
        a = tl.load(A + m[:, None] * 2048 + col[None, :], m[:, None] < M, 0)
        b = tl.load(B + n[None, :] * 2048 + col[:, None])
        acc = tl.dot(a, b, acc)
    offsets = part * M * 5120 + m[:, None] * 5120 + n[None, :]
    tl.store(C + offsets, acc, m[:, None] < M)


@triton.jit
def _reduce_splits(PARTIAL, C, ELEMENTS: tl.constexpr,
                   SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parts = tl.arange(0, SPLIT)
    values = tl.load(
        PARTIAL + parts[:, None] * ELEMENTS + offsets[None, :],
        offsets[None, :] < ELEMENTS, 0,
    )
    tl.store(C + offsets, tl.sum(values, 0), offsets < ELEMENTS)


def launch_split(A, B, C, bm, bn, bk, split, warps, stages):
    m = A.shape[0]
    partial = C if split == 1 else torch.empty(
        (split, m, 5120), device=A.device, dtype=torch.float32,
    )
    _gemm_split[(triton.cdiv(m, bm), 5120 // bn, split)](
        A, B, partial, m, bm, bn, bk, split,
        num_warps=warps, num_stages=stages,
    )
    if split != 1:
        _reduce_splits[(triton.cdiv(m * 5120, 1024),)](
            partial, C, m * 5120, split, 1024, num_warps=4,
        )


@torch.no_grad()
def run(A, B, C):
    m = A.shape[0]
    if m <= 8:
        _gemv_small[(5120 // 4,)](A, B, C, m, 4, num_warps=4)
    elif m <= 256:
        launch_split(A, B, C, 32, 64, 128, 2, 4, 3)
    else:
        torch.mm(A, B.T, out=C)
