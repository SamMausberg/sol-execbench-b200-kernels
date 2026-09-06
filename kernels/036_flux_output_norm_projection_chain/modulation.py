# SPDX-License-Identifier: Apache-2.0
"""Fused FP32 SiLU and small-batch modulation projection by vector reductions."""
import triton
import triton.language as tl


@triton.jit
def _mod_gemv(T, W, Bias, O, BATCH: tl.constexpr, C: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, CHUNK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    k = tl.arange(0, CHUNK)
    total = tl.full((BN, BM), 0.0, tl.float32)
    for step in range(triton.cdiv(C, CHUNK)):
        column = step * CHUNK + k
        x = tl.load(T + m[:, None] * C + column[None, :],
                    (m[:, None] < BATCH) & (column[None, :] < C), 0)
        silu = x * tl.sigmoid(x)
        weight = tl.load(W + n[:, None] * C + column[None, :],
                         (n[:, None] < 2*C) & (column[None, :] < C), 0)
        total += tl.sum(weight[:, None, :] * silu[None, :, :], 2)
    bias = tl.load(Bias + n, n < 2*C, 0)
    tl.store(O + m[None, :] * (2*C) + n[:, None], total + bias[:, None],
             (m[None, :] < BATCH) & (n[:, None] < 2*C))
