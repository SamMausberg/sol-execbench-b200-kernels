# SPDX-License-Identifier: Apache-2.0
"""FP32 modulation projection with fused bias and six concrete output views."""
import torch
import triton
import triton.language as tl


@triton.jit
def _projection(E, W, Bias, Output, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                E0: tl.constexpr, E1: tl.constexpr, W0: tl.constexpr, W1: tl.constexpr,
                B0: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    columns = pn * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(triton.cdiv(K, BK)):
        k = block * BK + inner
        left = tl.load(E + rows[:, None] * E0 + k[None, :] * E1,
                       (rows[:, None] < M) & (k[None, :] < K), 0)
        right = tl.load(W + columns[None, :] * W0 + k[:, None] * W1,
                        (columns[None, :] < N) & (k[:, None] < K), 0)
        accumulator = tl.dot(left, right, accumulator, input_precision='tf32x3')
    bias = tl.load(Bias + columns * B0, columns < N, 0)
    tl.store(Output + rows[:, None] * N + columns[None, :], accumulator + bias[None, :],
             (rows[:, None] < M) & (columns[None, :] < N))


def launch(emb, weight, bias, output, bm=64, bn=64, bk=64, stages=3):
    m, k = emb.shape
    n = weight.shape[0]
    _projection[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        emb, weight, bias, output, m, n, k, *emb.stride(), *weight.stride(), bias.stride(0),
        bm, bn, bk, num_warps=4, num_stages=stages, enable_fp_fusion=False, arch='sm80')


@torch.no_grad()
def run(emb, weight, bias):
    m, k = emb.shape
    output = torch.empty((m, weight.shape[0]), device=emb.device, dtype=torch.float32)
    views = tuple(output.split(k, dim=1))
    launch(emb, weight, bias, output, bm=16 if m <= 16 else 32 if m <= 64 else 64)
    return views
