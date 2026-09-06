# SPDX-License-Identifier: Apache-2.0
"""Fuse SiLU into modulation GEMM and normalization into output GEMM."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _round_tf32(value):
    bits = value.to(tl.int32, bitcast=True)
    rounded = tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", constraints="=r,r",
                                       args=[bits], dtype=tl.int32, is_pure=True, pack=1)
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _modulation(T, W, Bias, O, BATCH: tl.constexpr, C: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(triton.cdiv(C, BK)):
        kk = start * BK + k
        x = tl.load(T + m[:, None] * C + kk[None, :],
                    (m[:, None] < BATCH) & (kk[None, :] < C), 0)
        silu = x * tl.sigmoid(x)
        w = tl.load(W + n[None, :] * C + kk[:, None],
                    (n[None, :] < 2 * C) & (kk[:, None] < C), 0)
        acc = tl.dot(_round_tf32(silu), _round_tf32(w), acc, input_precision="tf32")
    bias = tl.load(Bias + n, n < 2 * C, 0)
    tl.store(O + m[:, None] * (2 * C) + n[None, :], acc + bias[None, :],
             (m[:, None] < BATCH) & (n[None, :] < 2 * C))


@triton.jit
def _normalized_projection(X, Mod, W, Bias, O, EPS,
                           ROWS: tl.constexpr, SEQ: tl.constexpr, C: tl.constexpr,
                           N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                           BK: tl.constexpr, STATS: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    stats_col = tl.arange(0, STATS)
    total = tl.full((BM,), 0, tl.float32)
    for step in range(triton.cdiv(C, STATS)):
        k = step * STATS + stats_col
        x = tl.load(X + rows[:, None] * C + k[None, :],
                    (rows[:, None] < ROWS) & (k[None, :] < C), 0)
        total += tl.sum(x, 1)
    mean = total / C
    squares = tl.full((BM,), 0, tl.float32)
    for step in range(triton.cdiv(C, STATS)):
        k = step * STATS + stats_col
        x = tl.load(X + rows[:, None] * C + k[None, :],
                    (rows[:, None] < ROWS) & (k[None, :] < C), 0)
        centered = tl.where(k[None, :] < C, x - mean[:, None], 0)
        squares += tl.sum(centered * centered, 1)
    std = tl.sqrt(squares / C + EPS)
    batch = rows // SEQ
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(triton.cdiv(C, BK)):
        k = step * BK + inner
        mask = (rows[:, None] < ROWS) & (k[None, :] < C)
        x = tl.load(X + rows[:, None] * C + k[None, :], mask, 0)
        shift = tl.load(Mod + batch[:, None] * (2 * C) + k[None, :], mask, 0)
        scale = tl.load(Mod + batch[:, None] * (2 * C) + C + k[None, :], mask, 0)
        normalized = (x - mean[:, None]) / std[:, None]
        adapted = normalized * (1.0 + scale) + shift
        w = tl.load(W + columns[None, :] * C + k[:, None],
                    (columns[None, :] < N) & (k[:, None] < C), 0)
        acc = tl.dot(_round_tf32(adapted), _round_tf32(w), acc, input_precision="tf32")
    bias = tl.load(Bias + columns, columns < N, 0)
    tl.store(O + rows[:, None] * N + columns[None, :], acc + bias[None, :],
             (rows[:, None] < ROWS) & (columns[None, :] < N))


@torch.no_grad()
def run(hidden_states, temb, linear_weight, linear_bias,
        proj_out_weight, proj_out_bias, eps, output):
    inputs = (hidden_states, temb, linear_weight, linear_bias, proj_out_weight, proj_out_bias, output)
    if not all(tensor.is_contiguous() for tensor in inputs):
        mean = hidden_states.mean(-1, keepdim=True)
        variance = hidden_states.var(-1, keepdim=True, unbiased=False)
        normalized = (hidden_states - mean) / torch.sqrt(variance + eps)
        mod = F.linear(temb * torch.sigmoid(temb), linear_weight, linear_bias)
        shift, scale = mod.chunk(2, dim=-1)
        adapted = normalized * (1.0 + scale[:, None, :]) + shift[:, None, :]
        output.copy_(F.linear(adapted, proj_out_weight, proj_out_bias))
        return
    batch, seq, c = hidden_states.shape
    n = proj_out_weight.shape[0]
    mod = torch.empty((batch, 2 * c), device=hidden_states.device, dtype=hidden_states.dtype)
    _modulation[(triton.cdiv(batch, 16), triton.cdiv(2 * c, 32))](
        temb, linear_weight, linear_bias, mod, batch, c, 16, 32, 64,
        num_warps=4, num_stages=3, enable_fp_fusion=False, arch="sm80",
    )
    _normalized_projection[(triton.cdiv(batch * seq, 16), triton.cdiv(n, 64))](
        hidden_states, mod, proj_out_weight, proj_out_bias, output, eps,
        batch * seq, seq, c, n, 16, 64, 64, 256,
        num_warps=4, num_stages=3, enable_fp_fusion=False, arch="sm80",
    )
