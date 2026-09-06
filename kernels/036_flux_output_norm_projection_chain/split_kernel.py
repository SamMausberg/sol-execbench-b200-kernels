# SPDX-License-Identifier: Apache-2.0
"""Materialize fused normalization once, then use library projections."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _silu(X, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < SIZE, 0)
    tl.store(Y + i, x * tl.sigmoid(x), i < SIZE)


@triton.jit
def _norm_mod(X, M, Y, EPS, SEQ: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + k, k < C, 0)
    mean = tl.sum(x, 0) / C
    centered = tl.where(k < C, x - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / C
    normalized = centered / tl.sqrt(variance + EPS)
    batch = row // SEQ
    shift = tl.load(M + batch * (2 * C) + k, k < C, 0)
    scale = tl.load(M + batch * (2 * C) + C + k, k < C, 0)
    adapted = normalized * (1.0 + scale) + shift
    tl.store(Y + row * C + k, adapted, k < C)


@torch.no_grad()
def run(hidden_states, temb, linear_weight, linear_bias,
        proj_out_weight, proj_out_bias, eps, output):
    tensors = (hidden_states, temb, linear_weight, linear_bias, proj_out_weight, proj_out_bias, output)
    if not all(t.is_contiguous() for t in tensors):
        mean = hidden_states.mean(-1, keepdim=True)
        variance = hidden_states.var(-1, keepdim=True, unbiased=False)
        normalized = (hidden_states - mean) / torch.sqrt(variance + eps)
        mod = F.linear(temb * torch.sigmoid(temb), linear_weight, linear_bias)
        shift, scale = mod.chunk(2, dim=-1)
        adapted = normalized * (1.0 + scale[:, None, :]) + shift[:, None, :]
        output.copy_(F.linear(adapted, proj_out_weight, proj_out_bias))
        return
    batch, seq, c = hidden_states.shape
    silu = torch.empty_like(temb)
    adapted = torch.empty_like(hidden_states)
    _silu[(triton.cdiv(temb.numel(), 256),)](temb, silu, temb.numel(), 256,
                                          enable_fp_fusion=False)
    mod = F.linear(silu, linear_weight, linear_bias)
    _norm_mod[(batch * seq,)](hidden_states, mod, adapted, eps, seq, c,
                             triton.next_power_of_2(c), num_warps=4,
                             enable_fp_fusion=False)
    torch.addmm(proj_out_bias, adapted.view(-1, c), proj_out_weight.t(),
                out=output.view(-1, proj_out_weight.shape[0]))
