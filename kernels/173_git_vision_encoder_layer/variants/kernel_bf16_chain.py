# SPDX-License-Identifier: Apache-2.0
"""FP32 GIT vision encoder with fused projections and a BF16 high/low product accumulation chain."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dot(a, b, accumulator):
    a_high = a.to(tl.bfloat16)
    b_high = b.to(tl.bfloat16)
    a_low = (a - a_high.to(tl.float32)).to(tl.bfloat16)
    b_low = (b - b_high.to(tl.float32)).to(tl.bfloat16)
    accumulator = tl.dot(a_low, b_high, accumulator)
    accumulator = tl.dot(a_high, b_low, accumulator)
    return tl.dot(a_high, b_high, accumulator)


@triton.jit
def _layer_norm(x, weight, bias, output, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    value = tl.load(x + row * 768 + channel, channel < 768, other=0.0)
    mean = tl.sum(value, 0) / 768
    centered = value - mean
    variance = tl.sum(tl.where(channel < 768, centered * centered, 0.0), 0) / 768
    normalized = centered / tl.sqrt(variance + EPS)
    scale = tl.load(weight + channel, channel < 768, other=0.0)
    shift = tl.load(bias + channel, channel < 768, other=0.0)
    tl.store(output + row * 768 + channel, normalized * scale + shift, channel < 768)


@triton.jit
def _qkv_projection(x, qw, qb, kw, kb, vw, vb, q, k, v,
                    ROWS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    projection = tl.program_id(1) // tl.cdiv(768, BN)
    column = (tl.program_id(1) % tl.cdiv(768, BN)) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    weight = tl.where(projection == 0, qw, tl.where(projection == 1, kw, vw))
    bias = tl.where(projection == 0, qb, tl.where(projection == 1, kb, vb))
    output = tl.where(projection == 0, q, tl.where(projection == 1, k, v))
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(768, BK)):
        reduction = block * BK + inner
        a = tl.load(x + row[:, None] * 768 + reduction[None, :],
                    (row[:, None] < ROWS) & (reduction[None, :] < 768), other=0.0)
        b = tl.load(weight + column[None, :] * 768 + reduction[:, None],
                    (column[None, :] < 768) & (reduction[:, None] < 768), other=0.0)
        accumulator = _dot(a, b, accumulator)
    shift = tl.load(bias + column, column < 768, other=0.0)
    tl.store(output + row[:, None] * 768 + column[None, :], accumulator + shift[None, :],
             (row[:, None] < ROWS) & (column[None, :] < 768))


@triton.jit
def _attention(q, k, v, output, SEQ: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    batch_head = tl.program_id(1)
    batch, head = batch_head // 12, batch_head % 12
    dimension = tl.arange(0, 64)
    base = batch * SEQ * 768 + head * 64
    query = tl.load(q + base + row[:, None] * 768 + dimension[None, :],
                    row[:, None] < SEQ, other=0.0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    normalizer = tl.full((BM,), 0.0, tl.float32)
    result = tl.full((BM, 64), 0.0, tl.float32)
    for block in range(tl.cdiv(SEQ, BN)):
        column = block * BN + tl.arange(0, BN)
        key = tl.load(k + base + column[None, :] * 768 + dimension[:, None],
                      column[None, :] < SEQ, other=0.0)
        score = _dot(query, key, tl.full((BM, BN), 0, tl.float32)) * 0.125
        score = tl.where(column[None, :] < SEQ, score, -float("inf"))
        updated = tl.maximum(maximum, tl.max(score, 1))
        probability = tl.exp2((score - updated[:, None]) * 1.4426950408889634)
        correction = tl.exp2((maximum - updated) * 1.4426950408889634)
        normalizer = normalizer * correction + tl.sum(probability, 1)
        result = result * correction[:, None]
        value = tl.load(v + base + column[:, None] * 768 + dimension[None, :],
                        column[:, None] < SEQ, other=0.0)
        result = _dot(probability, value, result)
        maximum = updated
    tl.store(output + base + row[:, None] * 768 + dimension[None, :],
             result / normalizer[:, None], row[:, None] < SEQ)


@triton.jit
def _linear_epilogue(a, weight, bias, residual, output,
                     M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     EPILOGUE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        reduction = block * BK + inner
        left = tl.load(a + row[:, None] * K + reduction[None, :],
                       (row[:, None] < M) & (reduction[None, :] < K), other=0.0)
        right = tl.load(weight + column[None, :] * K + reduction[:, None],
                        (column[None, :] < N) & (reduction[:, None] < K), other=0.0)
        accumulator = _dot(left, right, accumulator)
    shift = tl.load(bias + column, column < N, other=0.0)
    value = accumulator + shift[None, :]
    mask = (row[:, None] < M) & (column[None, :] < N)
    if EPILOGUE == 1:
        # Keep the FP32 product before sigmoid and the final multiplication.
        sigmoid = 1.0 / (1.0 + tl.exp(-(1.702 * value)))
        value = value * sigmoid
    elif EPILOGUE == 2:
        skip = tl.load(residual + row[:, None] * N + column[None, :], mask, other=0.0)
        value = skip + value
    tl.store(output + row[:, None] * N + column[None, :], value, mask)


@torch.no_grad()
def run(hidden_states, layer_norm1_weight, layer_norm1_bias,
        q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias,
        out_proj_weight, out_proj_bias, layer_norm2_weight, layer_norm2_bias,
        fc1_weight, fc1_bias, fc2_weight, fc2_bias, layer_norm_eps, output):
    batch, sequence, width = hidden_states.shape
    rows = batch * sequence
    normalized = torch.empty((rows, width), device=hidden_states.device, dtype=hidden_states.dtype)
    projections = torch.empty((3, rows, width), device=hidden_states.device, dtype=hidden_states.dtype)
    query, key, value = projections.unbind(0)
    attended = torch.empty_like(normalized)
    after_attention = torch.empty_like(normalized)
    intermediate = torch.empty((rows, 3072), device=hidden_states.device, dtype=hidden_states.dtype)
    _layer_norm[(rows,)](hidden_states, layer_norm1_weight, layer_norm1_bias, normalized,
                        layer_norm_eps, 1024, num_warps=4, enable_fp_fusion=False)
    _qkv_projection[(triton.cdiv(rows, 64), 3 * 12)](
        normalized, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
        v_proj_weight, v_proj_bias, query, key, value, rows, 64, 64, 32,
        num_warps=4, num_stages=2, enable_fp_fusion=False)
    _attention[(triton.cdiv(sequence, 32), batch * 12)](
        query, key, value, attended, sequence, 32, 64,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    _linear_epilogue[(triton.cdiv(rows, 64), 12)](
        attended, out_proj_weight, out_proj_bias, hidden_states, after_attention,
        rows, 768, 768, 2, 64, 64, 32, num_warps=4, num_stages=2, enable_fp_fusion=False)
    _layer_norm[(rows,)](after_attention, layer_norm2_weight, layer_norm2_bias, normalized,
                        layer_norm_eps, 1024, num_warps=4, enable_fp_fusion=False)
    _linear_epilogue[(triton.cdiv(rows, 64), 48)](
        normalized, fc1_weight, fc1_bias, normalized, intermediate,
        rows, 3072, 768, 1, 64, 64, 32, num_warps=4, num_stages=2, enable_fp_fusion=False)
    _linear_epilogue[(triton.cdiv(rows, 64), 12)](
        intermediate, fc2_weight, fc2_bias, after_attention, output,
        rows, 768, 3072, 2, 64, 64, 32, num_warps=4, num_stages=2, enable_fp_fusion=False)
