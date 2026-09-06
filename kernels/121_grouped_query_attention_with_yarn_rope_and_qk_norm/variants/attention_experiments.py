"""YARN attention with fused BF16 normalization, rotation and grouped attention."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _yarn_coefficients(
    positions,
    inv_freq,
    coefficients,
    N_ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    POS_BATCH_STRIDE: tl.constexpr,
    POS_SEQ_STRIDE: tl.constexpr,
    ATTENTION_FACTOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, channel = index // 64, index % 64
    active = row < N_ROWS
    position = tl.load(
        positions + (row // SEQ_LEN) * POS_BATCH_STRIDE
        + (row % SEQ_LEN) * POS_SEQ_STRIDE,
        active,
        other=0,
    ).to(tl.float32)
    frequency = tl.load(inv_freq + channel)
    angle = position * frequency
    cosine = libdevice.cos(angle) * ATTENTION_FACTOR
    sine = libdevice.sin(angle) * ATTENTION_FACTOR
    tl.store(coefficients + row * 128 + channel, cosine, active)
    tl.store(coefficients + row * 128 + channel + 64, sine, active)


@triton.jit
def _normalize_rotate_row(
    x,
    weight,
    coefficients,
    output,
    row,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel = tl.arange(0, BLOCK)
    active = channel < WIDTH
    value = tl.load(x + row * WIDTH + channel, active, other=0.0).to(tl.float32)
    scale = tl.load(weight + channel, active, other=0.0).to(tl.float32)
    variance = tl.sum(value * value, 0) / WIDTH
    inverse_rms = tl.rsqrt(variance + EPS)
    normalized = ((value * inverse_rms) * scale).to(tl.bfloat16).to(tl.float32)
    partner = tl.gather(normalized, channel ^ 64, axis=0)
    partner = tl.where(channel % 128 < 64, -partner, partner)
    cosine = tl.load(coefficients + row * 128 + channel % 64).to(tl.float32)
    sine = tl.load(coefficients + row * 128 + channel % 64 + 64).to(tl.float32)
    # The reference stores each product in BF16 before adding the products.
    direct = (normalized * cosine).to(tl.bfloat16).to(tl.float32)
    rotated = (partner * sine).to(tl.bfloat16).to(tl.float32)
    tl.store(output + row * WIDTH + channel, direct + rotated, active)


@triton.jit
def _normalize_rotate_qk(
    query,
    key,
    q_weight,
    k_weight,
    coefficients,
    query_out,
    key_out,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.program_id(1) == 0:
        _normalize_rotate_row(query, q_weight, coefficients, query_out,
                              row, 5120, EPS, 8192)
    else:
        _normalize_rotate_row(key, k_weight, coefficients, key_out,
                              row, 1024, EPS, 1024)


@triton.jit
def _grouped_attention(
    query,
    key,
    value,
    mask,
    output,
    SEQ_LEN: tl.constexpr,
    MASK_BATCH_STRIDE: tl.constexpr,
    MASK_ROW_STRIDE: tl.constexpr,
    MASK_COL_STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SKIP_EMPTY: tl.constexpr = False,
    HEAD_MAJOR: tl.constexpr = False,
    USE_TMA: tl.constexpr = False,
    WARP_SPECIALIZE: tl.constexpr = False,
):
    query_row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    batch_head = tl.program_id(1)
    batch, head = batch_head // 40, batch_head % 40
    kv_head = head // 5
    dimension = tl.arange(0, 128)
    keys = tl.arange(0, BLOCK_N)
    if HEAD_MAJOR:
        q_base = (batch * 40 + head) * SEQ_LEN * 128
        kv_base = (batch * 8 + kv_head) * SEQ_LEN * 128
        q_stride: tl.constexpr = 128
        kv_stride: tl.constexpr = 128
    else:
        q_base = batch * SEQ_LEN * 5120 + head * 128
        kv_base = batch * SEQ_LEN * 1024 + kv_head * 128
        q_stride: tl.constexpr = 5120
        kv_stride: tl.constexpr = 1024
    if USE_TMA:
        q_desc = tl.make_tensor_descriptor(
            query + q_base, [SEQ_LEN, 128], [q_stride, 1], [BLOCK_M, 128])
        k_desc = tl.make_tensor_descriptor(
            key + kv_base, [SEQ_LEN, 128], [kv_stride, 1], [BLOCK_N, 128])
        v_desc = tl.make_tensor_descriptor(
            value + kv_base, [SEQ_LEN, 128], [kv_stride, 1], [BLOCK_N, 128])
        q = q_desc.load([tl.program_id(0) * BLOCK_M, 0])
    else:
        q = tl.load(
            query + q_base + query_row[:, None] * q_stride + dimension[None, :],
            query_row[:, None] < SEQ_LEN,
            other=0.0,
        )
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    normalizer = tl.full((BLOCK_M,), 0.0, tl.float32)
    accumulated = tl.full((BLOCK_M, 128), 0.0, tl.float32)
    for start in tl.range(0, SEQ_LEN, BLOCK_N, warp_specialize=WARP_SPECIALIZE):
        key_row = start + keys
        additive_mask = tl.load(
            mask + batch * MASK_BATCH_STRIDE
            + query_row[:, None] * MASK_ROW_STRIDE
            + key_row[None, :] * MASK_COL_STRIDE,
            (query_row[:, None] < SEQ_LEN) & (key_row[None, :] < SEQ_LEN),
            other=-float("inf"),
        ).to(tl.float32)
        if not SKIP_EMPTY or tl.max(tl.max(additive_mask, 1), 0) != -float("inf"):
            if USE_TMA:
                k = k_desc.load([start, 0]).T
            else:
                k = tl.load(
                    key + kv_base + key_row[None, :] * kv_stride + dimension[:, None],
                    key_row[None, :] < SEQ_LEN,
                    other=0.0,
                )
            product = tl.dot(q, k).to(tl.bfloat16).to(tl.float32)
            score = (product * SCALE).to(tl.bfloat16).to(tl.float32)
            score = (score + additive_mask).to(tl.bfloat16).to(tl.float32)
            updated_maximum = tl.maximum(maximum, tl.max(score, 1))
            # A tile can be fully masked even when its row has valid later keys.
            safe_maximum = tl.where(updated_maximum == -float("inf"), 0.0, updated_maximum)
            probability = tl.exp2((score - safe_maximum[:, None]) * 1.4426950408889634)
            correction = tl.exp2((maximum - safe_maximum) * 1.4426950408889634)
            normalizer = normalizer * correction + tl.sum(probability, 1)
            accumulated = accumulated * correction[:, None]
            if USE_TMA:
                v = v_desc.load([start, 0])
            else:
                v = tl.load(
                    value + kv_base + key_row[:, None] * kv_stride + dimension[None, :],
                    key_row[:, None] < SEQ_LEN,
                    other=0.0,
                )
            accumulated = tl.dot(probability.to(tl.bfloat16), v, accumulated)
            maximum = updated_maximum
    result = accumulated / normalizer[:, None]
    tl.store(
        output + (batch * SEQ_LEN + query_row[:, None]) * 5120
        + head * 128 + dimension[None, :],
        result,
        query_row[:, None] < SEQ_LEN,
    )


@torch.no_grad()
def run(
    hidden_states,
    position_ids,
    attention_mask,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
    inv_freq,
    rms_norm_eps,
    attention_factor,
    scaling,
    output,
):
    batch, seq, hidden = hidden_states.shape
    rows = batch * seq
    query = F.linear(hidden_states, q_proj_weight)
    key = F.linear(hidden_states, k_proj_weight)
    value = F.linear(hidden_states, v_proj_weight)
    coefficients = torch.empty((rows, 128), dtype=torch.bfloat16, device=hidden_states.device)
    query_rotated, key_rotated = torch.empty_like(query), torch.empty_like(key)
    _yarn_coefficients[(triton.cdiv(rows * 64, 256),)](
        position_ids, inv_freq, coefficients, rows, seq,
        position_ids.stride(0), position_ids.stride(1), attention_factor, 256,
        num_warps=4, enable_fp_fusion=False,
    )
    _normalize_rotate_qk[(rows, 2)](
        query, key, q_norm_weight, k_norm_weight, coefficients,
        query_rotated, key_rotated, rms_norm_eps,
        num_warps=8, enable_fp_fusion=False,
    )
    merged = torch.empty_like(query).view(rows, hidden)
    _grouped_attention[(triton.cdiv(seq, 64), batch * 40)](
        query_rotated, key_rotated, value, attention_mask, merged, seq,
        attention_mask.stride(0), attention_mask.stride(2),
        attention_mask.stride(3), scaling, 64, 64,
        num_warps=4, num_stages=3, enable_fp_fusion=False,
    )
    torch.mm(merged, o_proj_weight.t(), out=output.view(rows, hidden))
