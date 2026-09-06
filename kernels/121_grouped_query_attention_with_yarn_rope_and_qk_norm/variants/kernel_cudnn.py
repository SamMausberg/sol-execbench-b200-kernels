"""YARN attention with fused BF16 normalization/rotation and cuDNN GQA."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
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
    q_heads = query_rotated.view(batch, seq, 40, 128).transpose(1, 2)
    k_heads = key_rotated.view(batch, seq, 8, 128).transpose(1, 2)
    v_heads = value.view(batch, seq, 8, 128).transpose(1, 2)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        attention = F.scaled_dot_product_attention(
            q_heads, k_heads, v_heads, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False, scale=scaling, enable_gqa=True,
        )
    merged = attention.transpose(1, 2).contiguous().view(rows, hidden)
    torch.mm(merged, o_proj_weight.t(), out=output.view(rows, hidden))
