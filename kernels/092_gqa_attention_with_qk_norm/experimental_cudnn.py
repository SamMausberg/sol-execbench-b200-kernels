"""Per-head QK RMSNorm/RoPE with grouped causal cuDNN attention."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.attention import SDPBackend, sdpa_kernel


@triton.jit
def _prepare_qk(
    Q, K, QW, KW, COS, SIN, YQ, YK,
    EPS: tl.constexpr, S: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    QWS: tl.constexpr, KWS: tl.constexpr,
    HEADS: tl.constexpr,
):
    program = tl.program_id(0)
    token = program // (96 // HEADS + 1)
    group = program % (96 // HEADS + 1)
    heads = tl.arange(0, HEADS)
    channels = tl.arange(0, 128)
    if group < 96 // HEADS:
        offsets = (token * 96 + group * HEADS + heads[:, None]) * 128 + channels[None, :]
        values = tl.load(Q + offsets).to(tl.float32)
        weight = tl.load(QW + channels * QWS).to(tl.float32)
    else:
        offsets = (token * 8 + heads[:, None]) * 128 + channels[None, :]
        values = tl.load(K + offsets, heads[:, None] < 8, other=0).to(tl.float32)
        weight = tl.load(KW + channels * KWS).to(tl.float32)
    variance = tl.sum(values * values, axis=1) * (1.0 / 128)
    normalized = values * tl.rsqrt(variance + EPS)[:, None]
    normalized = (normalized * weight[None, :]).to(tl.bfloat16).to(tl.float32)
    other_channels = tl.broadcast_to((channels ^ 64)[None, :], (HEADS, 128))
    rotated = tl.gather(normalized, other_channels, axis=1)
    rotated = tl.where(channels[None, :] < 64, -rotated, rotated)
    b, s = token // S, token % S
    cosine = tl.load(COS + b * C0 + s * C1 + channels * C2).to(tl.float32)
    sine = tl.load(SIN + b * S0 + s * S1 + channels * S2).to(tl.float32)
    # Every elementwise operation in the reference produces a BF16 tensor.
    direct = (normalized * cosine[None, :]).to(tl.bfloat16).to(tl.float32)
    rotated = (rotated * sine[None, :]).to(tl.bfloat16).to(tl.float32)
    result = direct + rotated
    if group < 96 // HEADS:
        tl.store(YQ + offsets, result)
    else:
        tl.store(YK + offsets, result, heads[:, None] < 8)


def prepare(query, key, q_norm_weight, k_norm_weight, cos, sin, eps, heads=16, warps=None):
    batch, sequence = query.shape[:2]
    if warps is None:
        warps = 8 if batch * sequence <= 128 else 4
    prepared_query = torch.empty_like(query)
    prepared_key = torch.empty_like(key)
    _prepare_qk[(batch * sequence * (96 // heads + 1),)](
        query, key, q_norm_weight, k_norm_weight, cos, sin,
        prepared_query, prepared_key, eps, sequence,
        *cos.stride(), *sin.stride(),
        q_norm_weight.stride(0), k_norm_weight.stride(0),
        heads, num_warps=warps, enable_fp_fusion=False,
    )
    return prepared_query, prepared_key


@torch.no_grad()
def run(
    hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
    v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight,
    cos, sin, rms_norm_eps, output,
):
    batch, sequence, _ = hidden_states.shape
    query = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    key = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    value = F.linear(hidden_states, v_proj_weight, v_proj_bias)
    query, key = prepare(query, key, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps)
    query = query.view(batch, sequence, 96, 128).transpose(1, 2)
    key = key.view(batch, sequence, 8, 128).transpose(1, 2)
    value = value.view(batch, sequence, 8, 128).transpose(1, 2)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=True,
        )
    attended = attended.transpose(1, 2).contiguous().view(batch * sequence, 12288)
    torch.mm(attended, o_proj_weight.t(), out=output.view(batch * sequence, 4096))
