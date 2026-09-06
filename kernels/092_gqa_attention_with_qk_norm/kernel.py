"""Per-head QK RMSNorm/RoPE with rounded grouped causal attention."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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


@triton.jit
def _causal_attention(
    Q, K, V, O, S: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr,
    TWO_PASS: tl.constexpr,
):
    block = tl.program_id(0)
    group_index = tl.program_id(1)
    groups_per_batch: tl.constexpr = 96 // GROUP
    batch = group_index // groups_per_batch
    group = group_index % groups_per_batch
    row = block * BM + tl.arange(0, BM)
    sequence = row // GROUP
    head = group * GROUP + row % GROUP
    kv_head = group * GROUP // 12
    dimension = tl.arange(0, 128)
    columns = tl.arange(0, BN)
    q_offset = (batch * S + sequence[:, None]) * 12288 + head[:, None] * 128 + dimension[None, :]
    q = tl.load(Q + q_offset, sequence[:, None] < S, other=0.0)
    kv_base = batch * S * 1024 + kv_head * 128
    end = tl.minimum(S, tl.cdiv((block + 1) * BM, GROUP))
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    normalizer = tl.full((BM,), 0.0, tl.float32)
    accumulated = tl.full((BM, 128), 0.0, tl.float32)
    for start in range(0, end, BN):
        column = start + columns
        k = tl.load(K + kv_base + column[None, :] * 1024 + dimension[:, None],
                    column[None, :] < S, other=0.0)
        product = tl.dot(q, k).to(tl.bfloat16).to(tl.float32)
        score = (product * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)
        score = tl.where((column[None, :] <= sequence[:, None]) & (column[None, :] < S),
                         score, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(score, axis=1))
        probability = tl.exp2((score - next_maximum[:, None]) * 1.4426950408889634)
        correction = tl.exp2((maximum - next_maximum) * 1.4426950408889634)
        normalizer = normalizer * correction + tl.sum(probability, axis=1)
        if not TWO_PASS:
            accumulated = accumulated * correction[:, None]
            v = tl.load(V + kv_base + column[:, None] * 1024 + dimension[None, :],
                        column[:, None] < S, other=0.0)
            accumulated = tl.dot(probability.to(tl.bfloat16), v, accumulated)
        elif S <= BN:
            # A full sequence tile already has the global normalization.
            probability = (probability / normalizer[:, None]).to(tl.bfloat16)
            v = tl.load(V + kv_base + column[:, None] * 1024 + dimension[None, :],
                        column[:, None] < S, other=0.0)
            accumulated = tl.dot(probability, v, accumulated)
        maximum = next_maximum
    if TWO_PASS and S > BN:
        for start in range(0, end, BN):
            column = start + columns
            k = tl.load(K + kv_base + column[None, :] * 1024 + dimension[:, None],
                        column[None, :] < S, other=0.0)
            product = tl.dot(q, k).to(tl.bfloat16).to(tl.float32)
            score = (product * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)
            score = tl.where((column[None, :] <= sequence[:, None]) & (column[None, :] < S),
                             score, -float("inf"))
            probability = tl.exp2((score - maximum[:, None]) * 1.4426950408889634)
            probability = (probability / normalizer[:, None]).to(tl.bfloat16)
            v = tl.load(V + kv_base + column[:, None] * 1024 + dimension[None, :],
                        column[:, None] < S, other=0.0)
            accumulated = tl.dot(probability, v, accumulated)
    if TWO_PASS:
        result = accumulated
    else:
        result = accumulated / normalizer[:, None]
    tl.store(O + q_offset, result, sequence[:, None] < S)


def attention(query, key, value, block_m=64, block_n=None, group=None, two_pass=True, warps=4):
    batch, sequence = query.shape[:2]
    if block_n is None:
        block_n = 128 if sequence <= 128 or sequence > 512 else 64
    if group is None:
        group = 1 if block_n == 128 else 12
    output = torch.empty_like(query)
    _causal_attention[(triton.cdiv(sequence * group, block_m), batch * 96 // group)](
        query, key, value, output, sequence, block_m, block_n, group, two_pass,
        num_warps=warps, num_stages=1, enable_fp_fusion=False,
    )
    return output


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
    attended = attention(query, key, value).view(batch * sequence, 12288)
    torch.mm(attended, o_proj_weight.t(), out=output.view(batch * sequence, 4096))
