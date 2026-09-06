"""Paired FP32 RoPE rotation with shared coefficients for query and key."""

import triton
import triton.language as tl


@triton.jit
def _rope_pairs(
    query,
    key,
    cos,
    sin,
    query_out,
    key_out,
    Q_PAIRS: tl.constexpr,
    K_PAIRS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = pair // HALF_DIM
    channel = pair % HALF_DIM
    first = row * (2 * HALF_DIM) + channel
    second = first + HALF_DIM
    coefficient = (row % SEQ_LEN) * (2 * HALF_DIM) + channel
    active = pair < Q_PAIRS

    c0 = tl.load(cos + coefficient, active, other=0.0)
    c1 = tl.load(cos + coefficient + HALF_DIM, active, other=0.0)
    s0 = tl.load(sin + coefficient, active, other=0.0)
    s1 = tl.load(sin + coefficient + HALF_DIM, active, other=0.0)

    q0 = tl.load(query + first, active, other=0.0)
    q1 = tl.load(query + second, active, other=0.0)
    tl.store(query_out + first, q0 * c0 - q1 * s0, active)
    tl.store(query_out + second, q1 * c1 + q0 * s1, active)

    # Query and key have the same sequence/dimension layout. Their flattened
    # offsets therefore use the same cosine/sine entries even with GQA heads.
    if tl.program_id(0) * BLOCK < K_PAIRS:
        k_active = pair < K_PAIRS
        k0 = tl.load(key + first, k_active, other=0.0)
        k1 = tl.load(key + second, k_active, other=0.0)
        tl.store(key_out + first, k0 * c0 - k1 * s0, k_active)
        tl.store(key_out + second, k1 * c1 + k0 * s1, k_active)


def run(query, key, cos, sin, query_rotated, key_rotated):
    q_pairs = query.numel() // 2
    k_pairs = key.numel() // 2
    block = 256 if q_pairs < 262144 else 1024
    _rope_pairs[(triton.cdiv(q_pairs, block),)](
        query,
        key,
        cos,
        sin,
        query_rotated,
        key_rotated,
        q_pairs,
        k_pairs,
        query.shape[-2],
        query.shape[-1] // 2,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
