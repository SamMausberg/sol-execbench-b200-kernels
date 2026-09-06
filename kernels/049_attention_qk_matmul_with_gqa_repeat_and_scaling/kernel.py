# SPDX-License-Identifier: Apache-2.0
"""Grouped attention scores without materializing repeated key heads."""

import triton
import triton.language as tl


@triton.jit
def _grouped_qk(
    Q, K, O, SCALE: tl.constexpr,
    SEQ: tl.constexpr, DIM: tl.constexpr, GROUP: tl.constexpr,
    KV_HEADS: tl.constexpr,
    QB: tl.constexpr, QH: tl.constexpr, QS: tl.constexpr, QD: tl.constexpr,
    KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr, KD: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # A group's consecutive query heads form one tall matrix sharing the same K.
    group_id = tl.program_id(2)
    batch = group_id // KV_HEADS
    kv_head = group_id % KV_HEADS
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    d = tl.arange(0, BK)
    q_head = kv_head * GROUP + m // SEQ
    q_row = m % SEQ
    q_base = Q + batch * QB + q_head[:, None] * QH + q_row[:, None] * QS
    k_base = K + batch * KB + kv_head * KH + n[None, :] * KS
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(triton.cdiv(DIM, BK)):
        col = start * BK + d
        q = tl.load(
            q_base + col[None, :] * QD,
            (m[:, None] < GROUP * SEQ) & (col[None, :] < DIM), 0,
        )
        k = tl.load(
            k_base + col[:, None] * KD,
            (n[None, :] < SEQ) & (col[:, None] < DIM), 0,
        )
        acc = tl.dot(q, k, acc)
    value = acc * SCALE
    out_offset = (group_id * GROUP * SEQ + m[:, None]) * SEQ + n[None, :]
    tl.store(O + out_offset, value, (m[:, None] < GROUP * SEQ) & (n[None, :] < SEQ))


def run(query, key, scaling, attn_scores):
    batch, heads, seq, dim = query.shape
    kv_heads = key.shape[1]
    group = heads // kv_heads
    bm, bn, bk = 64, 64, 64
    _grouped_qk[(triton.cdiv(group * seq, bm), triton.cdiv(seq, bn), batch * kv_heads)](
        query, key, attn_scores, scaling, seq, dim, group, kv_heads,
        *query.stride(), *key.stride(), bm, bn, bk,
        num_warps=4, num_stages=3,
    )
