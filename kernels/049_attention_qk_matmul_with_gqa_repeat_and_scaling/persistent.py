# SPDX-License-Identifier: Apache-2.0
"""Experimental persistent grouped-QK schedule for Blackwell tensor memory reuse."""

import triton
import triton.language as tl


@triton.jit
def grouped_qk_persistent(
    Q, K, O, SCALE: tl.constexpr,
    SEQ: tl.constexpr, DIM: tl.constexpr, GROUP: tl.constexpr,
    KV_HEADS: tl.constexpr, BATCH: tl.constexpr,
    QB: tl.constexpr, QH: tl.constexpr, QS: tl.constexpr, QD: tl.constexpr,
    KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr, KD: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    STREAM: tl.constexpr,
):
    mt: tl.constexpr = triton.cdiv(GROUP * SEQ, BM)
    nt: tl.constexpr = triton.cdiv(SEQ, BN)
    for tile in range(tl.program_id(0), BATCH * KV_HEADS * mt * nt, tl.num_programs(0)):
        group_id = tile // (mt * nt)
        batch = group_id // KV_HEADS
        kv_head = group_id % KV_HEADS
        m = (tile % mt) * BM + tl.arange(0, BM)
        n = ((tile // mt) % nt) * BN + tl.arange(0, BN)
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
                cache_modifier=".cg" if STREAM else "",
            )
            k = tl.load(
                k_base + col[:, None] * KD,
                (n[None, :] < SEQ) & (col[:, None] < DIM), 0,
                cache_modifier=".cg" if STREAM else "",
            )
            acc = tl.dot(q, k, acc)
        value = acc * SCALE
        out_offset = (group_id * GROUP * SEQ + m[:, None]) * SEQ + n[None, :]
        tl.store(
            O + out_offset, value, (m[:, None] < GROUP * SEQ) & (n[None, :] < SEQ),
            cache_modifier=".cs" if STREAM else "",
        )
