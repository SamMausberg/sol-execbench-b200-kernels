# SPDX-License-Identifier: Apache-2.0
"""Experimental tensor-map loads with grouped, persistent attention-score tiles."""

import triton
import triton.language as tl


@triton.jit
def grouped_qk_tma(
    Q, K, O, SCALE: tl.constexpr,
    SEQ: tl.constexpr, DIM: tl.constexpr, GROUP: tl.constexpr,
    KV_HEADS: tl.constexpr, BATCH: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    PROGRAMS: tl.constexpr, STREAM: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr = False,
):
    mt: tl.constexpr = triton.cdiv(GROUP * SEQ, BM)
    nt: tl.constexpr = triton.cdiv(SEQ, BN)
    for tile in tl.range(tl.program_id(0), BATCH * KV_HEADS * mt * nt, PROGRAMS,
                         warp_specialize=WARP_SPECIALIZE):
        group_id = tile // (mt * nt)
        tile_m = tile % mt
        tile_n = (tile // mt) % nt
        q_row = group_id * GROUP * SEQ + tile_m * BM
        k_row = group_id * SEQ + tile_n * BN
        acc = tl.full((BM, BN), 0, tl.float32)
        for start in range(triton.cdiv(DIM, BK)):
            q = Q.load([q_row, start * BK])
            k = K.load([k_row, start * BK])
            acc = tl.dot(q, k.T, acc)
        m = tile_m * BM + tl.arange(0, BM)
        n = tile_n * BN + tl.arange(0, BN)
        out_offset = (group_id * GROUP * SEQ + m[:, None]) * SEQ + n[None, :]
        tl.store(
            O + out_offset, acc * SCALE,
            (m[:, None] < GROUP * SEQ) & (n[None, :] < SEQ),
            cache_modifier=".cs" if STREAM else "",
        )
