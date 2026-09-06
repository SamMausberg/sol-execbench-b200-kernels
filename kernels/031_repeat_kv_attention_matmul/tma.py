# SPDX-License-Identifier: Apache-2.0
"""Persistent QK tiles with tensor-map loads and ordinary masked stores."""

import triton
import triton.language as tl


@triton.jit
def _qk_tma(
    Q, K, O, SEQ: tl.constexpr, HEADS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    PROGRAMS: tl.constexpr, STREAM: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr = False,
):
    mt: tl.constexpr = triton.cdiv(SEQ, BM)
    nt: tl.constexpr = triton.cdiv(SEQ, BN)
    for tile in tl.range(
        tl.program_id(0), HEADS * mt * nt, PROGRAMS,
        warp_specialize=WARP_SPECIALIZE,
    ):
        head = tile // (mt * nt)
        tile_m = tile % mt
        tile_n = (tile // mt) % nt
        q_row = head * SEQ + tile_m * BM
        k_row = head * SEQ + tile_n * BN
        acc = tl.full((BM, BN), 0, tl.float32)
        for start in range(triton.cdiv(128, BK)):
            q = Q.load([q_row, start * BK])
            k = K.load([k_row, start * BK])
            acc = tl.dot(q, k.T, acc)
        m = tile_m * BM + tl.arange(0, BM)
        n = tile_n * BN + tl.arange(0, BN)
        offsets = (head * SEQ + m[:, None]) * SEQ + n[None, :]
        tl.store(
            O + offsets, acc * (128 ** -0.5),
            (m[:, None] < SEQ) & (n[None, :] < SEQ),
            cache_modifier=".cs" if STREAM else "",
        )
