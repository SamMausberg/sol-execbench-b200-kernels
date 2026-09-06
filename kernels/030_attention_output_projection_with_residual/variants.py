# SPDX-License-Identifier: Apache-2.0
"""Persistent projection schedules, preserving the BF16 intermediate result."""

import triton
import triton.language as tl


@triton.jit
def projection_residual_tiled(
    A, W, R, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    PROGRAMS: tl.constexpr, WS: tl.constexpr, GROUP: tl.constexpr,
    SUBTILE: tl.constexpr,
):
    mt: tl.constexpr = triton.cdiv(M, BM)
    tile = tl.program_id(0)
    row = (tile % mt) * BM
    col = (tile // mt) * BN
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in tl.range(triton.cdiv(K, BK), warp_specialize=WS):
        a = A.load([row, start * BK])
        w = W.load([col, start * BK])
        acc = tl.dot(a, w.T, acc)
    if SUBTILE:
        halves = acc.reshape(BM, 2, BN // 2).permute(0, 2, 1)
        low, high = tl.split(halves)
        r0 = R.load([row, col]).to(tl.float32)
        c0 = low.to(tl.bfloat16).to(tl.float32) + r0
        O.store([row, col], c0.to(tl.bfloat16))
        r1 = R.load([row, col + BN // 2]).to(tl.float32)
        c1 = high.to(tl.bfloat16).to(tl.float32) + r1
        O.store([row, col + BN // 2], c1.to(tl.bfloat16))
    else:
        residual = R.load([row, col]).to(tl.float32)
        result = acc.to(tl.bfloat16).to(tl.float32) + residual
        O.store([row, col], result.to(tl.bfloat16))


@triton.jit
def _tile_coordinates(tile, mt: tl.constexpr, nt: tl.constexpr, GROUP: tl.constexpr):
    group = tile // (GROUP * nt)
    first_row = group * GROUP
    rows = tl.minimum(mt - first_row, GROUP)
    return first_row + tile % rows, (tile % (GROUP * nt)) // rows


@triton.jit
def projection_residual(
    A, W, R, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    PROGRAMS: tl.constexpr, WS: tl.constexpr, GROUP: tl.constexpr,
    SUBTILE: tl.constexpr,
):
    mt: tl.constexpr = triton.cdiv(M, BM)
    nt: tl.constexpr = triton.cdiv(N, BN)
    # Independent epilogue counter permits overlap across persistent tiles.
    epilogue_tile = tl.program_id(0) - PROGRAMS
    for tile in tl.range(tl.program_id(0), mt * nt, PROGRAMS,
                         flatten=True, warp_specialize=WS):
        rm, rn = _tile_coordinates(tile, mt, nt, GROUP)
        acc = tl.full((BM, BN), 0, tl.float32)
        for start in range(triton.cdiv(K, BK)):
            a = A.load([rm * BM, start * BK])
            w = W.load([rn * BN, start * BK])
            acc = tl.dot(a, w.T, acc)
        epilogue_tile += PROGRAMS
        om, on = _tile_coordinates(epilogue_tile, mt, nt, GROUP)
        if SUBTILE:
            halves = acc.reshape(BM, 2, BN // 2).permute(0, 2, 1)
            low, high = tl.split(halves)
            r0 = R.load([om * BM, on * BN]).to(tl.float32)
            c0 = low.to(tl.bfloat16).to(tl.float32) + r0
            O.store([om * BM, on * BN], c0.to(tl.bfloat16))
            r1 = R.load([om * BM, on * BN + BN // 2]).to(tl.float32)
            c1 = high.to(tl.bfloat16).to(tl.float32) + r1
            O.store([om * BM, on * BN + BN // 2], c1.to(tl.bfloat16))
        else:
            residual = R.load([om * BM, on * BN]).to(tl.float32)
            result = acc.to(tl.bfloat16).to(tl.float32) + residual
            O.store([om * BM, on * BN], result.to(tl.bfloat16))
