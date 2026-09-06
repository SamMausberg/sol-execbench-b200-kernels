# SPDX-License-Identifier: Apache-2.0
"""Normalize row fragments with fewer warp lanes per row."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_split(
    Q, K, WQ, WK, OQ, OK, EPS: tl.constexpr, ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, STREAM: tl.constexpr = False,
    PARTS: tl.constexpr = 2,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, 128 // PARTS)
    if tl.program_id(1) == 0:
        source, weight, output = Q, WQ, OQ
    else:
        source, weight, output = K, WK, OK
    offsets = row[:, None] * 128 + col[None, :]
    weights = (row[:, None] % 48) * 128 + col[None, :]
    squares = tl.full((BLOCK_ROWS, 128 // PARTS), 0.0, tl.float32)
    values = ()
    for part in tl.static_range(PARTS):
        x = tl.load(source + offsets + part * (128 // PARTS), row[:, None] < ROWS, 0, cache_modifier=".cg" if STREAM else "")
        values += (x,)
        squares = squares + x * x
    inverse_rms = tl.rsqrt(tl.sum(squares, 1) * (1.0 / 128) + EPS)
    for part in tl.static_range(PARTS):
        w = tl.load(weight + weights + part * (128 // PARTS))
        norm = values[part] * inverse_rms[:, None]
        tl.store(output + offsets + part * (128 // PARTS), norm * w, row[:, None] < ROWS, cache_modifier=".cs" if STREAM else "")
