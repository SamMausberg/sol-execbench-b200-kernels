# SPDX-License-Identifier: Apache-2.0
"""Loop over row tiles to limit CTA scheduling overhead on large tensors."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_persistent(
    Q, K, WQ, WK, OQ, OK, EPS: tl.constexpr, ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, STREAM: tl.constexpr = False,
    UNROLL: tl.constexpr = 1,
):
    col = tl.arange(0, 128)
    if tl.program_id(1) == 0:
        source, weight, output = Q, WQ, OQ
    else:
        source, weight, output = K, WK, OK
    for tile in tl.range(tl.program_id(0), tl.cdiv(ROWS, BLOCK_ROWS), tl.num_programs(0), loop_unroll_factor=UNROLL):
        row = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offsets = row[:, None] * 128 + col[None, :]
        x = tl.load(source + offsets, row[:, None] < ROWS, 0, cache_modifier=".cg" if STREAM else "")
        variance = tl.sum(x * x, 1) * (1.0 / 128)
        inverse_rms = tl.rsqrt(variance + EPS)
        w = tl.load(weight + (row[:, None] % 48) * 128 + col[None, :])
        norm = x * inverse_rms[:, None]
        result = norm * w
        tl.store(output + offsets, result, row[:, None] < ROWS, cache_modifier=".cs" if STREAM else "")
