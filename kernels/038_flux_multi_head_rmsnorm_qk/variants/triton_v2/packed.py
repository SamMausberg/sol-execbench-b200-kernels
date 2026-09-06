# SPDX-License-Identifier: Apache-2.0
"""B200 packed FP32 arithmetic with vectorized short rows."""

import triton
import triton.language as tl


@triton.jit
def _mul_pair(a, b):
    return tl.inline_asm_elementwise(
        """{
        .reg .b64 a, b, result;
        mov.b64 a, {$2, $3};
        mov.b64 b, {$4, $5};
        mul.f32x2 result, a, b;
        mov.b64 {$0, $1}, result;
        }""",
        constraints="=f,=f,f,f,f,f",
        args=[a, b], dtype=tl.float32, is_pure=True, pack=2,
    )


@triton.jit
def _rmsnorm_packed(
    Q, K, WQ, WK, OQ, OK, EPS: tl.constexpr, ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, STREAM: tl.constexpr = False,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, 128)
    if tl.program_id(1) == 0:
        source, weight, output = Q, WQ, OQ
    else:
        source, weight, output = K, WK, OK
    offsets = row[:, None] * 128 + col[None, :]
    x = tl.load(source + offsets, row[:, None] < ROWS, 0, cache_modifier=".cg" if STREAM else "")
    variance = tl.sum(_mul_pair(x, x), 1) * (1.0 / 128)
    inverse_rms = tl.rsqrt(variance + EPS)
    w = tl.load(weight + (row[:, None] % 48) * 128 + col[None, :])
    norm = _mul_pair(x, inverse_rms[:, None])
    result = _mul_pair(norm, w)
    tl.store(output + offsets, result, row[:, None] < ROWS, cache_modifier=".cs" if STREAM else "")
