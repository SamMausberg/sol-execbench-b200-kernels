# SPDX-License-Identifier: Apache-2.0
"""Load both channel halves and fuse tanh GELU with the gate product."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _geglu(X, Y, D: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // D
    col = offset % D
    valid = offset < N
    x = tl.load(X + row * (2 * D) + col, valid, 0)
    linear = tl.load(X + row * (2 * D) + col + D, valid, 0)
    cube = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * cube)
    gelu = (0.5 * x) * (1.0 + libdevice.tanh(inner))
    tl.store(Y + offset, gelu * linear, valid)


@triton.jit
def _geglu_rows(X, Y, D: tl.constexpr, BLOCK: tl.constexpr, FAST: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = col < D
    x = tl.load(X + row * (2 * D) + col, valid, 0)
    linear = tl.load(X + row * (2 * D) + col + D, valid, 0)
    cube = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * cube)
    if FAST == 1:
        tanh = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;", constraints="=f,f", args=[inner],
            dtype=tl.float32, is_pure=True, pack=1,
        )
    elif FAST == 2:
        tanh = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    else:
        tanh = libdevice.tanh(inner)
    gelu = (0.5 * x) * (1.0 + tanh)
    tl.store(Y + row * D + col, gelu * linear, valid)


def run(x, output):
    d = x.shape[-1] // 2
    n = output.numel()
    rows = n // d
    if rows <= 512:
        block = 2048
        _geglu_rows[(rows, triton.cdiv(d, block))](
            x, output, d, block, 2, num_warps=8, enable_fp_fusion=False,
        )
    else:
        block = 512 if rows <= 2048 else 2048
        _geglu[(triton.cdiv(n, block),)](
            x, output, d, n, block, num_warps=4, enable_fp_fusion=False,
        )
