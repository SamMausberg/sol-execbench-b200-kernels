# SPDX-License-Identifier: Apache-2.0
"""FP32 tanh GELU using the GPU's native tanh approximation."""

import triton
import triton.language as tl


@triton.jit
def _gelu(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, CACHE: tl.constexpr = ""):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    x = tl.load(X + offset, valid, 0, cache_modifier=CACHE)
    cube = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * cube)
    tanh = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", constraints="=f,f", args=[inner],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    result = (0.5 * x) * (1.0 + tanh)
    tl.store(Y + offset, result, valid)


def run(x, output):
    n = x.numel()
    block, warps = (2048, 4) if n <= 4194304 else (4096, 16)
    _gelu[(triton.cdiv(n, block),)](
        x, output, n, block, num_warps=warps, enable_fp_fusion=False,
    )
