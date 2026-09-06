# SPDX-License-Identifier: Apache-2.0
"""SiLU backward from the saved sigmoid, fused into one memory pass."""

import triton
import triton.language as tl


@triton.jit
def _silu_backward(DY, X, SIGMOID, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    dy = tl.load(DY + offset, valid, 0)
    x = tl.load(X + offset, valid, 0)
    sigmoid = tl.load(SIGMOID + offset, valid, 0)
    bracket = 1.0 + x * (1.0 - sigmoid)
    result = dy * (sigmoid * bracket)
    tl.store(DX + offset, result, valid)


def run(grad_output, x, sigmoid_x, grad_input):
    n = x.numel()
    block = 256 if n < 65536 else 1024
    _silu_backward[(triton.cdiv(n, block),)](
        grad_output, x, sigmoid_x, grad_input, n, block,
        num_warps=4, enable_fp_fusion=False,
    )
