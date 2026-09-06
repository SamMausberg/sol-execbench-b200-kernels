# SPDX-License-Identifier: Apache-2.0
"""Tanh GELU with FP32 intermediates matching the source expression."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gelu(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    x = tl.load(X + offset, valid, 0)
    cube = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * cube)
    result = (0.5 * x) * (1.0 + libdevice.tanh(inner))
    tl.store(Y + offset, result, valid)


def run(x, output):
    n = x.numel()
    block = 1024
    _gelu[(triton.cdiv(n, block),)](
        x, output, n, block, num_warps=4, enable_fp_fusion=False,
    )
