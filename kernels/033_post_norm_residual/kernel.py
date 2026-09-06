"""Fused RMS normalization preserving the BF16 rounding before residual add."""

import triton
import triton.language as tl


@triton.jit
def _post_norm(
    X, R, W, Y,
    EPS: tl.constexpr, N: tl.constexpr, S: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr,
    RS0: tl.constexpr, RS1: tl.constexpr, RS2: tl.constexpr,
    WS: tl.constexpr,
    YS0: tl.constexpr, YS1: tl.constexpr, YS2: tl.constexpr,
    BLOCK: tl.constexpr, PREFETCH_RESIDUAL: tl.constexpr = False,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < N
    x_offset = row // S * XS0 + row % S * XS1 + column * XS2
    r_offset = row // S * RS0 + row % S * RS1 + column * RS2
    y_offset = row // S * YS0 + row % S * YS1 + column * YS2
    x = tl.load(X + x_offset, valid, other=0).to(tl.float32)
    if PREFETCH_RESIDUAL:
        residual = tl.load(R + r_offset, valid, other=0).to(tl.float32)
    variance = tl.sum(x * x, 0) / N
    scaled = x * tl.rsqrt(variance + EPS)
    weight = tl.load(W + column * WS, valid, other=0).to(tl.float32)
    # The reference materializes normalization in BF16 before adding residual.
    normalized = (weight * scaled).to(tl.bfloat16).to(tl.float32)
    if not PREFETCH_RESIDUAL:
        residual = tl.load(R + r_offset, valid, other=0).to(tl.float32)
    tl.store(Y + y_offset, normalized + residual, valid)


def run(sublayer_output, residual, weight, eps, output):
    batch, sequence, width = sublayer_output.shape
    warps = 16 if batch * sequence <= 512 else 4
    _post_norm[(batch * sequence,)](
        sublayer_output, residual, weight, output, eps, width, sequence,
        *sublayer_output.stride(), *residual.stride(), weight.stride(0),
        *output.stride(), triton.next_power_of_2(width),
        PREFETCH_RESIDUAL=True, num_warps=warps, enable_fp_fusion=False,
    )
