"""Fused Gaussian sparse activation matching the reference FP32 quantile."""

import math
import struct

import triton
import triton.language as tl


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _horner_f32(coefficients, value):
    result = _f32(coefficients[0])
    for coefficient in coefficients[1:]:
        result = _f32(_f32(result * value) + _f32(coefficient))
    return result


def _normal_quantile_f32(probability):
    """Evaluate the reference rational approximation from a Python scalar.

    The reference launches separate FP32 PyTorch operations. Explicit rounding
    here preserves that arithmetic, including the supplied scalar's conversion
    to FP32. This calculation depends only on the current scalar argument.
    """
    p = _f32(probability)
    if not 0.0 < p < 1.0:
        return float("nan")
    if p < _f32(0.02425) or p > _f32(1.0 - 0.02425):
        tail = p if p < _f32(0.02425) else _f32(1.0 - p)
        q = _f32(math.sqrt(_f32(-2.0 * _f32(math.log(tail)))))
        numerator = _horner_f32(
            (-7.784894002430293e-3, -3.223964580411365e-1,
             -2.400758277161838, -2.549732539343734,
             4.374664141464968, 2.938163982698783), q
        )
        denominator = _horner_f32(
            (7.784695709041462e-3, 3.224671290700398e-1,
             2.445134137142996, 3.754408661907416, 1.0), q
        )
        if p > _f32(1.0 - 0.02425):
            numerator = -numerator
    else:
        q = _f32(p - 0.5)
        r = _f32(q * q)
        numerator = _f32(_horner_f32(
            (-3.969683028665376e1, 2.209460984245205e2,
             -2.759285104469687e2, 1.383577518672690e2,
             -3.066479806614716e1, 2.506628277459239), r
        ) * q)
        denominator = _horner_f32(
            (-5.447609879822406e1, 1.615858368580409e2,
             -1.556989798598866e2, 6.680131188771972e1,
             -1.328068155288572e1, 1.0), r
        )
    return _f32(numerator / denominator)


@triton.jit
def _sum_pair(x, xx, other_x, other_xx):
    return x + other_x, xx + other_xx


@triton.jit
def _activation(
    X, Y, Z: tl.constexpr,
    N: tl.constexpr, S: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr,
    YS0: tl.constexpr, YS1: tl.constexpr, YS2: tl.constexpr,
    BLOCK: tl.constexpr,
    MOMENTS: tl.constexpr = False,
    LOAD_CACHE: tl.constexpr = "", STORE_CACHE: tl.constexpr = "",
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    x_offset = (row // S) * XS0 + (row % S) * XS1 + column * XS2
    y_offset = (row // S) * YS0 + (row % S) * YS1 + column * YS2
    valid = column < N
    x = tl.load(X + x_offset, valid, other=0, cache_modifier=LOAD_CACHE).to(tl.float32)
    if MOMENTS:
        total, total_square = tl.reduce((x, x * x), 0, _sum_pair)
        mean = total / N
        mean_square = total_square / N
        variance = mean_square - mean * mean
        # Recompute centered variance when subtracting moments would amplify
        # rounding errors. This also handles constant and strongly shifted rows.
        if not (variance > 0.0625 * mean_square):
            centered = tl.where(valid, x - mean, 0.0)
            variance = tl.sum(centered * centered, 0) / N
    else:
        mean = tl.sum(x, 0) / N
        centered = tl.where(valid, x - mean, 0.0)
        variance = tl.sum(centered * centered, 0) / N
    threshold = mean + tl.sqrt(variance) * Z
    result = tl.maximum(x - threshold, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(Y + y_offset, result, valid, cache_modifier=STORE_CACHE)


@triton.jit
def _copy(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, values, offsets < N)


def run(inputs, target_sparsity, output):
    if target_sparsity == 0.0:
        if inputs.is_contiguous() and output.is_contiguous():
            count = inputs.numel()
            _copy[(triton.cdiv(count, 4096),)](inputs, output, count, 4096)
        else:
            output.copy_(inputs)
        return
    batch, sequence, width = inputs.shape
    block = triton.next_power_of_2(width)
    warps = 4 if block <= 4096 else 8
    multiplier = _normal_quantile_f32(target_sparsity)
    _activation[(batch * sequence,)](
        inputs, output, multiplier, width, sequence,
        *inputs.stride(), *output.stride(), block,
        MOMENTS=width == 12288 and batch * sequence <= 1024,
        num_warps=warps, enable_fp_fusion=False,
    )
