# SPDX-License-Identifier: Apache-2.0
"""Three causal depthwise convolutions and Hyena gating in one launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _convolve(U, W, B, batch, channel, position, valid,
              UB: tl.constexpr, UC: tl.constexpr, US: tl.constexpr,
              WC: tl.constexpr, WK: tl.constexpr, BS: tl.constexpr):
    # PyTorch's native FP32 depthwise convolution starts with the bias,
    # then accumulates valid taps in increasing kernel order.
    value = tl.load(B + channel * BS, valid, other=0)
    start = batch * UB + channel * UC
    for tap in tl.static_range(3):
        source_position = position + tap - 2
        present = valid & (source_position >= 0)
        sample = tl.load(U + start + source_position * US, present, other=0)
        weight = tl.load(W + channel * WC + tap * WK, valid, other=0)
        value = tl.where(present, tl.fma(sample, weight, value), value)
    return value


@triton.jit
def _hyena(U, W, B, G, X0, X1,
           SIZE: tl.constexpr, SEQ: tl.constexpr,
           UB: tl.constexpr, UC: tl.constexpr, US: tl.constexpr,
           WC: tl.constexpr, WK: tl.constexpr, BS: tl.constexpr,
           GB: tl.constexpr, GC: tl.constexpr, GS: tl.constexpr,
           A0B: tl.constexpr, A0C: tl.constexpr, A0S: tl.constexpr,
           A1B: tl.constexpr, A1C: tl.constexpr, A1S: tl.constexpr,
           BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < SIZE
    position = index % SEQ
    channel = (index // SEQ) % 256
    batch = index // (SEQ * 256)
    x0 = _convolve(U, W, B, batch, channel, position, valid, UB, UC, US, WC, WK, BS)
    x1 = _convolve(U, W, B, batch, channel + 256, position, valid, UB, UC, US, WC, WK, BS)
    v = _convolve(U, W, B, batch, channel + 512, position, valid, UB, UC, US, WC, WK, BS)
    tl.store(X0 + batch * A0B + channel * A0C + position * A0S, x0, valid)
    tl.store(X1 + batch * A1B + channel * A1C + position * A1S, x1, valid)
    tl.store(G + batch * GB + channel * GC + position * GS, v * x0, valid)


def configured_run(u, short_filter_weight, short_filter_bias, v_gated, x0, x1,
                   block=512, warps=4):
    size = u.shape[0] * 256 * u.shape[2]
    _hyena[(triton.cdiv(size, block),)](
        u, short_filter_weight, short_filter_bias, v_gated, x0, x1,
        size, u.shape[2], *u.stride(), short_filter_weight.stride(0),
        short_filter_weight.stride(2), short_filter_bias.stride(0),
        *v_gated.stride(), *x0.stride(), *x1.stride(), block,
        num_warps=warps, enable_fp_fusion=False,
    )


@torch.no_grad()
def run(u, short_filter_weight, short_filter_bias, v_gated, x0, x1):
    configured_run(u, short_filter_weight, short_filter_bias, v_gated, x0, x1)
