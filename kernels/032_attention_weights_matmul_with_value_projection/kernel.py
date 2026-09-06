# SPDX-License-Identifier: Apache-2.0
"""Attention/value products stored directly in the requested sequence layout."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _av_plain(A, V, O, SEQ: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    combined_head = tl.program_id(2)
    batch = combined_head // 40
    head = combined_head % 40
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a_base = A + combined_head * SEQ * SEQ + m[:, None] * SEQ
    v_base = V + combined_head * SEQ * 128 + n[None, :]
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(triton.cdiv(SEQ, BK)):
        inner = block * BK + k
        a = tl.load(a_base + inner[None, :],
                    (m[:, None] < SEQ) & (inner[None, :] < SEQ), 0)
        v = tl.load(v_base + inner[:, None] * 128,
                    (inner[:, None] < SEQ) & (n[None, :] < 128), 0)
        accumulator = tl.dot(a, v, accumulator)
    offsets = (batch * SEQ + m[:, None]) * 5120 + head * 128 + n[None, :]
    tl.store(O + offsets, accumulator, (m[:, None] < SEQ) & (n[None, :] < 128))


@triton.jit
def _av_tma(A, V, O, OD, SEQ: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            TMA_STORE: tl.constexpr):
    combined_head = tl.program_id(2)
    batch = combined_head // 40
    head = combined_head % 40
    row = tl.program_id(0) * BM
    column = tl.program_id(1) * BN
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(triton.cdiv(SEQ, BK)):
        a = A.load([combined_head * SEQ + row, block * BK])
        v = V.load([combined_head * SEQ + block * BK, column])
        accumulator = tl.dot(a, v, accumulator)
    if TMA_STORE:
        OD.store([batch * SEQ + row, head * 128 + column], accumulator.to(tl.bfloat16))
    else:
        m = row + tl.arange(0, BM)
        n = column + tl.arange(0, BN)
        offsets = (batch * SEQ + m[:, None]) * 5120 + head * 128 + n[None, :]
        tl.store(O + offsets, accumulator, (m[:, None] < SEQ) & (n[None, :] < 128))


@torch.no_grad()
def launch(attn_weights, value_states, output, bm=64, bn=128, bk=64,
           warps=4, stages=2, tma=False, tma_store=False, mma_v2=False):
    batch, heads, sequence, _ = attn_weights.shape
    grid = (triton.cdiv(sequence, bm), triton.cdiv(128, bn), batch * heads)
    options = dict(num_warps=warps, num_stages=stages)
    if mma_v2:
        options["arch"] = "sm80"
    if tma:
        # Head boundaries align to the tile dimensions for this path. This
        # prevents a descriptor tile from reading values from the next head.
        assert sequence % bm == 0 and sequence % bk == 0
        a = TensorDescriptor(attn_weights, [batch * heads * sequence, sequence],
                             [sequence, 1], [bm, bk])
        v = TensorDescriptor(value_states, [batch * heads * sequence, 128],
                             [128, 1], [bk, bn])
        out = TensorDescriptor(output, [batch * sequence, 5120], [5120, 1], [bm, bn])
        _av_tma[grid](a, v, output, out, sequence, bm, bn, bk, tma_store, **options)
    else:
        _av_plain[grid](attn_weights, value_states, output, sequence, bm, bn, bk, **options)


@torch.no_grad()
def run(attn_weights, value_states):
    batch, heads, sequence, _ = attn_weights.shape
    if any(not tensor.is_contiguous() or tensor.data_ptr() % 16
           for tensor in (attn_weights, value_states)):
        return torch.matmul(attn_weights, value_states).transpose(1, 2).reshape(batch, sequence, 5120)
    output = torch.empty((batch, sequence, 5120), dtype=attn_weights.dtype, device=attn_weights.device)
    if sequence % 128 == 0:
        launch(attn_weights, value_states, output, 64, 128, 64, 4, 2, True)
    else:
        launch(attn_weights, value_states, output, 32, 64, 64, 4, 3, False, False, True)
    return output
