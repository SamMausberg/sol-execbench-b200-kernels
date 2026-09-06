# SPDX-License-Identifier: Apache-2.0
"""Per-head BF16 QK scores with FP32 accumulation and fused scaling."""

import triton
import triton.language as tl
import torch
from triton.tools.tensor_descriptor import TensorDescriptor

from tma import _qk_tma


@triton.jit
def _qk_scores(
    Q, K, O, SEQ: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    STREAM: tl.constexpr = False,
):
    head = tl.program_id(2)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    d = tl.arange(0, BK)
    q_base = Q + head * SEQ * 128 + m[:, None] * 128
    k_base = K + head * SEQ * 128 + n[None, :] * 128
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(triton.cdiv(128, BK)):
        col = start * BK + d
        q = tl.load(
            q_base + col[None, :],
            (m[:, None] < SEQ) & (col[None, :] < 128), 0,
            cache_modifier=".cg" if STREAM else "",
        )
        k = tl.load(
            k_base + col[:, None],
            (n[None, :] < SEQ) & (col[:, None] < 128), 0,
            cache_modifier=".cg" if STREAM else "",
        )
        acc = tl.dot(q, k, acc)
    result = acc * (128 ** -0.5)
    offsets = head * SEQ * SEQ + m[:, None] * SEQ + n[None, :]
    tl.store(
        O + offsets, result,
        (m[:, None] < SEQ) & (n[None, :] < SEQ),
        cache_modifier=".cs" if STREAM else "",
    )


@torch.no_grad()
def run(query, key, attn_weights):
    batch, heads, seq, _ = query.shape
    if seq % 8 == 0:
        q = query.view(batch * heads, seq, 128)
        k = key.view(batch * heads, seq, 128).transpose(1, 2)
        out = attn_weights.view(batch * heads, seq, seq)
        torch.baddbmm(out, q, k, beta=0, alpha=128 ** -0.5, out=out)
    elif seq <= 256:
        bm, bn, bk = 32, 64, 128
        _qk_scores[
            (triton.cdiv(seq, bm), triton.cdiv(seq, bn), batch * heads)
        ](
            query, key, attn_weights, seq, bm, bn, bk,
            num_warps=4, num_stages=2,
        )
    else:
        bm, bn, bk = 128, 128, 128
        programs = min(
            148, batch * heads * triton.cdiv(seq, bm) * triton.cdiv(seq, bn),
        )
        shape = [batch * heads * seq, 128]
        q_desc = TensorDescriptor(query, shape, [128, 1], [bm, bk])
        k_desc = TensorDescriptor(key, shape, [128, 1], [bn, bk])
        _qk_tma[(programs,)](
            q_desc, k_desc, attn_weights, seq, batch * heads,
            bm, bn, bk, programs, True, num_warps=4, num_stages=2,
        )
