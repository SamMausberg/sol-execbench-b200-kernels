# SPDX-License-Identifier: Apache-2.0
"""Choose grouped GEMM implementations by output alignment and launch size."""

import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from kernel import _grouped_qk
from library import run as library_run
from tma import grouped_qk_tma


def run(query, key, scaling, attn_scores):
    batch, heads, seq, dim = query.shape
    kv_heads = key.shape[1]
    group = heads // kv_heads
    if batch * seq <= 256 or (seq < 512 and seq % 8):
        bm = 16 if seq <= 128 else 32
        bn, bk = 32, 64
        _grouped_qk[(triton.cdiv(group * seq, bm), triton.cdiv(seq, bn), batch * kv_heads)](
            query, key, attn_scores, scaling, seq, dim, group, kv_heads,
            *query.stride(), *key.stride(), bm, bn, bk,
            num_warps=4, num_stages=3,
        )
    elif seq % 8 and query.is_contiguous() and key.is_contiguous():
        # Odd output row strides prevent an efficient aligned library epilogue.
        bm, bn, bk = 128, 128, 64
        qdesc = TensorDescriptor(query, [batch * heads * seq, dim], [dim, 1], [bm, bk])
        kdesc = TensorDescriptor(key, [batch * kv_heads * seq, dim], [dim, 1], [bn, bk])
        tiles = triton.cdiv(group * seq, bm) * triton.cdiv(seq, bn) * batch * kv_heads
        programs = min(296, tiles)
        grouped_qk_tma[(programs,)](
            qdesc, kdesc, attn_scores, scaling, seq, dim, group, kv_heads, batch,
            bm, bn, bk, programs, True, num_warps=4, num_stages=3,
        )
    else:
        library_run(query, key, scaling, attn_scores)
