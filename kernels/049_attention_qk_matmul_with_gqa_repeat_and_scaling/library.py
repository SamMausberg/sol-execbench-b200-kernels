# SPDX-License-Identifier: Apache-2.0
"""Grouped library GEMM with scaling passed directly to its epilogue."""

import torch


@torch.no_grad()
def run(query, key, scaling, attn_scores):
    batch, heads, seq, dim = query.shape
    kv_heads = key.shape[1]
    group = heads // kv_heads
    q = query.reshape(batch * kv_heads, group * seq, dim)
    k = key.reshape(batch * kv_heads, seq, dim).transpose(1, 2)
    out = attn_scores.view(batch * kv_heads, group * seq, seq)
    torch.baddbmm(out, q, k, beta=0, alpha=scaling, out=out)
