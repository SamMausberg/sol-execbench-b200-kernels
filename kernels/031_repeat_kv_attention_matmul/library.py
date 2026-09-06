# SPDX-License-Identifier: Apache-2.0
"""Library comparison with FP32 scale fused into the GEMM epilogue."""

import torch


@torch.no_grad()
def run(query, key, attn_weights):
    batch, heads, seq, dim = query.shape
    q = query.view(batch * heads, seq, dim)
    k = key.view(batch * heads, seq, dim).transpose(1, 2)
    out = attn_weights.view(batch * heads, seq, seq)
    torch.baddbmm(out, q, k, beta=0, alpha=128 ** -0.5, out=out)
