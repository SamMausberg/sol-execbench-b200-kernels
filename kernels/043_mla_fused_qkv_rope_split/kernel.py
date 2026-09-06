# SPDX-License-Identifier: Apache-2.0
"""MLA projections with fused RMSNorm and concrete split views."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm(X, W, Y, ROWS: tl.constexpr, EPS: tl.constexpr,
              WIDTH: tl.constexpr = 1536, BLOCK: tl.constexpr = 2048):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * WIDTH + columns, columns < WIDTH, 0).to(tl.float32)
    variance = tl.sum(values * values, 0) / WIDTH
    normalized = (values * tl.rsqrt(variance + EPS)).to(Y.dtype.element_ty)
    weight = tl.load(W + columns, columns < WIDTH, 0).to(tl.float32)
    # The reference rounds the unweighted normalized value before multiplying
    # the BF16 weight. Preserve both BF16 rounding points.
    result = normalized.to(tl.float32) * weight
    tl.store(Y + row * WIDTH + columns, result, columns < WIDTH)


@torch.no_grad()
def normalize(latent, weight, output, epsilon):
    weight = weight.contiguous()
    _rms_norm[(latent.shape[0],)](
        latent, weight, output, latent.shape[0], epsilon,
        num_warps=4, enable_fp_fusion=False,
    )


def split_views(q, kv, batch, sequence):
    q = q.view(batch, sequence, 128, 192)
    kv = kv.view(batch, sequence, 576)
    return q[..., :128], q[..., 128:], kv[..., :512], kv[..., 512:].unsqueeze(2)


@torch.no_grad()
def run(hidden_states, q_a_proj_weight, q_a_layernorm_weight,
        q_b_proj_weight, kv_a_proj_weight, rms_norm_eps):
    batch, sequence, hidden = hidden_states.shape
    rows = batch * sequence
    x = hidden_states.reshape(rows, hidden)
    latent = torch.empty((rows, 1536), dtype=x.dtype, device=x.device)
    normalized = torch.empty_like(latent)
    q = torch.empty((rows, 24576), dtype=x.dtype, device=x.device)
    kv = torch.empty((rows, 576), dtype=x.dtype, device=x.device)
    torch.mm(x, q_a_proj_weight.T, out=latent)
    normalize(latent, q_a_layernorm_weight, normalized, rms_norm_eps)
    torch.mm(normalized, q_b_proj_weight.T, out=q)
    torch.mm(x, kv_a_proj_weight.T, out=kv)
    return split_views(q, kv, batch, sequence)
