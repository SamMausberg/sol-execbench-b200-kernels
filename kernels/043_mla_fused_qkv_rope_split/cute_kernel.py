# SPDX-License-Identifier: Apache-2.0
"""Grouped initial MLA projections with fused norm and concrete split views."""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import HardwareInfo

from cute_mla import SM100GroupedMla
from kernel import normalize, split_views, run as run_library

_compiled = {}


def matrix(tensor):
    rows, columns = tensor.shape
    return from_dlpack(tensor.view(1, rows, columns).permute(1, 2, 0), assumed_align=16)


@torch.no_grad()
def project(x, q_weight, kv_weight, latent, kv, bm=128, bn=128,
            cluster_m=1, two_cta=False):
    a, bq, bk, dq, dk = [matrix(t) for t in (x, q_weight, kv_weight, latent, kv)]
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    key = (x.shape[0], x.device.index, bm, bn, cluster_m, two_cta)
    compiled = _compiled.get(key)
    if compiled is None:
        gemm = SM100GroupedMla(cutlass.Float32, cutlass.Float32, two_cta,
                              (bm, bn), (cluster_m, 1))
        clusters = HardwareInfo().get_max_active_clusters(cluster_m)
        compiled = cute.compile(gemm, a, bq, dq, dq, bk, dk, dk,
                                1.0, 0.0, clusters, stream)
        _compiled[key] = compiled
    compiled(a, bq, dq, dq, bk, dk, dk, 1.0, 0.0, stream)


@torch.no_grad()
def run(hidden_states, q_a_proj_weight, q_a_layernorm_weight,
        q_b_proj_weight, kv_a_proj_weight, rms_norm_eps):
    inputs = (hidden_states, q_a_proj_weight, q_a_layernorm_weight,
              q_b_proj_weight, kv_a_proj_weight)
    if any(not t.is_contiguous() or t.data_ptr() % 16 for t in inputs):
        return run_library(*inputs, rms_norm_eps)
    batch, sequence, hidden = hidden_states.shape
    rows = batch * sequence
    x = hidden_states.reshape(rows, hidden)
    latent = torch.empty((rows, 1536), dtype=x.dtype, device=x.device)
    normalized = torch.empty_like(latent)
    q = torch.empty((rows, 24576), dtype=x.dtype, device=x.device)
    kv = torch.empty((rows, 576), dtype=x.dtype, device=x.device)
    config = (64, 64, 1, False) if rows <= 256 else (128, 128, 1, False)
    project(x, q_a_proj_weight, kv_a_proj_weight, latent, kv, *config)
    normalize(latent, q_a_layernorm_weight, normalized, rms_norm_eps)
    torch.mm(normalized, q_b_proj_weight.T, out=q)
    return split_views(q, kv, batch, sequence)
