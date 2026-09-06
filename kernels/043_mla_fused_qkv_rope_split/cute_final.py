# SPDX-License-Identifier: Apache-2.0
"""Prepare final-projection descriptors before launching the MLA chain."""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.utils import HardwareInfo

from cute_kernel import matrix
from cute_mla import SM100GroupedMla
from kernel import normalize, split_views, run as run_library

_compiled = {}


def prepare(x, weight, output, bm=256, bn=256, cluster_m=2, two_cta=True):
    a, b, d = [matrix(tensor) for tensor in (x, weight, output)]
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    key = (x.shape[0], x.shape[1], weight.shape[0], x.device.index,
           bm, bn, cluster_m, two_cta)
    compiled = _compiled.get(key)
    if compiled is None:
        gemm = SM100GroupedMla(cutlass.Float32, cutlass.Float32, two_cta,
                              (bm, bn), (cluster_m, 1))
        # The single-matrix grid never schedules the second descriptor.
        gemm.second_enabled = False
        clusters = HardwareInfo().get_max_active_clusters(cluster_m)
        compiled = cute.compile(gemm, a, b, d, d, b, d, d,
                                1.0, 0.0, clusters, stream)
        _compiled[key] = compiled
    # This closure is local to this invocation, with current tensor pointers.
    # Only the compiled function is retained in the shape/configuration cache.
    return lambda: compiled(a, b, d, d, b, d, d, 1.0, 0.0, stream)


@torch.no_grad()
def run(hidden_states, q_a_proj_weight, q_a_layernorm_weight,
        q_b_proj_weight, kv_a_proj_weight, rms_norm_eps):
    inputs = (hidden_states, q_a_proj_weight, q_a_layernorm_weight,
              q_b_proj_weight, kv_a_proj_weight)
    if any(not tensor.is_contiguous() or tensor.data_ptr() % 16 for tensor in inputs):
        return run_library(*inputs, rms_norm_eps)
    batch, sequence, hidden = hidden_states.shape
    rows = batch * sequence
    x = hidden_states.reshape(rows, hidden)
    latent = torch.empty((rows, 1536), dtype=x.dtype, device=x.device)
    normalized = torch.empty_like(latent)
    q = torch.empty((rows, 24576), dtype=x.dtype, device=x.device)
    kv = torch.empty((rows, 576), dtype=x.dtype, device=x.device)
    config = (64, 64, 1, False) if rows <= 256 else (256, 256, 2, True)
    final_projection = prepare(normalized, q_b_proj_weight, q, *config)
    torch.mm(x, q_a_proj_weight.T, out=latent)
    torch.mm(x, kv_a_proj_weight.T, out=kv)
    normalize(latent, q_a_layernorm_weight, normalized, rms_norm_eps)
    final_projection()
    return split_views(q, kv, batch, sequence)
