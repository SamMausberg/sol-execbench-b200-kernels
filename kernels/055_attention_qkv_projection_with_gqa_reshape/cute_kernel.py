# SPDX-License-Identifier: Apache-2.0
"""Grouped Blackwell projections with concrete transposed output views."""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import HardwareInfo

from cute_qkv import SM100GroupedQkv

_compiled = {}


def matrix(tensor):
    rows, cols = tensor.shape
    return from_dlpack(tensor.view(1, rows, cols).permute(1, 2, 0), assumed_align=16)


@torch.no_grad()
def launch(x, qw, kw, vw, q, k, v, bm=128, bn=256,
           cluster_m=1, cluster_n=1, two_cta=False):
    a, bq, bk, bv, dq, dk, dv = [matrix(t) for t in (x, qw, kw, vw, q, k, v)]
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    key = (x.shape[0], x.device.index, bm, bn, cluster_m, cluster_n, two_cta)
    compiled = _compiled.get(key)
    if compiled is None:
        gemm = SM100GroupedQkv(cutlass.Float32, cutlass.Float32, two_cta,
                              (bm, bn), (cluster_m, cluster_n))
        clusters = HardwareInfo().get_max_active_clusters(cluster_m * cluster_n)
        # The inherited C arguments provide layout metadata only. This variant
        # has no C/bias loads; every output element comes from its matrix product.
        compiled = cute.compile(gemm, a, bq, dq, dq, bk, dk, dk, bv, dv, dv,
                                1.0, 0.0, clusters, stream)
        _compiled[key] = compiled
    compiled(a, bq, dq, dq, bk, dk, dk, bv, dv, dv, 1.0, 0.0, stream)


@torch.no_grad()
def run(hidden_states, q_weight, k_weight, v_weight):
    batch, sequence, hidden = hidden_states.shape
    rows = batch * sequence
    x = hidden_states.reshape(rows, hidden)
    q = torch.empty((rows, 2048), dtype=x.dtype, device=x.device)
    k = torch.empty((rows, 512), dtype=x.dtype, device=x.device)
    v = torch.empty_like(k)
    if rows <= 256:
        config = (64, 64, 1, 1, False)
    elif rows <= 512:
        config = (128, 128, 1, 1, False)
    else:
        config = (256, 256, 2, 1, True)
    launch(x, q_weight, k_weight, v_weight, q, k, v, *config)
    return (q.view(batch, sequence, 16, 128).transpose(1, 2),
            k.view(batch, sequence, 4, 128).transpose(1, 2),
            v.view(batch, sequence, 4, 128).transpose(1, 2))
