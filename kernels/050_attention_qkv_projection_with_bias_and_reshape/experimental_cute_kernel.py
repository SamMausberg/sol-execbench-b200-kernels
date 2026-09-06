# SPDX-License-Identifier: Apache-2.0
"""Fixed Q/K/V Blackwell GEMM with BF16 rounding before bias addition."""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import HardwareInfo

from cute_qkv_direct import SM100QkvBiasDirect as SM100QkvBias
import kernel

_compiled = {}


def _matrix(x, rows, cols):
    return from_dlpack(x.view(1, rows, cols).permute(1, 2, 0), assumed_align=16)


def _bias(x, rows):
    return from_dlpack(x.view(1, -1, 1).expand(rows, -1, -1), assumed_align=16)


@torch.no_grad()
def configured_run(hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
                   v_proj_weight, v_proj_bias, query_states, key_states, value_states,
                   bm=128, bn=128, cluster_m=1, cluster_n=1, two_cta=False, direct=False, occupancy=1):
    tensors = (hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
               v_proj_weight, v_proj_bias, query_states, key_states, value_states)
    if not all(x.is_contiguous() for x in tensors):
        return kernel.launch(*tensors)
    rows = hidden_states.numel() // 640
    a = _matrix(hidden_states, rows, 640)
    bq, bk, bv = (_matrix(tensors[i], n, 640) for i, n in ((1, 1024), (3, 256), (5, 256)))
    cq, ck, cv = (_bias(tensors[i], rows) for i in (2, 4, 6))
    dq, dk, dv = (_matrix(tensors[i], rows, n) for i, n in ((7, 1024), (8, 256), (9, 256)))
    stream = cuda.CUstream(torch.cuda.current_stream(hidden_states.device).cuda_stream)
    if not direct or rows % bm != 0 or occupancy != 1:
        raise ValueError("Direct-store experiment requires full M tiles and occupancy1")
    key = (rows, hidden_states.device.index, bm, bn, cluster_m, cluster_n, two_cta, direct, occupancy)
    compiled = _compiled.get(key)
    if compiled is None:
        kernel_class = SM100QkvBias
        if direct:
            from cute_qkv_direct import SM100QkvBiasDirect
            kernel_class = SM100QkvBiasDirect
        gemm = kernel_class(cutlass.Float32, cutlass.Float32, two_cta,
                           (bm, bn), (cluster_m, cluster_n))
        clusters = HardwareInfo().get_max_active_clusters(cluster_m * cluster_n) * occupancy
        compiled = cute.compile(gemm, a, bq, cq, dq, bk, ck, dk, bv, cv, dv,
                                1.0, 1.0, clusters, stream)
        _compiled[key] = compiled
    compiled(a, bq, cq, dq, bk, ck, dk, bv, cv, dv, 1.0, 1.0, stream)


def run(*tensors):
    rows = tensors[0].numel() // 640
    config = (64, 64, 1, 1, False) if rows <= 512 else (128, 128, 1, 1, False)
    configured_run(*tensors, *config)
