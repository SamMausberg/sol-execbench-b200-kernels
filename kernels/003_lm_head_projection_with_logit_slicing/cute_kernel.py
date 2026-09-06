# SPDX-License-Identifier: Apache-2.0
"""BF16 LM-head GEMM using Blackwell's persistent TMA/UMMA pipeline."""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import HardwareInfo

from cute_gemm import PersistentDenseGemmKernel

_compiled = {}


def _tensor(value, rows, columns):
    return from_dlpack(value.view(1, rows, columns).permute(1, 2, 0), assumed_align=16)


@torch.no_grad()
def configured_run(hidden_states, weight, output, bm=128, bn=256,
                   cluster_m=1, cluster_n=1, two_cta=False,
                   transpose=False, raster_m=True):
    if not all(x.is_contiguous() and x.data_ptr() % 16 == 0
               for x in (hidden_states, weight, output)):
        torch.mm(hidden_states.reshape(-1, hidden_states.shape[-1]), weight.t(),
                 out=output.view(-1, weight.shape[0]))
        return
    k = hidden_states.shape[-1]
    m, n = hidden_states.numel() // k, weight.shape[0]
    if transpose:
        a, b = _tensor(weight,n,k), _tensor(hidden_states,m,k)
        c = from_dlpack(output.view(1,m,n).permute(2,1,0),assumed_align=16)
        gemm_m, gemm_n, c_major = n, m, 'm'
    else:
        a, b, c = _tensor(hidden_states, m, k), _tensor(weight, n, k), _tensor(output, m, n)
        gemm_m, gemm_n, c_major = m, n, 'n'
    stream = cuda.CUstream(torch.cuda.current_stream(hidden_states.device).cuda_stream)
    key = (m, n, k, hidden_states.device.index, bm, bn, cluster_m, cluster_n,
           two_cta, transpose, raster_m)
    compiled = _compiled.get(key)
    if compiled is None:
        gemm = PersistentDenseGemmKernel(cutlass.Float32, two_cta, (bm, bn),
                                        (cluster_m, cluster_n), True, raster_m)
        if not gemm.can_implement((gemm_m, gemm_n, k, 1), cutlass.BFloat16, cutlass.BFloat16,
                                  cutlass.BFloat16, 'k', 'k', c_major):
            raise ValueError(f'Unsupported LM-head configuration: {key}')
        clusters = HardwareInfo().get_max_active_clusters(cluster_m * cluster_n)
        compiled = cute.compile(gemm, a, b, c, clusters, stream)
        _compiled[key] = compiled
    compiled(a, b, c, stream)


def run(hidden_states, weight):
    output = torch.empty((*hidden_states.shape[:-1], weight.shape[0]),
                         dtype=hidden_states.dtype, device=hidden_states.device)
    configured_run(hidden_states, weight, output)
    return output
