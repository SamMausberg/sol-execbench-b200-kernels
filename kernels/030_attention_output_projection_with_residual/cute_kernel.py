# SPDX-License-Identifier: Apache-2.0
"""Launch a Blackwell GEMM with BF16 rounding before the residual epilogue."""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import HardwareInfo

from cute_residual import SM100ProjectionResidual
from kernel import library_run


_compiled = {}


def _tensor(x, rows, cols):
    view = x.view(1, rows, cols).permute(1, 2, 0)
    return from_dlpack(view, assumed_align=16)


@torch.no_grad()
def configured_run(attn_output, residual, o_proj_weight, output,
                   bm=128, bn=128, cluster_m=1, cluster_n=1, two_cta=False):
    if not all(x.is_contiguous() for x in (attn_output, residual, o_proj_weight, output)):
        return library_run(attn_output, residual, o_proj_weight, output)
    n = attn_output.shape[-1]
    m = attn_output.numel() // n
    a = _tensor(attn_output, m, n)
    b = _tensor(o_proj_weight, n, n)
    c = _tensor(residual, m, n)
    d = _tensor(output, m, n)
    stream = cuda.CUstream(torch.cuda.current_stream(attn_output.device).cuda_stream)
    key = (m, n, attn_output.device.index, bm, bn, cluster_m, cluster_n, two_cta)
    compiled = _compiled.get(key)
    if compiled is None:
        gemm = SM100ProjectionResidual(cutlass.Float32, cutlass.Float32, two_cta,
                                      (bm, bn), (cluster_m, cluster_n))
        if not gemm.can_implement(a, b, c, d):
            raise ValueError(f"Unsupported projection configuration: {key}")
        clusters = HardwareInfo().get_max_active_clusters(cluster_m * cluster_n)
        compiled = cute.compile(gemm, a, b, c, d, 1.0, 1.0, clusters, stream)
        _compiled[key] = compiled
    compiled(a, b, c, d, 1.0, 1.0, stream)


def run(attn_output, residual, o_proj_weight, output):
    rows = attn_output.numel() // attn_output.shape[-1]
    if rows <= 512:
        config = (64, 64, 1, 1, False)
    elif rows <= 1024:
        config = (128, 128, 1, 1, False)
    else:
        config = (256, 256, 2, 1, True)
    configured_run(attn_output, residual, o_proj_weight, output, *config)
