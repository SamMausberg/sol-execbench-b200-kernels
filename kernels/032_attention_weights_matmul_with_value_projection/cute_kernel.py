# SPDX-License-Identifier: Apache-2.0
"""Persistent attention/value GEMM with a nested batch layout for direct stores."""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import HardwareInfo
from cute_av import AttentionValueGemm
from kernel import run as triton_run

_compiled = {}


@cute.jit
def _launch(gemm: cutlass.Constexpr, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
            clusters: cutlass.Constexpr, stream: cuda.CUstream):
    # Each matrix has a (head, example) batch mode. Output head stride is128,
    # whereas example stride is S*5120; preserving this nested layout removes
    # the transpose/copy after the product without overlapping output elements.
    a = cute.group_modes(a, 2, 4)
    b = cute.group_modes(b, 2, 4)
    c = cute.group_modes(c, 2, 4)
    gemm(a, b, c, clusters, stream)


@cute.jit
def _launch_reverse(gemm: cutlass.Constexpr, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
                    clusters: cutlass.Constexpr, stream: cuda.CUstream):
    a = cute.group_modes(a, 2, 4)
    b = cute.group_modes(b, 2, 4)
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 0, 2, 3]))
    c = cute.group_modes(c, 2, 4)
    gemm(b, a, c, clusters, stream)


@torch.no_grad()
def launch(weights, values, output, bm=128, bn=128, cluster_m=1, two_cta=False, tma_store=True, reverse=False):
    batch, _, sequence, _ = weights.shape
    assert weights.is_contiguous() and values.is_contiguous() and output.is_contiguous()
    assert sequence % 8 == 0
    a = from_dlpack(weights.permute(2, 3, 1, 0), assumed_align=16)
    b = from_dlpack(values.permute(3, 2, 1, 0), assumed_align=16)
    c = from_dlpack(output.view(batch, sequence, 40, 128).permute(1, 3, 2, 0), assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream(weights.device).cuda_stream)
    key = (weights.device.index, batch, sequence, bm, bn, cluster_m, two_cta, tma_store, reverse)
    compiled = _compiled.get(key)
    if compiled is None:
        gemm = AttentionValueGemm(cutlass.Float32, two_cta, (bm, bn), (cluster_m, 1), tma_store)
        clusters = HardwareInfo().get_max_active_clusters(cluster_m)
        compiled = cute.compile(_launch_reverse if reverse else _launch, gemm, a, b, c, clusters, stream)
        _compiled[key] = compiled
    compiled(a, b, c, stream)


@torch.no_grad()
def run(weights, values):
    batch, _, sequence, _ = weights.shape
    if sequence % 128 or any(not tensor.is_contiguous() or tensor.data_ptr() % 16
                             for tensor in (weights, values)):
        return triton_run(weights, values)
    output = torch.empty((batch, sequence, 5120), dtype=weights.dtype, device=weights.device)
    launch(weights, values, output)
    return output
