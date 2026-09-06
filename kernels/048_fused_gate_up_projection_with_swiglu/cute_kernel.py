# SPDX-License-Identifier: Apache-2.0
"""Dense BF16 gate/up CTAs exchange activation tiles through cluster shared memory."""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import HardwareInfo

from cute_paired import SM100FusedGateUp

_compiled = {}


def matrix(value):
    rows,cols=value.shape
    return from_dlpack(value.view(1,rows,cols).permute(1,2,0),assumed_align=16)


@torch.no_grad()
def configured_run(x,gate_proj,up_proj,output,bm=128,bn=128,cluster_m=1,cluster_n=2):
    assert cluster_n % 2==0
    k=x.shape[-1]
    m=x.numel()//k
    n=gate_proj.shape[0]
    a,gate,up,out=[matrix(t) for t in (x.view(m,k),gate_proj,up_proj,output.view(m,n))]
    stream=cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    key=(m,n,k,x.device.index,bm,bn,cluster_m,cluster_n)
    compiled=_compiled.get(key)
    if compiled is None:
        gemm=SM100FusedGateUp(cutlass.Float32,cutlass.Float32,False,(bm,bn),(cluster_m,cluster_n))
        clusters=HardwareInfo().get_max_active_clusters(cluster_m*cluster_n)
        # The inherited third descriptor is a duplicate used only as metadata.
        # The interleaved scheduler has exactly two projection positions.
        compiled=cute.compile(gemm,a,gate,out,out,up,out,out,up,out,out,1.0,0.0,clusters,stream)
        _compiled[key]=compiled
    compiled(a,gate,out,out,up,out,out,up,out,out,1.0,0.0,stream)


def run(x,gate_proj,up_proj,output):
    configured_run(x,gate_proj,up_proj,output)
