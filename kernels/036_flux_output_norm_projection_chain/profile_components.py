#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record full-chain and individual kernel timings using the official timer."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.correctness import set_seed, compute_error_stats
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data import Definition, Workload
import kernel
import kernel_sm80
import split_kernel


def main():
    p = argparse.ArgumentParser();p.add_argument('--indices', default='3,1');p.add_argument('--output', type=Path, required=True)
    args=p.parse_args(); directory=Path(__file__).resolve().parent;root=directory.parents[1]
    definition=Definition(**json.loads((root/'.work/problems/36/definition.json').read_text()))
    workloads=[Workload(**json.loads(x)) for x in (root/'.work/problems/36/workload.jsonl').read_text().splitlines()]
    scope={};exec(definition.reference, scope)
    report={'clock_mode':'unlocked','torch_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'sources':{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ['kernel.py','kernel_sm80.py','split_kernel.py','profile_components.py']},'records':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with (root/'.work/gpu.lock').open('a') as lock:
        print('Waiting for GPU lock',flush=True);fcntl.flock(lock, fcntl.LOCK_EX)
        for index in map(int,args.indices.split(',')):
            set_seed(361907+index);w=workloads[index];inputs=gen_inputs(definition,w,'cuda');ref=scope['run'](*inputs);out=torch.empty_like(ref)
            x,t,weight,bias,pw,pbias,eps=inputs;batch,seq,c=x.shape;n=pw.shape[0]
            for name,fn in [('fused',kernel.run),('register_mma',kernel_sm80.run),('split_library',split_kernel.run)]:
                fn(*inputs,out);stats,bad=compute_error_stats(out,ref,w.tolerance)
                r={'workload':index,'axes':w.axes,'kind':name,'passed':not bad,'errors':stats.model_dump()}
                if not bad:r['latency_ms']=time_runnable(fn,inputs,[out],'cuda',warmup=3,rep=10)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    fn(*inputs,out);torch.cuda.synchronize()
                r['cuda_events']=[{'name':e.name,'start_us':e.time_range.start,'end_us':e.time_range.end,'duration_us':e.time_range.elapsed_us()} for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                report['records'].append(r);print(json.dumps(r),flush=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
            silu=t*torch.sigmoid(t);mod=F.linear(silu,weight,bias);adapted=torch.empty_like(x)
            split_kernel._norm_mod[(batch*seq,)](x,mod,adapted,eps,seq,c,triton.next_power_of_2(c),num_warps=4,enable_fp_fusion=False)
            projection=torch.empty((batch,2*c),device=x.device,dtype=x.dtype)
            def mod_fused(a,b,d,z):
                kernel._modulation[(triton.cdiv(batch,16),triton.cdiv(2*c,32))](a,b,d,z,batch,c,16,32,64,num_warps=4,num_stages=3,enable_fp_fusion=False)
            def mod_mma(a,b,d,z):
                kernel_sm80._modulation[(triton.cdiv(batch,16),triton.cdiv(2*c,32))](a,b,d,z,batch,c,16,32,64,num_warps=4,num_stages=3,enable_fp_fusion=False,arch='sm80')
            def mod_lib(a,b,d,z):torch.addmm(d,a,b.t(),out=z)
            def out_lib(a,b,d,z):torch.addmm(d,a.view(-1,c),b.t(),out=z.view(-1,n))
            def norm(a,b,z):split_kernel._norm_mod[(batch*seq,)](a,b,z,eps,seq,c,triton.next_power_of_2(c),num_warps=4,enable_fp_fusion=False)
            for name,fn,ins,outs in [('mod_fused',mod_fused,[t,weight,bias],[projection]),('mod_mma',mod_mma,[t,weight,bias],[projection]),('mod_library',mod_lib,[silu,weight,bias],[projection]),('norm',norm,[x,mod],[adapted]),('output_library',out_lib,[adapted,pw,pbias],[out])]:
                value=time_runnable(fn,ins,outs,'cuda',warmup=3,rep=10)
                r={'workload':index,'axes':w.axes,'kind':name,'latency_ms':value};report['records'].append(r);print(json.dumps(r),flush=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
            del inputs,ref,out,x,t,weight,bias,pw,pbias,silu,mod,adapted,projection
            torch.cuda.empty_cache()

if __name__=='__main__':main()
