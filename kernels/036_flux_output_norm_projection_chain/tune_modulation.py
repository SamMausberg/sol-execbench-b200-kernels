#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare bounded small-batch modulation schedules with official input shifting."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
from sol_execbench.core.bench.correctness import compute_error_stats,set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data import Definition,Workload
from modulation import _mod_gemv


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--indices',default='3,5,1');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    directory=Path(__file__).resolve().parent;root=directory.parents[1]
    definition=Definition(**json.loads((root/'.work/problems/36/definition.json').read_text()));ws=[Workload(**json.loads(s)) for s in (root/'.work/problems/36/workload.jsonl').read_text().splitlines()]
    report={'clock_mode':'unlocked','source_sha256':{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ['modulation.py','tune_modulation.py']},'records':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with (root/'.work/gpu.lock').open('a') as lock:
        print('Waiting for GPU lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        for index in map(int,args.indices.split(',')):
            set_seed(366901+index);w=ws[index];ins=gen_inputs(definition,w,'cuda');_,t,weight,bias,*_=ins;batch,c=t.shape
            ref=F.linear(t*torch.sigmoid(t),weight,bias);out=torch.empty_like(ref)
            configs=[(1,1,4096,4),(1,4,4096,4),(2,2,4096,4),(4,1,4096,4),(4,2,1024,4),(8,2,1024,8)]
            for bm,bn,chunk,warps in configs:
                def run(a,b,d,z):_mod_gemv[(triton.cdiv(2*c,bn),triton.cdiv(batch,bm))](a,b,d,z,batch,c,bm,bn,chunk,num_warps=warps,enable_fp_fusion=False)
                r={'workload':index,'axes':w.axes,'config':[bm,bn,chunk,warps]}
                try:
                    run(t,weight,bias,out);e,bad=compute_error_stats(out,ref,w.tolerance);r['passed']=not bad;r['errors']=e.model_dump()
                    if not bad:r['latency_ms']=time_runnable(run,[t,weight,bias],[out],'cuda',warmup=3,rep=10)
                except Exception as e:r['passed']=False;r['error']=f'{type(e).__name__}: {e}'
                report['records'].append(r);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(r),flush=True)
            del ins,t,weight,bias,ref,out
            torch.cuda.empty_cache()

if __name__=='__main__':main()
