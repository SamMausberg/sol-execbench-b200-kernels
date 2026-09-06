# SPDX-License-Identifier: Apache-2.0
"""Bounded numerical and CUPTI comparison for three-fragment, six-product BF16 GEMMs.

The outer flock process retains the GPU lock until the CUDA child exits.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    )
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-held", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workloads", default="0,1,3,15")
    parser.add_argument("--shapes", choices=["output", "none"], default="none")
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--rep", type=int, default=8)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--bias-split-k", type=int, choices=[0,2,4,8], default=0)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[3]
    if not args.lock_held:
        lock = os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock"))
        print(f"Waiting for GPU lock: {lock}", flush=True)
        os.execvp("flock", ["flock", "--exclusive", lock, sys.executable, str(Path(__file__).resolve()),
                            "--lock-held", *sys.argv[1:]])

    import torch
    import torch.nn.functional as F
    from loader import load_bridge
    from sol_execbench.core.bench.correctness import compute_error_stats
    from sol_execbench.core.bench.io import gen_inputs
    from sol_execbench.core.bench.timing import time_runnable
    from sol_execbench.core.data.definition import Definition
    from sol_execbench.core.data.workload import Workload

    initial_gpu = gpu_snapshot()
    bridge = load_bridge(verbose=True)
    problem = root / ".work/problems/36"
    definition = Definition.model_validate_json((problem / "definition.json").read_text())
    workloads = [Workload.model_validate_json(line) for line in (problem / "workload.jsonl").read_text().splitlines()]
    reference = runpy.run_path(str(problem / "reference.py"))["run"]
    scratch = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    records = []
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = directory / "reports" / f"expansion6-{args.shapes}-{stamp}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_sha256": bridge.source_sha256, "file_sha256": bridge.file_sha256,
        "source_path": bridge.source_path, "cublaslt_version": bridge.cublaslt_version,
        "arguments": vars(args), "started_at": stamp, "initial_gpu": initial_gpu,
        "timing_caveat": "Unlocked clocks. Exploratory measurements; GPU snapshots cannot prove absence of transient contention.",
        "records": records,
    }

    def record(value):
        records.append(value)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(value), flush=True)

    def fragments(a, b):
        ap = torch.empty((3,*a.shape),dtype=torch.bfloat16,device=a.device)
        bp = torch.empty((3,*b.shape),dtype=torch.bfloat16,device=b.device)
        partials = torch.empty((5,a.shape[0],b.shape[0]),dtype=torch.float32,device=a.device)
        return ap,bp,partials

    def expanded(a, b, bias, strategy):
        out = torch.empty((a.shape[0], b.shape[0]), dtype=torch.float32, device=a.device)
        ap,bp,partials = fragments(a, b)
        bridge.expanded(a,b,bias,out,ap,bp,partials,scratch,strategy,0,-args.bias_split_k)
        return out

    with torch.no_grad():
        accuracy_passed = True
        if not args.skip_correctness:
            for strategy in (0,1):
                strategy_passed = True
                for index in map(int,args.workloads.split(",")):
                    for weight_scale in (1.0,32.0):
                        torch.manual_seed(36000+index)
                        inputs=gen_inputs(definition,workloads[index],"cuda")
                        hidden,temb,linear_weight,linear_bias,output_weight,output_bias,eps=inputs
                        linear_weight.mul_(weight_scale)
                        output_weight.mul_(weight_scale)
                        expected=reference(*inputs)
                        norm=(hidden-hidden.mean(-1,keepdim=True))/torch.sqrt(hidden.var(-1,keepdim=True,unbiased=False)+eps)
                        activated=temb*torch.sigmoid(temb)
                        if args.bias_split_k:
                            for m,n in ((temb.shape[0],6144),(hidden.shape[0]*hidden.shape[1],64)):
                                record({"kind":"explicit_split_algorithm","shape_mnk":[m,n,3072],
                                        "algorithm":bridge.forced_algorithm(m,n,3072,True,0,args.bias_split_k)})
                        for approximate_modulation,approximate_output in ((True,False),(False,True),(True,True)):
                            modulation=expanded(activated,linear_weight,linear_bias,strategy) if approximate_modulation else F.linear(activated,linear_weight,linear_bias)
                            shift,scale=modulation.chunk(2,-1)
                            adapted=norm*(1.0+scale[:,None,:])+shift[:,None,:]
                            result=expanded(adapted.flatten(0,1),output_weight,output_bias,strategy).view_as(expected) if approximate_output else F.linear(adapted,output_weight,output_bias)
                            errors,failed=compute_error_stats(result,expected,workloads[index].tolerance)
                            tolerance=workloads[index].tolerance
                            matched=((result-expected).abs()<=tolerance.max_atol+tolerance.max_rtol*expected.abs()).float().mean().item()
                            record({"kind":"end_to_end_accuracy","workload":index,"axes":workloads[index].axes,
                                    "weight_scale":weight_scale,"strategy":strategy,"expanded_modulation":approximate_modulation,
                                    "expanded_output":approximate_output,"passed":not failed,"matched_ratio":matched,
                                    "tolerance":tolerance.model_dump(),"errors":errors.model_dump()})
                            strategy_passed = strategy_passed and not failed
                accuracy_passed = accuracy_passed and strategy_passed
                if not strategy_passed:
                    record({"kind":"stopped","reason":"Numerical failure; later strategies and timing sweep skipped.","strategy":strategy})
                    break
        shapes = []
        if args.shapes == "output" and accuracy_passed:
            shapes += [(m,64,3072) for m in (128,1024,8192)]
        for m,n,k in shapes:
            torch.manual_seed(36+m+n)
            a=torch.randn((m,k),device="cuda",dtype=torch.float32)
            b=torch.randn((n,k),device="cuda",dtype=torch.float32)
            bias=torch.randn(n,device="cuda",dtype=torch.float32)
            out=torch.empty((m,n),device="cuda",dtype=torch.float32)
            ap,bp,partials=fragments(a,b)
            bridge.split(a,b,ap,bp)
            expected=F.linear(a,b,bias)
            baseline_candidates=bridge.algorithms(m,n,k,True,True,0,1)
            plain_candidates=bridge.algorithms(m,n,k,False,False,0,1)
            biased_candidates=bridge.algorithms(m,n,k,False,True,0,1)
            batch2_candidates=bridge.algorithms(m,n,k,False,False,0,2)
            batch3_candidates=bridge.algorithms(m,n,k,False,False,0,3)
            record({"kind":"algorithms","shape_mnk":[m,n,k],"baseline":baseline_candidates,
                    "plain":plain_candidates,"biased":biased_candidates,
                    "batch2":batch2_candidates,"batch3":batch3_candidates})
            forced_bias=bridge.forced_algorithm(m,n,k,True,0,args.bias_split_k) if args.bias_split_k else None

            def timed(fn,inputs,outputs):
                return time_runnable(fn,inputs,outputs,"cuda:0",warmup=2,rep=args.rep,seed=3601)

            split_ms=timed(lambda x,y,p,q:bridge.split(x,y,p,q),[a,b],[ap,bp])
            record({"kind":"timing","shape_mnk":[m,n,k],"path":"conversion","latency_ms":split_ms})
            for candidate in baseline_candidates[:args.max_candidates]:
                index=candidate["index"]
                bridge.baseline(a,b,bias,out,scratch,index)
                error=(out-expected).abs().max().item()
                elapsed=timed(lambda x,y,z,o:bridge.baseline(x,y,z,o,scratch,index),[a,b,bias],[out])
                record({"kind":"timing","shape_mnk":[m,n,k],"path":"emulated_bf16x9_plus_bias",
                        "algorithm":candidate,"max_absolute_error":error,"latency_ms":elapsed})
            for strategy in (0,1):
                best=None
                if strategy==0:
                    count=min(args.max_candidates,len(plain_candidates),len(biased_candidates))
                else:
                    count=min(args.max_candidates,len(batch2_candidates),len(batch3_candidates),len(biased_candidates))
                for index in range(count):
                    final_index=-args.bias_split_k if args.bias_split_k else index
                    bridge.products(ap,bp,bias,out,partials,scratch,strategy,index,final_index)
                    error=(out-expected).abs().max().item()
                    elapsed=timed(lambda p,q,z,o:bridge.products(p,q,z,o,partials,scratch,strategy,index,final_index),
                                  [ap,bp,bias],[out])
                    record({"kind":"timing","shape_mnk":[m,n,k],"path":"products_plus_bias","strategy":strategy,
                            "plain_algorithm":plain_candidates[index] if strategy==0 else None,
                            "batch2_algorithm":batch2_candidates[index] if strategy==1 else None,
                            "batch3_algorithm":batch3_candidates[index] if strategy==1 else None,
                            "bias_algorithm":forced_bias or biased_candidates[index],"max_absolute_error":error,"latency_ms":elapsed})
                    if best is None or elapsed<best[0]:
                        best=(elapsed,index)
                index=best[1]
                final_index=-args.bias_split_k if args.bias_split_k else index
                elapsed=timed(lambda x,y,z,o:bridge.expanded(x,y,z,o,ap,bp,partials,scratch,strategy,index,final_index),
                              [a,b,bias],[out])
                record({"kind":"timing","shape_mnk":[m,n,k],"path":"conversion_products_bias","strategy":strategy,
                        "selected_heuristic_index":index,"bias_algorithm":forced_bias or biased_candidates[index],"latency_ms":elapsed})
    report["final_gpu"]=gpu_snapshot()
    report["completed_at"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    target.write_text(json.dumps(report,indent=2)+"\n")
    print(f"Report: {target}",flush=True)


if __name__=="__main__":
    main()
