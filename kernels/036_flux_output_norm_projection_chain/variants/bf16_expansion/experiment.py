# SPDX-License-Identifier: Apache-2.0
"""Bounded numerical and CUPTI comparison for two-fragment BF16 GEMMs.

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
    parser.add_argument("--shapes", choices=["output", "modulation", "all", "none"], default="output")
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--rep", type=int, default=8)
    parser.add_argument("--skip-correctness", action="store_true")
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
    target = directory / "reports" / f"expansion-{args.shapes}-{stamp}.json"
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
        return tuple(torch.empty_like(x, dtype=torch.bfloat16) for x in [a, a, b, b])

    def expanded(a, b, bias, terms):
        out = torch.empty((a.shape[0], b.shape[0]), dtype=torch.float32, device=a.device)
        ah, al, bh, bl = fragments(a, b)
        bridge.expanded(a, b, bias, out, ah, al, bh, bl, scratch, terms, 0, 0)
        return out

    with torch.no_grad():
        if not args.skip_correctness:
            for index in map(int, args.workloads.split(",")):
                for weight_scale in [1.0, 32.0]:
                    torch.manual_seed(36000 + index)
                    inputs = gen_inputs(definition, workloads[index], "cuda")
                    hidden, temb, linear_weight, linear_bias, output_weight, output_bias, eps = inputs
                    linear_weight.mul_(weight_scale)
                    output_weight.mul_(weight_scale)
                    expected = reference(*inputs)
                    norm = (hidden-hidden.mean(-1,keepdim=True)) / torch.sqrt(hidden.var(-1,keepdim=True,unbiased=False)+eps)
                    activated = temb * torch.sigmoid(temb)
                    for terms in [3, 4]:
                        for approximate_modulation, approximate_output in [(True,False),(False,True),(True,True)]:
                            modulation = expanded(activated,linear_weight,linear_bias,terms) if approximate_modulation else F.linear(activated,linear_weight,linear_bias)
                            shift, scale = modulation.chunk(2,-1)
                            adapted = norm * (1.0+scale[:,None,:]) + shift[:,None,:]
                            result = expanded(adapted.flatten(0,1),output_weight,output_bias,terms).view_as(expected) if approximate_output else F.linear(adapted,output_weight,output_bias)
                            errors, failed = compute_error_stats(result,expected,workloads[index].tolerance)
                            tolerance = workloads[index].tolerance
                            matched = ((result-expected).abs() <= tolerance.max_atol+tolerance.max_rtol*expected.abs()).float().mean().item()
                            record({"kind":"end_to_end_accuracy","workload":index,"axes":workloads[index].axes,
                                    "weight_scale":weight_scale,"terms":terms,"expanded_modulation":approximate_modulation,
                                    "expanded_output":approximate_output,"passed":not failed,"matched_ratio":matched,
                                    "tolerance":tolerance.model_dump(),"errors":errors.model_dump()})
        shapes = []
        if args.shapes in ("output","all"):
            shapes += [(m,64,3072) for m in (128,1024,8192)]
        if args.shapes in ("modulation","all"):
            shapes += [(m,6144,3072) for m in (1,8,32)]
        for m,n,k in shapes:
            torch.manual_seed(36+m+n)
            a=torch.randn((m,k),device="cuda",dtype=torch.float32)
            b=torch.randn((n,k),device="cuda",dtype=torch.float32)
            bias=torch.randn(n,device="cuda",dtype=torch.float32)
            out=torch.empty((m,n),device="cuda",dtype=torch.float32)
            ah,al,bh,bl=fragments(a,b)
            bridge.split(a,b,ah,al,bh,bl)
            expected=F.linear(a,b,bias)
            baseline_candidates=bridge.algorithms(m,n,k,True,True,0)
            plain_candidates=bridge.algorithms(m,n,k,False,False,0)
            biased_candidates=bridge.algorithms(m,n,k,False,True,0)
            record({"kind":"algorithms","shape_mnk":[m,n,k],"baseline":baseline_candidates,
                    "plain":plain_candidates,"biased":biased_candidates})

            def timed(fn,inputs,outputs):
                return time_runnable(fn,inputs,outputs,"cuda:0",warmup=2,rep=args.rep,seed=3601)

            split_ms=timed(lambda x,y,p,q,r,s:bridge.split(x,y,p,q,r,s),[a,b],[ah,al,bh,bl])
            record({"kind":"timing","shape_mnk":[m,n,k],"path":"conversion","latency_ms":split_ms})
            for candidate in baseline_candidates[:args.max_candidates]:
                index=candidate["index"]
                bridge.baseline(a,b,bias,out,scratch,index)
                error=(out-expected).abs().max().item()
                elapsed=timed(lambda x,y,z,o:bridge.baseline(x,y,z,o,scratch,index),[a,b,bias],[out])
                record({"kind":"timing","shape_mnk":[m,n,k],"path":"emulated_bf16x9_plus_bias",
                        "algorithm":candidate,"max_absolute_error":error,"latency_ms":elapsed})
            for terms in (3,4):
                best=None
                count=min(args.max_candidates,len(plain_candidates),len(biased_candidates))
                for index in range(count):
                    bridge.products(ah,al,bh,bl,bias,out,scratch,terms,index,index)
                    error=(out-expected).abs().max().item()
                    elapsed=timed(lambda p,q,r,s,z,o:bridge.products(p,q,r,s,z,o,scratch,terms,index,index),
                                  [ah,al,bh,bl,bias],[out])
                    record({"kind":"timing","shape_mnk":[m,n,k],"path":"products_plus_bias","terms":terms,
                            "plain_algorithm":plain_candidates[index],"bias_algorithm":biased_candidates[index],
                            "max_absolute_error":error,"latency_ms":elapsed})
                    if best is None or elapsed<best[0]:
                        best=(elapsed,index)
                index=best[1]
                elapsed=timed(lambda x,y,z,o:bridge.expanded(x,y,z,o,ah,al,bh,bl,scratch,terms,index,index),
                              [a,b,bias],[out])
                record({"kind":"timing","shape_mnk":[m,n,k],"path":"conversion_products_bias","terms":terms,
                        "plain_algorithm":plain_candidates[index],"bias_algorithm":biased_candidates[index],"latency_ms":elapsed})
    report["final_gpu"]=gpu_snapshot()
    report["completed_at"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    target.write_text(json.dumps(report,indent=2)+"\n")
    print(f"Report: {target}",flush=True)


if __name__=="__main__":
    main()
