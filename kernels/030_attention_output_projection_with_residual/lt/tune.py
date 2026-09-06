#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare C++ GEMM/residual dispatch across cuBLASLt heuristics."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data import Definition, Workload

from loader import load_bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="15,3,0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep", type=int, default=15)
    parser.add_argument("--algorithms", type=int, default=8)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    sys.path.insert(0, str(directory.parent))
    from kernel import library_run
    problem = root / ".work/problems/30"
    definition = Definition(**json.loads((problem / "definition.json").read_text()))
    workloads = [Workload(**json.loads(row)) for row in
                 (problem / "workload.jsonl").read_text().splitlines()]
    namespace = {}
    exec(definition.reference, namespace)
    report = {"clock_mode": "unlocked", "records": [], "source_sha256": {
        n: hashlib.sha256((directory / n).read_bytes()).hexdigest()
        for n in ("binding.cpp", "cublaslt_bridge.h", "loader.py", "tune.py")}}
    bridge = load_bridge()
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        print("Waiting for shared GPU lock", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        for index in map(int, args.indices.split(",")):
            set_seed(30917 + index)
            workload = workloads[index]
            a, r, w = gen_inputs(definition, workload, "cuda")
            ref = namespace["run"](a, r, w)
            out = torch.empty_like(ref)
            n = a.shape[-1]
            temporary = torch.empty_like(out).view(-1, n)
            workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=a.device)
            choices = [None, *list(bridge.algorithms(a.view(-1, n), w, temporary, workspace.numel()))[:args.algorithms]]
            for choice in choices:
                record = {"workload": index, "axes": workload.axes, "algorithm": choice}

                def launch(x, residual, weight, dest):
                    if choice is None:
                        library_run(x, residual, weight, dest)
                    else:
                        bridge.project(x.view(-1, n), weight, residual.view(-1, n),
                                       dest.view(-1, n), temporary, workspace, choice["index"])

                try:
                    out.fill_(float("nan"))
                    launch(a, r, w, out)
                    error, exceeds = compute_error_stats(out, ref, workload.tolerance)
                    record["correctness"] = error.model_dump()
                    record["passed"] = not exceeds
                    if not exceeds:
                        record["latency_ms"] = time_runnable(
                            launch, [a, r, w], [out], "cuda", warmup=3, rep=args.rep,
                        )
                except Exception as error:
                    record["passed"] = False
                    record["error"] = f"{type(error).__name__}: {error}"
                report["records"].append(record)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(record), flush=True)
            del a, r, w, ref, out, temporary, workspace
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
