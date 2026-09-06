#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare a few explicit Blackwell GEMM schedules with the official timer."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import torch
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data import Definition, Workload

from cute_kernel import configured_run
from kernel import library_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="15,3,0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    problem = root / ".work/problems/30"
    definition = Definition(**json.loads((problem / "definition.json").read_text()))
    workloads = [Workload(**json.loads(row)) for row in
                 (problem / "workload.jsonl").read_text().splitlines()]
    namespace = {}
    exec(definition.reference, namespace)
    report = {
        "clock_mode": "unlocked",
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in ("kernel.py", "cute_kernel.py", "cute_residual.py", "tune_cute.py")},
        "records": [],
    }
    configs = [None, (128, 128, 1, 1, False), (128, 256, 1, 1, False),
               (256, 128, 2, 1, True), (256, 256, 2, 1, True)]
    if args.extended:
        configs = [None, (64, 64, 1, 1, False), (64, 128, 1, 1, False),
                   (128, 64, 1, 1, False), (128, 128, 1, 2, False),
                   (128, 128, 2, 1, False), (256, 256, 4, 1, True),
                   (256, 256, 2, 2, True), (256, 128, 2, 2, True),
                   (128, 256, 2, 1, True)]
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        print("Waiting for shared GPU lock", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        for index in map(int, args.indices.split(",")):
            set_seed(30799 + index)
            workload = workloads[index]
            inputs = gen_inputs(definition, workload, "cuda")
            ref = namespace["run"](*inputs)
            out = torch.empty_like(ref)
            for config in configs:
                record = {"workload": index, "axes": workload.axes, "config": config}

                def launch(a, residual, weight, dest):
                    if config is None:
                        library_run(a, residual, weight, dest)
                    else:
                        configured_run(a, residual, weight, dest, *config)

                try:
                    out.fill_(float("nan"))
                    launch(*inputs, out)
                    error, exceeds = compute_error_stats(out, ref, workload.tolerance)
                    record["correctness"] = error.model_dump()
                    record["passed"] = not exceeds
                    if not exceeds:
                        record["latency_ms"] = time_runnable(
                            launch, inputs, [out], "cuda", warmup=3, rep=args.rep,
                        )
                except Exception as error:
                    record["passed"] = False
                    record["error"] = f"{type(error).__name__}: {error}"
                report["records"].append(record)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(record), flush=True)
            del inputs, ref, out
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
