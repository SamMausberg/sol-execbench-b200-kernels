#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded projection schedule comparison using official shifted-input timing."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data import Definition, Workload

from kernel import library_run
from variants import projection_residual, projection_residual_tiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="15,3,0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--ws", choices=("true", "false", "both"), default="both")
    parser.add_argument("--schedule", choices=("persistent", "tiled"), default="persistent")
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
        "schedule": args.schedule,
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in ("kernel.py", "variants.py", "tune.py")},
        "records": [],
    }
    tiles = [(64, 128, 64, 4, 3), (128, 128, 64, 4, 3),
             (128, 256, 64, 4, 3), (128, 256, 128, 8, 2),
             (128, 128, 128, 4, 3), (128, 128, 128, 4, 2)]
    configs = [{"kind": "library"}]
    for ws in ([True, False] if args.ws == "both" else [args.ws == "true"]):
        for bm, bn, bk, warps, stages in tiles:
            for programs in ((148, 296) if args.schedule == "persistent" else (0,)):
                configs.append(dict(kind="tma", bm=bm, bn=bn, bk=bk, warps=warps,
                                    stages=stages, programs=programs, ws=ws,
                                    group=8, subtile=bn == 256))
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        print("Waiting for shared GPU lock", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        for index in map(int, args.indices.split(",")):
            set_seed(30991 + index)
            workload = workloads[index]
            a, r, w = gen_inputs(definition, workload, "cuda")
            ref = namespace["run"](a, r, w)
            out = torch.empty_like(ref)
            n = a.shape[-1]
            m = a.numel() // n
            for config in configs:
                record = {"workload": index, "axes": workload.axes, **config}

                def launch(x, residual, weight, dest):
                    if config["kind"] == "library":
                        return library_run(x, residual, weight, dest)
                    bm, bn, bk = config["bm"], config["bn"], config["bk"]
                    programs = min(config["programs"], triton.cdiv(m, bm) * triton.cdiv(n, bn))
                    if args.schedule == "tiled":
                        programs = triton.cdiv(m, bm) * triton.cdiv(n, bn)
                    out_bn = bn // 2 if config["subtile"] else bn
                    ad = TensorDescriptor(x, [m, n], [n, 1], [bm, bk])
                    wd = TensorDescriptor(weight, [n, n], [n, 1], [bn, bk])
                    rd = TensorDescriptor(residual, [m, n], [n, 1], [bm, out_bn])
                    od = TensorDescriptor(dest, [m, n], [n, 1], [bm, out_bn])
                    fn = projection_residual if args.schedule == "persistent" else projection_residual_tiled
                    fn[(programs,)](
                        ad, wd, rd, od, m, n, n, bm, bn, bk, programs,
                        config["ws"], config["group"], config["subtile"],
                        num_warps=config["warps"], num_stages=config["stages"],
                    )

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
            del a, r, w, ref, out
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
