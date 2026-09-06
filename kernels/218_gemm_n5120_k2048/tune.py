#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded small-M GEMM sweep using official correctness and shifted timing."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data.workload import ToleranceSpec

from kernel import _gemv_small, launch_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrices", nargs="+", type=int, default=[1, 6, 34, 172])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep", type=int, default=15)
    args = parser.parse_args()
    if any(m <= 0 for m in args.matrices):
        parser.error("matrix rows must be positive")
    directory = Path(__file__).parent
    tolerance = ToleranceSpec()
    results = {
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("kernel.py", "tune.py")
        },
        "clock_mode": "unlocked",
        "tolerance": tolerance.model_dump(),
        "records": [],
    }
    lock = Path(os.environ.get(
        "SOL_GPU_LOCK", "/workspace/sol-execbench-b200-kernels/.work/gpu.lock",
    ))
    lock.parent.mkdir(parents=True, exist_ok=True)
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        print("GPU lock acquired", flush=True)
        torch.manual_seed(218925)
        for m in args.matrices:
            a = torch.randn((m, 2048), device="cuda", dtype=torch.float16)
            b = torch.randn((5120, 2048), device="cuda", dtype=torch.float16)
            output = torch.empty((m, 5120), device="cuda", dtype=torch.float16)
            reference = a @ b.T
            configs = [("library", 0, 0, 0, 1, 0, 0)]
            if m <= 8:
                configs += [
                    ("gemv", 0, bn, 2048, 1, warps, 1)
                    for bn in (1, 2, 4, 8) for warps in (4, 8)
                ]
            configs += [
                ("split", bm, bn, bk, split, 4, 3)
                for bm, bn, bk in (
                    (16, 64, 128), (32, 64, 128),
                    (32, 128, 64), (64, 128, 64),
                )
                for split in (1, 2, 4, 8)
            ]
            for kind, bm, bn, bk, split, warps, stages in configs:
                def launch(x, y, z):
                    if kind == "library":
                        torch.mm(x, y.T, out=z)
                    elif kind == "gemv":
                        _gemv_small[(5120 // bn,)](x, y, z, m, bn, num_warps=warps)
                    else:
                        launch_split(x, y, z, bm, bn, bk, split, warps, stages)

                record = {
                    "m": m, "kind": kind, "bm": bm, "bn": bn, "bk": bk,
                    "split": split, "warps": warps, "stages": stages,
                }
                try:
                    output.fill_(float("nan"))
                    launch(a, b, output)
                    errors, exceeds = compute_error_stats(output, reference, tolerance)
                    record["errors"] = errors.model_dump()
                    if exceeds:
                        raise AssertionError("Official correctness tolerance failed")
                    record["latency_ms"] = time_runnable(
                        launch, [a, b], [output], "cuda", warmup=3, rep=args.rep,
                    )
                    record["passed"] = True
                except Exception as error:
                    record["passed"] = False
                    record["error"] = f"{type(error).__name__}: {error}"
                record["time"] = time.time()
                results["records"].append(record)
                print(json.dumps(record), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
