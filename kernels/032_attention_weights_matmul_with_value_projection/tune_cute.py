#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded CuTe batch-layout correctness and geometry comparison."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import torch
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data.workload import ToleranceSpec
from cute_kernel import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["2,256", "32,128", "1,2048"])
    parser.add_argument("--label", default="bounded")
    parser.add_argument("--rep", type=int, default=15)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--padded", action="store_true")
    args = parser.parse_args()
    if args.padded:
        from padded_kernel import launch as candidate_launch
    else:
        candidate_launch = launch
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    sys.path.insert(0, str(root / "tools"))
    from qualify_gpu import snapshot
    report = {"scope": "nested-batch CuTe layout check; shifted arguments; unlocked clocks",
              "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                for name in ("cute_av.py", "cute_kernel.py", "padded_kernel.py", "tune_cute.py")},
              "records": [], "process_samples": []}
    path = root / f".work/tuning/32-cute-{args.label}.json"
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    stop = threading.Event()
    family = {}
    def monitor():
        while not stop.is_set():
            sample = snapshot(os.getpid(), family)
            for context in sample["contexts"]:
                context["process_name"] = Path(context["process_name"]).name
            report["process_samples"].append(sample)
            stop.wait(1)
    tolerance = ToleranceSpec(max_atol=1e-5, max_rtol=0.05)
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            torch.manual_seed(323141)
            for shape in args.shapes:
                batch, sequence = map(int, shape.split(","))
                a = torch.randn((batch, 40, sequence, sequence), device="cuda", dtype=torch.bfloat16)
                v = torch.randn((batch, 40, sequence, 128), device="cuda", dtype=torch.bfloat16)
                output = torch.empty((batch, sequence, 5120), device="cuda", dtype=torch.bfloat16)
                expected = torch.matmul(a, v).transpose(1, 2).reshape(batch, sequence, 5120)
                configs = [(128, 128, 1, False, True)]
                if not args.smoke:
                    configs += [(64, 128, 1, False, True), (256, 128, 2, True, True),
                                (128, 128, 1, False, False)]
                if args.reverse:
                    configs = [(128, 128, 1, False, True, True), (128, 256, 1, False, True, True)]
                if args.padded:
                    configs = [(128, 128, False), (64, 128, False), (128, 128, True)]
                for config in configs:
                    record = {"shape": [batch, sequence], "config": config, "started_at": time.time()}
                    def invoke(x, value, out):
                        candidate_launch(x, value, out, *config)
                    try:
                        output.fill_(float("nan"))
                        invoke(a, v, output)
                        errors, failed = compute_error_stats(output, expected, tolerance)
                        record["errors"] = errors.model_dump()
                        record["bitwise_equal"] = torch.equal(output, expected)
                        record["passed"] = not failed
                        if not failed:
                            record["latency_ms"] = time_runnable(invoke, [a, v], [output], "cuda", warmup=3, rep=args.rep)
                    except Exception as exc:
                        record["passed"] = False
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    record["finished_at"] = time.time()
                    report["records"].append(record)
                    path.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(record), flush=True)
                    if not record["passed"]:
                        raise RuntimeError("CuTe smoke failed; stop before additional geometry")
        finally:
            stop.set()
            thread.join(timeout=10)
            path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
