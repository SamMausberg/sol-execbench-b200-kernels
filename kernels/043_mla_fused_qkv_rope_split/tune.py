#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded initial-projection tuning with shifted inputs and CUPTI timing."""

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

from cute_kernel import project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[131, 1492, 8192])
    parser.add_argument("--label", default="initial")
    parser.add_argument("--rep", type=int, default=15)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    sys.path.insert(0, str(root / "tools"))
    from qualify_gpu import snapshot

    report = {
        "scope": "initial projection geometry, cold cache, shifted arguments, unlocked clocks",
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in ("cute_kernel.py", "cute_mla.py", "tune.py")},
        "records": [], "process_samples": [],
    }
    path = root / f".work/tuning/43-project-{args.label}.json"
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

    tolerance = ToleranceSpec(max_atol=0.0044, max_rtol=0.05)
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            torch.manual_seed(430019)
            qw = torch.randn((1536, 7168), device="cuda", dtype=torch.bfloat16)
            kw = torch.randn((576, 7168), device="cuda", dtype=torch.bfloat16)
            for rows in args.rows:
                x = torch.randn((rows, 7168), device="cuda", dtype=torch.bfloat16)
                q = torch.empty((rows, 1536), device="cuda", dtype=torch.bfloat16)
                kv = torch.empty((rows, 576), device="cuda", dtype=torch.bfloat16)
                expected = (torch.mm(x, qw.T), torch.mm(x, kw.T))
                configs = [("library", 0, 0, 0, False)] + [
                    ("cute", bm, bn, cm, two) for bm, bn, cm, two in (
                        (64, 64, 1, False), (128, 128, 1, False),
                        (128, 256, 1, False), (256, 128, 2, True),
                        (256, 256, 2, True),
                    )
                ]
                for mode, bm, bn, cm, two in configs:
                    record = {"rows": rows, "mode": mode, "config": [bm, bn, cm, two],
                              "started_at": time.time()}

                    def invoke(a, b, c, oq, ok):
                        if mode == "library":
                            torch.mm(a, b.T, out=oq)
                            torch.mm(a, c.T, out=ok)
                        else:
                            project(a, b, c, oq, ok, bm, bn, cm, two)

                    try:
                        q.fill_(float("nan"))
                        kv.fill_(float("nan"))
                        invoke(x, qw, kw, q, kv)
                        errors = [compute_error_stats(out, ref, tolerance)
                                  for out, ref in zip((q, kv), expected)]
                        record["errors"] = [error.model_dump() for error, _ in errors]
                        record["bitwise_equal"] = [torch.equal(out, ref)
                                                  for out, ref in zip((q, kv), expected)]
                        record["passed"] = not any(failed for _, failed in errors)
                        if record["passed"]:
                            record["latency_ms"] = time_runnable(
                                invoke, [x, qw, kw], [q, kv], "cuda", warmup=3, rep=args.rep,
                            )
                    except Exception as exc:
                        record["passed"] = False
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    record["finished_at"] = time.time()
                    report["records"].append(record)
                    path.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(record), flush=True)
        finally:
            stop.set()
            thread.join(timeout=10)
            path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
