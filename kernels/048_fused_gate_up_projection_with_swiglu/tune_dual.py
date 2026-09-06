#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded paired-CuTe geometry comparison under the shared GPU lock."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if os.environ.get("SOL_48_TUNER_LOCKED") != "1":
    lock = os.environ.get("SOL_GPU_LOCK", str(ROOT / ".work/gpu.lock"))
    print(f"Waiting for GPU lock before CUDA imports: {lock}", flush=True)
    os.execvpe("flock", ["flock", "--exclusive", lock, sys.executable, __file__, *sys.argv[1:]],
               {**os.environ, "SOL_48_TUNER_LOCKED": "1"})

import torch
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data.workload import ToleranceSpec

from dual_kernel import configured_run
from kernel import library_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[128, 1024, 8192])
    parser.add_argument("--label", default="initial")
    parser.add_argument("--rep", type=int, default=12)
    parser.add_argument("--wide-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "tools"))
    from qualify_gpu import snapshot

    spec = importlib.util.spec_from_file_location("reference48", ROOT / ".work/problems/48/reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    output = ROOT / f".work/tuning/48-dual-{args.label}.json"
    report = {
        "scope": "exploratory paired GEMM geometry; official shifted CUPTI timer, cold L2, unlocked clocks",
        "source_sha256": {name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
                          for name in ("cute_dual.py", "dual_kernel.py", "tune_dual.py", "kernel.py")},
        "records": [], "process_samples": [],
    }
    stop = threading.Event()
    family = {}

    def monitor():
        while not stop.is_set():
            report["process_samples"].append(snapshot(os.getpid(), family))
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    tolerance = ToleranceSpec(max_atol=3.2e-5, max_rtol=0.05)
    try:
        torch.manual_seed(480031)
        gate = torch.randn((24576, 3072), dtype=torch.bfloat16, device="cuda")
        up = torch.randn_like(gate)
        for rows in args.rows:
            x = torch.randn((1, rows, 3072), dtype=torch.bfloat16, device="cuda")
            out = torch.empty((1, rows, 24576), dtype=torch.bfloat16, device="cuda")
            expected = reference.run(x, gate, up)
            configs = [None] + [dict(bm=bm, bn=bn, cluster_m=cm, cluster_n=1, two_cta=two)
                for bm, bn, cm, two in (
                    (128, 64, 1, False), (128, 128, 1, False), (128, 256, 1, False),
                    (256, 128, 2, True), (256, 256, 2, True), (128, 128, 2, False),
                )]
            if args.wide_only:
                configs = [c for c in configs if c is not None and c["bn"] == 256]
            for config in configs:
                record = {"rows": rows, "config": config or "library", "started_at": time.time()}
                def invoke(a, b, c, d):
                    if config is None:
                        library_run(a, b, c, d)
                    else:
                        configured_run(a, b, c, d, **config)
                try:
                    out.fill_(float("nan"))
                    invoke(x, gate, up, out)
                    stats, failed = compute_error_stats(out, expected, tolerance)
                    record.update(passed=not failed, correctness=stats.model_dump())
                    if not failed:
                        record["latency_ms"] = time_runnable(
                            invoke, [x, gate, up], [out], "cuda", warmup=3, rep=args.rep,
                        )
                except Exception as exc:
                    record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                    if "illegal" in str(exc).lower() or "launch failure" in str(exc).lower():
                        raise
                finally:
                    record["finished_at"] = time.time()
                    report["records"].append(record)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(record), flush=True)
            del x, out, expected
    finally:
        stop.set()
        thread.join(timeout=5)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
