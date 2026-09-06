#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded MLA cuBLASLt algorithm comparison using the shared generic bridge."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[131, 8192])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--label", default="initial")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    sys.path.insert(0, str(root / "kernels/010_attention_value_projection_with_transpose/variants"))
    sys.path.insert(0, str(root / "tools"))
    from cublaslt_loader import load_bridge
    from qualify_gpu import snapshot

    bridge = load_bridge()
    report = {
        "scope": "shape-only cuBLASLt algorithm tuning, shifted inputs/outputs, unlocked clocks",
        "bridge_source_sha256": bridge.source_sha256,
        "cublaslt_version": bridge.cublaslt_version,
        "tuner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": [], "process_samples": [],
    }
    output = root / f".work/tuning/43-lt-{args.label}.json"
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
            torch.manual_seed(431080)
            workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            for rows in args.rows:
                for projection, columns, inner in (("qa", 1536, 7168), ("qb", 24576, 1536), ("kv", 576, 7168)):
                    a = torch.randn((rows, inner), dtype=torch.bfloat16, device="cuda")
                    b = torch.randn((columns, inner), dtype=torch.bfloat16, device="cuda")
                    c = torch.empty((rows, columns), dtype=torch.bfloat16, device="cuda")
                    expected = torch.mm(a, b.T)
                    algorithms = bridge.algorithms(a, b, c, workspace.numel())
                    row = {"rows": rows, "projection": projection, "n": columns, "k": inner,
                           "workspace_limit": workspace.numel(), "candidates": []}
                    row["torch_latency_ms"] = time_runnable(
                        lambda aa, bb, cc: torch.mm(aa, bb.T, out=cc), [a, b], [c], "cuda", warmup=3, rep=15,
                    )
                    for algorithm in algorithms[:args.limit]:
                        record = {**algorithm, "started_at": time.time()}
                        try:
                            bridge.matmul(a, b, c, workspace, algorithm["index"])
                            stats, failed = compute_error_stats(c, expected, tolerance)
                            record.update(passed=not failed, errors=stats.model_dump(),
                                          bitwise_equal=torch.equal(c, expected))
                            if not failed:
                                record["latency_ms"] = time_runnable(
                                    lambda aa, bb, cc: bridge.matmul(aa, bb, cc, workspace, algorithm["index"]),
                                    [a, b], [c], "cuda", warmup=3, rep=15,
                                )
                        except Exception as exc:
                            record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                        record["finished_at"] = time.time()
                        row["candidates"].append(record)
                        print(json.dumps({"rows": rows, "projection": projection, **record}), flush=True)
                    report["records"].append(row)
                    output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            stop.set()
            thread.join(timeout=10)
            output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
