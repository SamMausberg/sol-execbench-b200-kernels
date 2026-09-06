#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded direct-layout AV tuning with official shifted-argument CUPTI timing."""
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
from kernel import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["32,128", "2,541", "1,131", "1,2048"])
    parser.add_argument("--mode", choices=("triton", "lt", "both", "direct"), default="both")
    parser.add_argument("--label", default="bounded")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--rep", type=int, default=15)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    sys.path.insert(0, str(root / "tools"))
    from qualify_gpu import snapshot
    bridge = None
    if args.mode in ("lt", "both"):
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
        from native_lt.loader import load_bridge
        bridge = load_bridge()
    report = {
        "scope": "bounded algorithm/geometry comparison; shifted arguments; cold L2; unlocked clocks",
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in ("kernel.py", "tune.py")},
        "lt_source_sha256": bridge.source_sha256 if bridge else None,
        "records": [], "process_samples": [],
    }
    path = root / f".work/tuning/32-{args.label}.json"
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
            torch.manual_seed(320031)
            for shape in args.shapes:
                batch, sequence = map(int, shape.split(","))
                a = torch.randn((batch, 40, sequence, sequence), device="cuda", dtype=torch.bfloat16)
                v = torch.randn((batch, 40, sequence, 128), device="cuda", dtype=torch.bfloat16)
                output = torch.empty((batch, sequence, 5120), device="cuda", dtype=torch.bfloat16)
                expected = torch.matmul(a, v).transpose(1, 2).reshape(batch, sequence, 5120)
                configs = [("library", {})]
                if batch == 1:
                    configs.append(("library_direct", {}))
                if bridge:
                    try:
                        configs += [("lt", config) for config in bridge.algorithms(a, v)[:args.limit]]
                    except Exception as exc:
                        report["records"].append({"shape": [batch, sequence], "mode": "lt_heuristics",
                                                  "passed": False, "error": str(exc)})
                if args.mode in ("triton", "both"):
                    configs += [("triton", dict(bm=bm, bn=bn, bk=bk, stages=stages,
                                                mma_v2=mma, warps=4, tma=False))
                                for bm, bn, bk, stages, mma in (
                                    (32, 64, 64, 3, True), (32, 128, 64, 3, True),
                                    (64, 128, 64, 3, True), (64, 128, 128, 3, True),
                                    (32, 128, 128, 2, False), (64, 128, 128, 2, False),
                                )]
                    if sequence % 128 == 0:
                        configs += [("triton", dict(bm=bm, bn=128, bk=bk, stages=stages,
                                                    warps=4, tma=True, tma_store=store))
                                    for bm, bk, stages, store in (
                                        (64, 128, 2, False), (128, 128, 2, False),
                                        (64, 128, 1, False), (128, 128, 1, False),
                                        (128, 128, 2, True),
                                    )]
                        if sequence % 256 == 0:
                            configs += [("triton", dict(bm=128, bn=128, bk=256, stages=2,
                                                        warps=4, tma=True))]
                if args.tiny:
                    configs = [("triton", dict(bm=bm, bn=bn, bk=bk, stages=stages,
                                                mma_v2=True, warps=warps, tma=False))
                               for bm, bn, bk, stages, warps in (
                                   (16, 64, 64, 2, 2), (16, 128, 64, 2, 4),
                                   (16, 128, 32, 3, 4), (32, 64, 64, 2, 2),
                                   (32, 64, 128, 2, 2), (32, 64, 64, 1, 4),
                                   (32, 128, 64, 2, 4), (64, 64, 64, 2, 4),
                               )]
                for mode, config in configs:
                    record = {"shape": [batch, sequence], "mode": mode, "config": config,
                              "started_at": time.time()}
                    def invoke(x, value, out):
                        if mode == "lt":
                            bridge.execute(x, value, out, config["index"])
                        elif mode == "library_direct":
                            interleaved = out.as_strided((40, sequence, 128), (128, 5120, 1))
                            torch.bmm(x.view(40, sequence, sequence),
                                      value.view(40, sequence, 128), out=interleaved)
                        else:
                            launch(x, value, out, **config)
                    def reference(x, value):
                        return torch.matmul(x, value).transpose(1, 2).reshape(batch, sequence, 5120)
                    try:
                        output.fill_(float("nan"))
                        actual = reference(a, v) if mode == "library" else output
                        if mode != "library":
                            invoke(a, v, output)
                        error, failed = compute_error_stats(actual, expected, tolerance)
                        record["errors"] = error.model_dump()
                        record["bitwise_equal"] = torch.equal(actual, expected)
                        record["passed"] = not failed
                        if not failed:
                            record["latency_ms"] = time_runnable(
                                reference if mode == "library" else invoke, [a, v],
                                [] if mode == "library" else [output], "cuda", warmup=3, rep=args.rep)
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
