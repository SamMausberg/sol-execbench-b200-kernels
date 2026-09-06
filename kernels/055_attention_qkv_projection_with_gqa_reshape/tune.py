#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded grouped-QKV geometry search with the official shifted CUPTI timer."""

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
    parser.add_argument("--rows", nargs="+", type=int, default=[131, 422, 2048, 8192])
    parser.add_argument("--label", default="initial")
    parser.add_argument("--rep", type=int, default=15)
    parser.add_argument("--backend", choices=("triton", "cute"), default="triton")
    parser.add_argument("--medium-tiles", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    sys.path.insert(0, str(root / "tools"))
    from qualify_gpu import snapshot

    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    output = root / f".work/tuning/55-{args.label}.json"
    result = {
        "scope": "geometry tuning with shifted inputs/outputs, cold cache, unlocked clocks",
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in ("kernel.py", "tune.py", "cute_kernel.py", "cute_qkv.py")},
        "records": [], "process_samples": [],
    }
    stop = threading.Event()
    family = {}

    def monitor():
        while not stop.is_set():
            sample = snapshot(os.getpid(), family)
            for row in sample["contexts"]:
                row["process_name"] = Path(row["process_name"]).name
            result["process_samples"].append(sample)
            stop.wait(1.0)

    tolerance = ToleranceSpec(max_atol=1e-5, max_rtol=0.05)
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if args.backend == "cute":
            from cute_kernel import launch as launch_cute
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            torch.manual_seed(550031)
            qw = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
            kw = torch.randn((512, 2048), device="cuda", dtype=torch.bfloat16)
            vw = torch.randn_like(kw)
            for rows in args.rows:
                x = torch.randn((rows, 2048), device="cuda", dtype=torch.bfloat16)
                q = torch.empty((rows, 2048), device="cuda", dtype=torch.bfloat16)
                k = torch.empty((rows, 512), device="cuda", dtype=torch.bfloat16)
                v = torch.empty_like(k)
                expected = [torch.mm(x, weight.T) for weight in (qw, kw, vw)]
                configs = [("library", 0, 0, 0, 0, 0, 0, False)]
                if args.backend == "cute":
                    cute_tiles = (
                        (64, 64, 1, False), (128, 128, 1, False),
                        (128, 256, 1, False), (256, 128, 2, True),
                        (256, 256, 2, True),
                    )
                    if args.medium_tiles:
                        cute_tiles = ((128, 128, 1, False), (128, 256, 1, False),
                                      (256, 256, 2, True))
                    configs += [("cute", bm, bn, 0, 0, 0, cluster, two)
                                for bm, bn, cluster, two in cute_tiles]
                elif rows <= 512:
                    configs += [("plain", bm, bn, bk, 4, 3, 4, False) for bm, bn, bk in (
                        (16, 64, 128), (32, 64, 128), (32, 128, 64),
                        (32, 128, 128), (64, 64, 128), (64, 128, 64),
                    )]
                    configs += [("tma", bm, bn, bk, 4, stages, 4, False) for bm, bn, bk, stages in (
                        (32, 64, 64, 3), (64, 64, 128, 2),
                        (64, 128, 64, 3), (128, 128, 64, 3),
                    )]
                else:
                    configs += [("tma", bm, bn, bk, warps, stages, group, persistent)
                                for bm, bn, bk, warps, stages, group, persistent in (
                        (64, 128, 64, 4, 3, 4, False),
                        (128, 128, 64, 4, 3, 4, False),
                        (128, 128, 128, 4, 2, 4, False),
                        (128, 256, 64, 4, 3, 4, False),
                        (128, 256, 64, 8, 3, 4, False),
                        (128, 128, 128, 4, 1, 4, False),
                        (128, 128, 64, 4, 1, 4, False),
                        (128, 128, 64, 4, 3, 1, False),
                        (128, 128, 64, 4, 3, 8, False),
                        (128, 128, 64, 4, 3, 4, True),
                    )]
                for mode, bm, bn, bk, warps, stages, group, persistent in configs:
                    config = dict(mode=mode, bm=bm, bn=bn, bk=bk, warps=warps,
                                  stages=stages, group=group, persistent=persistent)
                    if mode == "cute":
                        config = dict(mode=mode, bm=bm, bn=bn, cluster_m=group,
                                      cluster_n=1, two_cta=persistent)
                    record = {"rows": rows, "config": config, "started_at": time.time()}

                    def invoke(a, b, c, d, oq, ok, ov):
                        if mode == "library":
                            for weight, out in ((b, oq), (c, ok), (d, ov)):
                                torch.mm(a, weight.T, out=out)
                        elif mode == "cute":
                            launch_cute(a, b, c, d, oq, ok, ov,
                                        **{key: value for key, value in config.items() if key != "mode"})
                        else:
                            launch(a, b, c, d, oq, ok, ov, **config)

                    try:
                        for out in (q, k, v):
                            out.fill_(float("nan"))
                        invoke(x, qw, kw, vw, q, k, v)
                        errors = [compute_error_stats(out, ref, tolerance)
                                  for out, ref in zip((q, k, v), expected)]
                        record["correctness"] = [error.model_dump() for error, _ in errors]
                        record["passed"] = not any(failed for _, failed in errors)
                        if record["passed"]:
                            record["latency_ms"] = time_runnable(
                                invoke, [x, qw, kw, vw], [q, k, v], "cuda",
                                warmup=3, rep=args.rep,
                            )
                    except Exception as exc:
                        record["passed"] = False
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    record["finished_at"] = time.time()
                    result["records"].append(record)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps(record), flush=True)
                del x, q, k, v, expected
        finally:
            stop.set()
            thread.join(timeout=10)
            output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
