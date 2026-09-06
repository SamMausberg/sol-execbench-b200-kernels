#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explore Q/K row tilings with official input shifting and CUPTI timing."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import triton
from sol_execbench.core.bench.timing import time_runnable

from kernel import _rmsnorm_dual, _rmsnorm_grid
from packed import _rmsnorm_packed
from persistent import _rmsnorm_persistent
from split import _rmsnorm_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[128, 1024, 8192])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep", type=int, default=15)
    parser.add_argument("--cache", choices=("default", "stream", "both"), default="both")
    parser.add_argument("--kinds", nargs="+", choices=("grid", "dual", "packed", "persistent"), default=["grid", "dual", "packed"])
    parser.add_argument("--focused", action="store_true", help="compare the best first-sweep tile with short-lane and persistent variants")
    args = parser.parse_args()
    if any(n <= 0 for n in args.tokens):
        parser.error("token counts must be positive")
    source = Path(__file__).with_name("kernel.py")
    results = {"source_sha256": {name: hashlib.sha256(source.with_name(name).read_bytes()).hexdigest() for name in ("kernel.py", "packed.py", "persistent.py", "split.py")}, "records": []}
    lock = Path(os.environ.get("SOL_GPU_LOCK", "/workspace/sol-execbench-b200-kernels/.work/gpu.lock"))
    lock.parent.mkdir(parents=True, exist_ok=True)
    configurations = [
        (kind, block_rows, warps, stream, programs, 1)
        for kind in args.kinds
        for block_rows, warps in ((4, 4), (8, 4), (16, 4), (32, 4), (32, 8), (64, 8))
        for stream in ((False, True) if args.cache == "both" else (args.cache == "stream",))
        for programs in ((256, 1024, 4096) if kind == "persistent" else (0,))
    ]
    if args.focused:
        configurations = [("grid", 8, 4, True, 0, 1)]
        configurations += [("split", rows, 4, stream, 0, parts) for rows in (8, 16, 32) for stream in (False, True) for parts in (2, 4)]
        configurations += [("persistent", rows, 4, True, programs, 1) for rows in (8, 16) for programs in (256, 1024, 4096)]
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        print("GPU lock acquired", flush=True)
        torch.manual_seed(28419)
        for tokens in args.tokens:
            q = torch.randn((tokens, 48, 128), device="cuda")
            k = torch.randn_like(q)
            wq = torch.randn((48, 128), device="cuda")
            wk = torch.randn_like(wq)
            oq, ok = torch.empty_like(q), torch.empty_like(k)
            eps = 1e-6
            rq = (q * torch.rsqrt((q * q).mean(-1, keepdim=True) + eps)) * wq
            rk = (k * torch.rsqrt((k * k).mean(-1, keepdim=True) + eps)) * wk
            rows = tokens * 48
            for kind, block_rows, warps, stream, programs, parts in configurations:
                kernel = {"grid": _rmsnorm_grid, "dual": _rmsnorm_dual, "packed": _rmsnorm_packed, "persistent": _rmsnorm_persistent, "split": _rmsnorm_split}[kind]
                grid = (triton.cdiv(rows, block_rows),) if kind == "dual" else (triton.cdiv(rows, block_rows), 2)
                if programs:
                    grid = (min(programs, grid[0]), 2)

                def launch(a, b, wa, wb, e, oa, ob):
                    options = {"PARTS": parts} if kind == "split" else {}
                    kernel[grid](a, b, wa, wb, oa, ob, e, rows, block_rows, stream, num_warps=warps, enable_fp_fusion=False, **options)

                oq.fill_(float("nan"))
                ok.fill_(float("nan"))
                launch(q, k, wq, wk, eps, oq, ok)
                torch.testing.assert_close(oq, rq, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(ok, rk, atol=1e-5, rtol=1e-5)
                latency = time_runnable(launch, [q, k, wq, wk, eps], [oq, ok], "cuda", warmup=3, rep=args.rep)
                record = {"tokens": tokens, "kind": kind, "block_rows": block_rows, "num_warps": warps, "stream": stream, "programs": programs, "parts": parts, "latency_ms": latency, "time": time.time()}
                results["records"].append(record)
                print(json.dumps(record), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
            del q, k, wq, wk, oq, ok, rq, rk


if __name__ == "__main__":
    main()
