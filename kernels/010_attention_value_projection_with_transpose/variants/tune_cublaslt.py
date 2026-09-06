"""Bounded cuBLASLt heuristic search using official cold-L2 CUPTI timing."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti, time_runnable
from sol_execbench.core.data.workload import ToleranceSpec

from cublaslt_loader import load_bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, action="append")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--workspace-mib", type=int, default=32)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--fixed-addresses", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    bridge = load_bridge()
    dtype = getattr(torch, args.dtype)
    tolerance = ToleranceSpec(max_atol=args.atol, max_rtol=args.rtol, required_matched_ratio=0.99)
    records = []
    with Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock"))).open("a+") as lock:
        print("Waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.manual_seed(10821)
        workspace = torch.empty(args.workspace_mib * 1024 * 1024, dtype=torch.uint8, device="cuda")
        for m in args.m or (128, 1024, 2164, 8192):
            a = torch.randn((m, args.k), dtype=dtype, device="cuda")
            b = torch.randn((args.n, args.k), dtype=dtype, device="cuda")
            c = torch.empty((m, args.n), dtype=dtype, device="cuda")
            expected = F.linear(a, b)
            candidates = bridge.algorithms(a, b, c, workspace.numel())
            row = {"m": m, "n": args.n, "k": args.k, "dtype": args.dtype,
                   "workspace_limit": workspace.numel(), "heuristics": candidates,
                   "candidates": []}
            def measure(fn):
                if args.fixed_addresses:
                    return statistics.median(bench_gpu_time_with_cupti(
                        lambda: fn(a, b, c), warmup=3, rep=20, cold_l2_cache=True))
                return time_runnable(fn, [a, b], [c], device="cuda:0", warmup=3, rep=20,
                                     return_mode="median", methodology="cupti", seed=711)
            row["timing_inputs"] = "fixed addresses" if args.fixed_addresses else "official shifting allocator"
            row["torch_latency_ms"] = measure(lambda aa, bb, cc: torch.mm(aa, bb.t(), out=cc))
            print(json.dumps({"m": m, "torch_latency_ms": row["torch_latency_ms"],
                              "heuristic_candidates": len(candidates)}), flush=True)
            for algorithm in candidates[:args.limit]:
                record = dict(algorithm)
                try:
                    bridge.matmul(a, b, c, workspace, algorithm["index"])
                    torch.cuda.synchronize()
                    stats, exceeds = compute_error_stats(c, expected, tolerance)
                    record.update(passed=not exceeds, errors=stats.model_dump())
                    if not exceeds:
                        record["latency_ms"] = measure(
                            lambda aa, bb, cc: bridge.matmul(aa, bb, cc, workspace, algorithm["index"]))
                        record["speedup_vs_torch"] = row["torch_latency_ms"] / record["latency_ms"]
                except Exception as exc:
                    record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                row["candidates"].append(record)
                print(json.dumps({"m": m, **record}), flush=True)
            records.append(row)
    target = root / ".work/tuning/cublaslt"
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"{stamp}-{args.dtype}-n{args.n}-k{args.k}.json"
    path.write_text(json.dumps({"settings": vars(args),
                               "bridge_source_sha256": getattr(bridge, "source_sha256", None),
                               "bridge_source_path": getattr(bridge, "source_path", None),
                               "cublaslt_version": getattr(bridge, "cublaslt_version", None),
                               "records": records}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
