"""Validate explicit shape-selected algorithms before packaging the candidate."""

import datetime
import fcntl
import json
import os
from pathlib import Path

import torch
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data.workload import ToleranceSpec

from cublaslt_loader import load_bridge


def main():
    root = Path(__file__).resolve().parents[3]
    bridge = load_bridge()
    # Chosen from the bounded sweeps; this maps only matrix shape to algorithm.
    indices = {128: 2, 256: 1, 512: 0, 1024: 1, 1571: 1,
               2048: 0, 2164: 1, 4096: 0, 8192: 2}
    tolerance = ToleranceSpec(max_atol=1e-5, max_rtol=0.05, required_matched_ratio=0.99)
    fields = ("id", "tile", "split_k", "reduction", "swizzle", "custom", "stages", "inner_shape", "cluster_shape")
    records = []
    with Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock"))).open("a+") as lock:
        print("Waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.manual_seed(66031)
        for m, index in indices.items():
            a = torch.randn((m, 5120), dtype=torch.bfloat16, device="cuda")
            b = torch.randn((1024, 5120), dtype=torch.bfloat16, device="cuda")
            c = torch.empty((m, 1024), dtype=torch.bfloat16, device="cuda")
            algorithm = next(record for record in bridge.algorithms(a, b, c, 32 * 1024 * 1024)
                             if record["index"] == index)
            configuration = [algorithm[field] for field in fields]
            workspace = torch.empty(algorithm["workspace_bytes"], dtype=torch.uint8, device="cuda")
            expected = torch.mm(a, b.t())
            bridge.matmul_config(a, b, c, workspace, configuration)
            torch.cuda.synchronize()
            stats, exceeds = compute_error_stats(c, expected, tolerance)
            if exceeds:
                raise AssertionError(f"M={m} explicit config fails: {stats}")
            latency = time_runnable(
                lambda aa, bb, cc: bridge.matmul_config(aa, bb, cc, workspace, configuration),
                [a, b], [c], "cuda:0", warmup=3, rep=20, seed=723,
            )
            record = {"m": m, "n": 1024, "k": 5120, "algorithm": algorithm,
                      "configuration": configuration, "workspace_bytes": workspace.numel(),
                      "passed": True, "errors": stats.model_dump(), "latency_ms": latency}
            print(json.dumps(record), flush=True)
            records.append(record)
    target = root / ".work/tuning/cublaslt/selected-value-projection-v2.json"
    target.write_text(json.dumps({"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  "bridge_source_sha256": bridge.source_sha256,
                                  "bridge_source_path": bridge.source_path,
                                  "cublaslt_version": bridge.cublaslt_version,
                                  "records": records}, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
