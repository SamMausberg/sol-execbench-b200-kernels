"""Compare launch geometry with cold L2; official validation uses campaign.py."""

import argparse
import datetime
import fcntl
import json
from pathlib import Path
import statistics

import torch
import triton
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti

from kernel import _rope_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=int, action="append")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    workloads = [
        json.loads(line)
        for line in (root / ".work/problems/88/workload.jsonl").read_text().splitlines()
    ]
    indices = args.workload or [0, 1, 8, 14]
    configs = [(128, 4), (256, 4), (512, 4), (1024, 4), (2048, 4),
               (512, 8), (1024, 8), (2048, 8), (4096, 8)]
    output = {"clock_mode": "unlocked", "purpose": "launch tuning",
              "methodology": "official CUPTI helper, cold L2", "workloads": []}

    with (root / ".work/gpu.lock").open("a+") as lock:
        print("Waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.manual_seed(2026)
        for index in indices:
            axes = workloads[index]["axes"]
            batch, heads, kv_heads, seq = (
                axes[k] for k in ("batch_size", "num_heads", "num_kv_heads", "seq_len")
            )
            query = torch.randn((batch, heads, seq, 128), device="cuda")
            key = torch.randn((batch, kv_heads, seq, 128), device="cuda")
            cos = torch.randn((seq, 128), device="cuda")
            sin = torch.randn_like(cos)
            query_out, key_out = torch.empty_like(query), torch.empty_like(key)
            q_pairs, k_pairs = query.numel() // 2, key.numel() // 2
            measurements = []
            for block, warps in configs:
                def launch():
                    return _rope_pairs[(triton.cdiv(q_pairs, block),)](
                        query, key, cos, sin, query_out, key_out,
                        q_pairs, k_pairs, seq, 64, block,
                        num_warps=warps, enable_fp_fusion=False,
                    )

                compiled = launch()
                latencies = [value * 1000 for value in bench_gpu_time_with_cupti(
                    launch, warmup=3, rep=args.iterations, cold_l2_cache=True,
                )]
                measurements.append({
                    "block": block, "warps": warps,
                    "median_us": statistics.median(latencies),
                    "mean_us": statistics.mean(latencies),
                    "registers": compiled.n_regs,
                    "spills": compiled.n_spills,
                })
            measurements.sort(key=lambda row: row["median_us"])
            record = {"index": index, "axes": axes, "measurements": measurements}
            output["workloads"].append(record)
            print(json.dumps(record), flush=True)
            del query, key, cos, sin, query_out, key_out

    directory = root / ".work/tuning/88"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{stamp}.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
