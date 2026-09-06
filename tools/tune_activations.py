#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sweep launch configurations using the evaluator's cold-cache CUPTI timer.

This is an exploratory tuner. Promote a choice only after running every
workload through the unmodified official evaluator and its input checks.
"""

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import statistics

import torch
import triton
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti


ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    25: "025_video_latent_gelu_activation",
    84: "084_silu_activation_backward",
    85: "085_geglu_activation",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=int, choices=NAMES)
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--source", type=Path, help="alternate candidate module")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.source or ROOT / "kernels" / NAMES[args.problem] / "kernel.py"
    spec = importlib.util.spec_from_file_location("activation_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sizes = args.sizes or ({
        84: [131, 4096, 40960, 655360, 4194304, 16777216],
        25: [1048576, 4194304, 32669696, 134217728],
        85: [655360, 2621440, 10485760, 41943040],
    }[args.problem])
    if any(n <= 0 for n in sizes):
        parser.error("sizes must be positive")
    if args.problem == 85 and any(n % 5120 for n in sizes):
        parser.error("GEGLU output sizes must be divisible by inner_dim=5120")
    results = []
    lock = Path(os.environ.get("SOL_GPU_LOCK", "/workspace/sol-execbench-b200-kernels/.work/gpu.lock"))
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        torch.manual_seed(821)
        for n in sizes:
            out = torch.empty(n, device="cuda")
            x = torch.randn(n if args.problem != 85 else (n // 5120, 10240), device="cuda")
            if args.problem == 84:
                dy = torch.randn_like(x)
                sig = torch.randn_like(x)
                reference = dy * (sig * (1.0 + x * (1.0 - sig)))
                kernel, params = module._silu_backward, (dy, x, sig, out, n)
            elif args.problem == 25:
                inner = 0.7978845608028654 * (x + 0.044715 * (x * x * x))
                reference = (0.5 * x) * (1.0 + torch.tanh(inner))
                kernel, params = module._gelu, (x, out, n)
            else:
                gate, linear = x.chunk(2, dim=-1)
                reference = (torch.nn.functional.gelu(gate, approximate="tanh") * linear).flatten()
                kernel, params = module._geglu, (x, out, 5120, n)
            for block in [256, 512, 1024, 2048, 4096]:
                for warps in [4, 8, 16] if block >= 1024 else [1, 2, 4, 8]:
                    def launch():
                        kernel[(triton.cdiv(n, block),)](
                            *params, block, num_warps=warps, enable_fp_fusion=False,
                        )
                    out.fill_(float("nan"))
                    launch()
                    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)
                    timings = bench_gpu_time_with_cupti(launch, warmup=3, rep=15)
                    ms = statistics.median(timings)
                    result = {"n": n, "block": block, "num_warps": warps, "latency_ms": ms}
                    results.append(result)
                    print(json.dumps(result), flush=True)
            del out, x, reference
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
