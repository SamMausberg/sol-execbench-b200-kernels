#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare GEGLU launch geometry and tanh implementations under cold-cache timing."""

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
spec = importlib.util.spec_from_file_location("geglu", ROOT / "kernels/085_geglu_activation/kernel.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
results = []
lock = Path(os.environ.get("SOL_GPU_LOCK", "/workspace/sol-execbench-b200-kernels/.work/gpu.lock"))
with lock.open("a") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    torch.manual_seed(821)
    for rows in [128, 512, 2048, 8192]:
        x = torch.randn(rows, 10240, device="cuda")
        out = torch.empty(rows, 5120, device="cuda")
        gate, linear = x.chunk(2, dim=1)
        ref = torch.nn.functional.gelu(gate, approximate="tanh") * linear
        for block in [256, 512, 1024, 2048]:
            for warps in [4, 8]:
                for mode in ["flat", "rows", "rows_exp"]:
                    def launch():
                        if mode == "flat":
                            module._geglu[(triton.cdiv(rows * 5120, block),)](
                                x, out, 5120, rows * 5120, block,
                                num_warps=warps, enable_fp_fusion=False,
                            )
                        else:
                            module._geglu_rows[(rows, triton.cdiv(5120, block))](
                                x, out, 5120, block, 2 if mode == "rows_exp" else 0,
                                num_warps=warps, enable_fp_fusion=False,
                            )
                    out.fill_(float("nan"))
                    launch()
                    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
                    times = bench_gpu_time_with_cupti(launch, warmup=3, rep=15)
                    row = {"rows": rows, "block": block, "num_warps": warps, "mode": mode, "latency_ms": statistics.median(times)}
                    print(json.dumps(row), flush=True)
                    results.append(row)
        del x, out, gate, linear, ref
(ROOT / ".work/tuning/85-rows.json").write_text(json.dumps(results, indent=2) + "\n")
