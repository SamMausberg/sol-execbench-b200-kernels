# SPDX-License-Identifier: Apache-2.0
"""Check explicit cuBLASLt FAST_TF32 against actual problem 36 inputs."""

import datetime
import argparse
import fcntl
import json
import os
from pathlib import Path
import runpy

import torch
import torch.nn.functional as F
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "emulated"], default="fast")
    parser.add_argument("--scale-weights", type=float, default=1.0)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    if args.mode == "fast":
        from fast_tf32_loader import load_bridge
    else:
        from emulated_bf16x9_loader import load_bridge
    bridge = load_bridge()
    root = Path(__file__).resolve().parents[3]
    problem = root / ".work/problems/36"
    definition = Definition.model_validate_json((problem / "definition.json").read_text())
    workloads = [Workload.model_validate_json(line) for line in (problem / "workload.jsonl").read_text().splitlines()]
    reference = runpy.run_path(str(problem / "reference.py"))["run"]
    records = []
    with Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock"))).open("a+") as lock:
        print("Waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        scratch = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        for index in [0, len(workloads) - 1]:
            inputs = gen_inputs(definition, workloads[index], "cuda")
            hidden, temb, linear_weight, linear_bias, output_weight, output_bias, eps = inputs
            if args.scale_weights != 1.0:
                linear_weight.mul_(args.scale_weights)
                output_weight.mul_(args.scale_weights)
            expected = reference(*inputs)
            mean = hidden.mean(-1, keepdim=True)
            variance = hidden.var(-1, keepdim=True, unbiased=False)
            norm = (hidden - mean) / torch.sqrt(variance + eps)
            activated = temb * torch.sigmoid(temb)
            def fast_mm(a, b):
                c = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
                bridge.matmul(a, b, c, scratch, 0)
                return c
            for fast_modulation, fast_output in [(True, False), (False, True), (True, True)]:
                modulation = (fast_mm(activated, linear_weight) + linear_bias) if fast_modulation else F.linear(activated, linear_weight, linear_bias)
                shift, scale = modulation.chunk(2, -1)
                modulated = norm * (1.0 + scale[:, None, :]) + shift[:, None, :]
                projected = (fast_mm(modulated.flatten(0, 1), output_weight) + output_bias).view(*hidden.shape[:2], 64) if fast_output else F.linear(modulated, output_weight, output_bias)
                errors, failed = compute_error_stats(projected, expected, workloads[index].tolerance)
                delta = (projected - expected).abs()
                tol = workloads[index].tolerance
                matched = (delta <= tol.max_atol + tol.max_rtol * expected.abs()).float().mean().item()
                record = {"workload": index, "axes": workloads[index].axes, "fast_modulation": fast_modulation,
                          "fast_output": fast_output, "passed": not failed, "matched_ratio": matched,
                          "errors": errors.model_dump()}
                records.append(record)
                print(json.dumps(record), flush=True)
            if args.skip_timing:
                continue
            for label, a, b in [("modulation", activated, linear_weight), ("output", modulated.flatten(0, 1), output_weight)]:
                c = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
                candidates = bridge.algorithms(a, b, c, scratch.numel())
                native = time_runnable(lambda left, right, out: torch.mm(left, right.t(), out=out), [a, b], [c], "cuda:0", warmup=3, rep=20, seed=173)
                timings = []
                for candidate in candidates[:4]:
                    algorithm_index = candidate["index"]
                    elapsed = time_runnable(lambda left, right, out: bridge.matmul(left, right, out, scratch, algorithm_index),
                                            [a, b], [c], "cuda:0", warmup=3, rep=20, seed=173)
                    timings.append({**candidate, "latency_ms": elapsed})
                record = {"workload": index, "component": label, "shape_mnk": [a.shape[0], b.shape[0], a.shape[1]],
                          "native_latency_ms": native, "candidates": timings}
                records.append(record)
                print(json.dumps(record), flush=True)
    target = root / ".work/tuning/cublaslt"
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"{args.mode}-fp32-36-scale{args.scale_weights:g}-{stamp}.json"
    path.write_text(json.dumps({"source_sha256": bridge.source_sha256, "source_path": bridge.source_path,
                               "cublaslt_version": bridge.cublaslt_version,
                               "compute_mode": "CUBLAS_COMPUTE_32F_FAST_TF32" if args.mode == "fast" else "CUBLAS_COMPUTE_32F_EMULATED_16BFX9",
                               "weight_scale": args.scale_weights, "records": records}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
