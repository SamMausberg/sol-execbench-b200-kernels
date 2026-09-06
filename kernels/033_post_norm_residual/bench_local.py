"""Tune the fused RMSNorm kernel with official inputs and cold L2 CUPTI timing."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

import kernel
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti
from sol_execbench.core.data import Definition, Workload
from sol_execbench.core.data.workload import ToleranceSpec


def check_edges(reference):
    set_seed(913)
    dtype = torch.bfloat16
    # Exact variance 2.5 makes this distinguish the required BF16 intermediate
    # from an otherwise plausible FP32 normalization-plus-residual fusion.
    alternating = ((torch.arange(4096, device="cuda") & 1) + 1).to(dtype)
    alternating = alternating[None, None, :].expand(2, 7, -1).contiguous()
    ones = torch.ones(4096, device="cuda", dtype=dtype)
    normalized = reference(alternating, torch.zeros_like(alternating), ones, 1e-6)
    random_x = torch.randn((2, 7, 4096), device="cuda", dtype=dtype)
    random_r = torch.randn_like(random_x)
    cases = [
        ("bf16_round_before_cancellation", alternating, -normalized, ones, 1e-6, torch.zeros_like(alternating)),
        ("zero_input", torch.zeros_like(random_x), random_r, ones, 1e-6, random_r),
        ("zero_weight", random_x, random_r, torch.zeros_like(ones), 1e-6, random_r),
        ("strided_nondefault_epsilon",
         torch.randn((2, 4096, 7), device="cuda", dtype=dtype).transpose(1, 2),
         torch.randn((2, 4096, 7), device="cuda", dtype=dtype).transpose(1, 2),
         torch.randn((8192,), device="cuda", dtype=dtype)[::2], 1.0, None),
    ]
    tolerance = ToleranceSpec(max_atol=1e-5, max_rtol=0.05)
    results = []
    for name, x, residual, weight, eps, exact in cases:
        expected = reference(x, residual, weight, eps)
        output = torch.empty_like(x)
        kernel.run(x, residual, weight, eps, output)
        error, exceeds = compute_error_stats(output, expected, tolerance)
        if exceeds or (exact is not None and not torch.equal(output, exact)):
            raise RuntimeError(f"Edge case {name} failed: {error}")
        results.append({"case": name, "passed": True, "correctness": error.model_dump()})
    print(f"{len(results)} rounding/layout/scalar edge cases passed", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("--indices", default="0,6,7,10")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warps", default="4,8")
    parser.add_argument("--prefetch", choices=("all", "yes", "no"), default="all")
    parser.add_argument("--edges-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    definition = Definition(**json.loads((args.problem / "definition.json").read_text()))
    workloads = [Workload(**json.loads(line)) for line in
                 (args.problem / "workload.jsonl").read_text().splitlines() if line.strip()]
    namespace = {}
    exec(definition.reference, namespace)
    if args.edges_only:
        results = check_edges(namespace["run"])
        if args.output:
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        return
    results = []
    for index in map(int, args.indices.split(",")):
        workload = workloads[index]
        set_seed(200)
        x, residual, weight, eps = gen_inputs(definition, workload, "cuda")
        output = torch.empty_like(x)
        expected = namespace["run"](x, residual, weight, eps)
        batch, sequence, width = x.shape
        row = {"index": index, "shape": list(x.shape), "variants": {}}
        prefetch_options = (False, True) if args.prefetch == "all" else (args.prefetch == "yes",)
        for warps in map(int, args.warps.split(",")):
            for prefetch in prefetch_options:
                def call():
                    return kernel._post_norm[(batch * sequence,)](
                        x, residual, weight, output, eps, width, sequence,
                        *x.stride(), *residual.stride(), weight.stride(0),
                        *output.stride(), triton.next_power_of_2(width),
                        PREFETCH_RESIDUAL=prefetch, num_warps=warps,
                        enable_fp_fusion=False,
                    )
                call()
                error, exceeds = compute_error_stats(output, expected, workload.tolerance)
                label = f"warps_{warps}_prefetch_{prefetch}"
                row["variants"][label] = {"correctness": error.model_dump(), "passed": not exceeds}
                if exceeds:
                    print(index, label, "FAILED", error, flush=True)
                    continue
                times = bench_gpu_time_with_cupti(call, warmup=3, rep=args.iterations, cold_l2_cache=True)
                latency_us = statistics.fmean(times) * 1000
                row["variants"][label]["latency_us"] = latency_us
                print(index, list(x.shape), label, f"{latency_us:.4f} us", flush=True)
        results.append(row)
        del x, residual, weight, output, expected
        torch.cuda.empty_cache()
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
