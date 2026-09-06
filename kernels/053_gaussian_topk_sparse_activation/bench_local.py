"""Inspect correctness and tune row launch sizes on official workload inputs.

Run with the evaluator environment and an externally acquired GPU file lock:
  flock ../../.work/gpu.lock /workspace/venvs/sol-execbench/bin/python \
      bench_local.py /path/to/official/problem --tune --output tune.json

This development sweep uses the evaluator's input generator, tolerance checks,
and cold-L2 CUPTI timer. Final validation still uses the unmodified evaluator CLI.
"""

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
    """Exercise scalar branches and layout behavior absent from public workloads."""
    set_seed(817)
    cases = {
        "constant": torch.full((2, 7, 511), 3.0, device="cuda", dtype=torch.bfloat16),
        "strided": torch.randn((2, 511, 7), device="cuda", dtype=torch.bfloat16).transpose(1, 2),
        "shifted": (torch.randn((2, 7, 511), device="cuda") + 256).bfloat16(),
        "single_feature": torch.randn((2, 7, 1), device="cuda", dtype=torch.bfloat16),
        "wide_constant": torch.full((1, 2, 12288), 3.0, device="cuda", dtype=torch.bfloat16),
        "wide_shifted": (torch.randn((1, 2, 12288), device="cuda") + 256).bfloat16(),
    }
    tolerance = ToleranceSpec(max_atol=1e-5, max_rtol=0.05)
    for name, inputs in cases.items():
        for sparsity in (0.0, 0.001, 0.02425, 0.5, 0.97575, 0.999):
            output = torch.empty_like(inputs)
            expected = reference(inputs, sparsity)
            kernel.run(inputs, sparsity, output)
            error, exceeds = compute_error_stats(output, expected, tolerance)
            if exceeds or (sparsity == 0.0 and not torch.equal(output, inputs)):
                raise RuntimeError(f"Edge case {name}, sparsity {sparsity} failed: {error}")
    print(f"{len(cases) * 6} scalar/layout/numerical edge cases passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument("--edges", action="store_true")
    parser.add_argument("--indices", type=str)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    definition = Definition(**json.loads((args.problem / "definition.json").read_text()))
    workloads = [Workload(**json.loads(line)) for line in
                 (args.problem / "workload.jsonl").read_text().splitlines() if line.strip()]
    namespace = {}
    exec(definition.reference, namespace)
    reference = namespace["run"]
    if args.edges:
        check_edges(reference)
    indices = set(map(int, args.indices.split(","))) if args.indices else set(range(len(workloads)))
    results = []
    for index, workload in enumerate(workloads):
        if index not in indices:
            continue
        set_seed(200)
        inputs, sparsity = gen_inputs(definition, workload, "cuda")
        output = torch.empty_like(inputs)
        expected = reference(inputs, sparsity)
        kernel.run(inputs, sparsity, output)
        error, exceeds = compute_error_stats(output, expected, workload.tolerance)
        if exceeds:
            raise RuntimeError(f"Workload {index} failed: {error}")
        shape = list(inputs.shape)
        row = {"index": index, "shape": shape, "target_sparsity": sparsity,
               "correctness": error.model_dump(), "timings_us": {}}
        if args.experiments and sparsity != 0:
            warps = 4 if shape[-1] <= 4096 else 8
            variants = [(warps, moments, cache, store)
                        for moments in (False, True)
                        for cache, store in (("", ""), (".cg", ".cs"))]
        elif args.tune and sparsity != 0:
            variants = [(warps, False, "", "") for warps in (4, 8, 16, 32)]
        else:
            variants = [(None, False, "", "")]
        for warps, moments, cache, store in variants:
            if warps is None:
                call = lambda: kernel.run(inputs, sparsity, output)
                label = "current"
            else:
                batch, sequence, width = shape
                multiplier = kernel._normal_quantile_f32(sparsity)
                block = triton.next_power_of_2(width)

                def call():
                    return kernel._activation[(batch * sequence,)](
                        inputs, output, multiplier, width, sequence,
                        *inputs.stride(), *output.stride(), block,
                        MOMENTS=moments, LOAD_CACHE=cache, STORE_CACHE=store,
                        num_warps=warps, enable_fp_fusion=False,
                    )
                label = f"warps_{warps}"
                if args.experiments:
                    label += f"_moments_{moments}_cache_{cache or 'default'}_store_{store or 'default'}"
            call()
            torch.cuda.synchronize()
            variant_error, variant_exceeds = compute_error_stats(output, expected, workload.tolerance)
            if variant_exceeds:
                row["timings_us"][label] = {"error": variant_error.model_dump()}
                continue
            latency = bench_gpu_time_with_cupti(call, warmup=3, rep=args.iterations)
            if isinstance(latency, list):
                latency = statistics.fmean(latency)
            row["timings_us"][label] = latency * 1000
            print(index, shape, sparsity, label, f"{latency * 1000:.4f} us", flush=True)
        results.append(row)
        del inputs, output, expected
        torch.cuda.empty_cache()
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
