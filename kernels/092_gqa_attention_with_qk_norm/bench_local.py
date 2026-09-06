"""Validate new inputs and profile individual GQA stages with CUPTI."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import kernel
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti
from sol_execbench.core.data import Definition, Workload


def error_report(output, expected, tolerance):
    error, exceeds = compute_error_stats(output, expected, tolerance)
    difference = (output.float() - expected.float()).abs()
    bound = tolerance.max_atol + tolerance.max_rtol * expected.float().abs()
    return {
        "passed": not exceeds,
        "matched_ratio": float((difference <= bound).float().mean()),
        "correctness": error.model_dump(),
    }


def profile_stages(inputs, repetitions, expected, tolerance):
    hidden, qw, qb, kw, kb, vw, vb, ow, qnw, knw, cos, sin, eps = inputs
    b, s, _ = hidden.shape
    query = F.linear(hidden, qw, qb)
    key = F.linear(hidden, kw, kb)
    value = F.linear(hidden, vw, vb)
    query_rotated, key_rotated = kernel.prepare(query, key, qnw, knw, cos, sin, eps)
    q = query_rotated.view(b, s, 96, 128).transpose(1, 2)
    k = key_rotated.view(b, s, 8, 128).transpose(1, 2)
    v = value.view(b, s, 8, 128).transpose(1, 2)

    def attention():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    attended = attention()
    merged = attended.transpose(1, 2).contiguous().view(b * s, 12288)
    output = torch.empty_like(hidden)
    calls = {
        "q_projection": lambda: F.linear(hidden, qw, qb),
        "k_projection": lambda: F.linear(hidden, kw, kb),
        "v_projection": lambda: F.linear(hidden, vw, vb),
        "qk_preparation": lambda: kernel.prepare(query, key, qnw, knw, cos, sin, eps),
        "flash_attention": attention,
        "output_projection": lambda: torch.mm(merged, ow.t(), out=output.view(b * s, 4096)),
        "complete": lambda: kernel.run(*inputs, output),
    }
    report = {"flash_output_stride": list(attended.stride()), "merged_is_view": merged.data_ptr() == attended.data_ptr()}
    report["latency_us"] = {
        name: statistics.fmean(bench_gpu_time_with_cupti(
            call, warmup=3, rep=repetitions, cold_l2_cache=True)) * 1000
        for name, call in calls.items()
    }
    report["attention_backends"] = {}
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        name = str(backend)
        try:
            def candidate():
                with sdpa_kernel(backend):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            alternative = candidate()
            merged_alternative = alternative.transpose(1, 2).contiguous().view(b * s, 12288)
            torch.mm(merged_alternative, ow.t(), out=output.view(b * s, 4096))
            data = error_report(output, expected, tolerance)
            data["latency_us"] = statistics.fmean(bench_gpu_time_with_cupti(
                candidate, warmup=3, rep=repetitions, cold_l2_cache=True)) * 1000
            data["stride"] = list(alternative.stride())
            data["merge_is_view"] = alternative.data_ptr() == merged_alternative.data_ptr()
            report["attention_backends"][name] = data
        except RuntimeError as error:
            report["attention_backends"][name] = {"error": str(error)}
    return report


def modify(inputs, variant):
    if variant == "random":
        pass
    elif variant == "no_rotation":
        inputs[10].fill_(1)
        inputs[11].zero_()
    elif variant == "zero_q_norm":
        inputs[8].zero_()
    elif variant == "zero_values":
        inputs[5].zero_()
        inputs[6].zero_()
    elif variant == "small_input_epsilon":
        inputs[0].mul_(0.001)
        inputs[2].zero_()
        inputs[4].zero_()
        inputs[-1] = 0.01
    elif variant == "strong_q_norm":
        inputs[8].mul_(4)
    else:
        raise ValueError(variant)


def tune_preparation(inputs, repetitions, expected, tolerance):
    hidden, qw, qb, kw, kb, vw, vb, ow, qnw, knw, cos, sin, eps = inputs
    b, s = hidden.shape[:2]
    query = F.linear(hidden, qw, qb)
    key = F.linear(hidden, kw, kb)
    value = F.linear(hidden, vw, vb).view(b, s, 8, 128).transpose(1, 2)
    expected_q, expected_k = kernel.prepare(query, key, qnw, knw, cos, sin, eps)
    result = {}
    for heads in (8, 16, 32):
        for warps in (4, 8):
            def call():
                return kernel.prepare(query, key, qnw, knw, cos, sin, eps, heads, warps)
            q, k = call()
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                attended = F.scaled_dot_product_attention(
                    q.view(b, s, 96, 128).transpose(1, 2),
                    k.view(b, s, 8, 128).transpose(1, 2), value,
                    is_causal=True, enable_gqa=True)
            output = F.linear(attended.transpose(1, 2).contiguous().view(b, s, 12288), ow)
            data = error_report(output, expected, tolerance)
            data["same_preparation_bits"] = torch.equal(q, expected_q) and torch.equal(k, expected_k)
            data["latency_us"] = statistics.fmean(
                bench_gpu_time_with_cupti(call, warmup=3, rep=repetitions, cold_l2_cache=True)) * 1000
            result[f"heads_{heads}_warps_{warps}"] = data
    return result


def tune_attention(inputs, repetitions, expected, tolerance):
    hidden, qw, qb, kw, kb, vw, vb, ow, qnw, knw, cos, sin, eps = inputs
    query = F.linear(hidden, qw, qb)
    key = F.linear(hidden, kw, kb)
    value = F.linear(hidden, vw, vb)
    query, key = kernel.prepare(query, key, qnw, knw, cos, sin, eps)
    result = {}
    configs = [
        (64, 64, 1, False, 4), (64, 64, 12, False, 4),
        (64, 64, 1, True, 4), (64, 64, 12, True, 4),
        (64, 128, 1, True, 4), (64, 128, 12, True, 4),
        (32, 256, 12, True, 4), (32, 256, 1, True, 4),
    ]
    for config in configs:
        def call():
            return kernel.attention(query, key, value, *config)
        attended = call()
        output = F.linear(attended, ow)
        data = error_report(output, expected, tolerance)
        data["latency_us"] = statistics.fmean(bench_gpu_time_with_cupti(
            call, warmup=3, rep=repetitions, cold_l2_cache=True)) * 1000
        result[str(config)] = data
        print(json.dumps({"config": config, **data}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("--indices", default="0,5,12,13")
    parser.add_argument("--seeds", default="913")
    parser.add_argument("--variants", default="random")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--tune-preparation", action="store_true")
    parser.add_argument("--tune-attention", action="store_true")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    definition = Definition(**json.loads((args.problem / "definition.json").read_text()))
    workloads = [Workload(**json.loads(line)) for line in
                 (args.problem / "workload.jsonl").read_text().splitlines() if line.strip()]
    reference_namespace = {}
    exec(definition.reference, reference_namespace)
    results = []
    for index in map(int, args.indices.split(",")):
        workload = workloads[index]
        for seed in map(int, args.seeds.split(",")):
            for variant in args.variants.split(","):
                set_seed(seed)
                inputs = list(gen_inputs(definition, workload, "cuda"))
                modify(inputs, variant)
                expected = reference_namespace["run"](*inputs)
                output = torch.empty_like(expected)
                kernel.run(*inputs, output)
                row = {"index": index, "seed": seed, "variant": variant,
                       "axes": workload.axes,
                       **error_report(output, expected, workload.tolerance)}
                print(json.dumps(row), flush=True)
                if args.profile and row["passed"]:
                    row["profile"] = profile_stages(inputs, args.iterations, expected, workload.tolerance)
                    print(json.dumps(row["profile"]), flush=True)
                if args.tune_preparation and row["passed"]:
                    row["preparation_latency_us"] = tune_preparation(inputs, args.iterations, expected, workload.tolerance)
                    print(json.dumps(row["preparation_latency_us"]), flush=True)
                if args.tune_attention and row["passed"]:
                    row["attention_variants"] = tune_attention(inputs, args.iterations, expected, workload.tolerance)
                results.append(row)
                del inputs, expected, output
                torch.cuda.empty_cache()
                if args.output:
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if any(not row["passed"] for row in results):
        raise SystemExit("Input variation check failed")


if __name__ == "__main__":
    main()
