"""Check grouped QKV projection and measure launch layouts with CUPTI."""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch

import kernel
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti
from sol_execbench.core.bench.timing import GPU_TIMING_ACTIVITY_KINDS
from sol_execbench.core.bench.cupti_utils import collect_cupti_activities
from sol_execbench.core.data import Definition, Workload


CONFIGS = [
    (16, 64, 128, 4, 3, 4), (32, 64, 128, 4, 3, 4),
    (32, 128, 128, 4, 3, 4), (64, 64, 128, 4, 3, 4),
    (64, 128, 128, 4, 3, 4), (64, 128, 64, 4, 3, 4),
    (128, 64, 64, 4, 3, 4), (128, 128, 64, 8, 3, 4),
    (128, 128, 128, 8, 3, 4), (32, 128, 64, 4, 3, 4),
    (16, 128, 128, 4, 3, 4), (32, 256, 64, 4, 3, 4),
]


def gpu_processes():
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True).splitlines()
    result = []
    for row in rows:
        pid, name, memory = row.split(",", 2)
        result.append({"pid": int(pid), "process_basename": Path(name.strip()).name,
                       "memory_mib": memory.strip(), "is_this_process": int(pid) == os.getpid()})
    return result


def change_inputs(inputs, variant):
    if variant == "random":
        return
    if variant == "cancel_bias":
        inputs[0].fill_(1.125)
        for weight_index, bias_index in ((1, 2), (3, 4), (5, 6)):
            width = inputs[weight_index].shape[0]
            row = (1.0 + (torch.arange(width, device="cuda") % 7) / 128).to(torch.bfloat16)
            inputs[weight_index].copy_(row[:, None])
            inputs[bias_index].copy_(-(row.float() * 720).to(torch.bfloat16))
    elif variant == "zero_input":
        inputs[0].zero_()
    elif variant == "zero_weights":
        for i in (1, 3, 5):
            inputs[i].zero_()
    elif variant == "strided":
        b, s, d = inputs[0].shape
        storage = torch.empty((b, s, d * 2), dtype=torch.bfloat16, device="cuda")
        storage[..., ::2].copy_(inputs[0])
        inputs[0] = storage[..., ::2]
        for i in (1, 3, 5):
            inputs[i] = inputs[i].t().contiguous().t()
        for i in (2, 4, 6):
            storage = torch.empty(inputs[i].numel() * 2, dtype=torch.bfloat16, device="cuda")
            storage[::2].copy_(inputs[i])
            inputs[i] = storage[::2]
    else:
        raise ValueError(variant)


def check(outputs, expected, tolerance, exact_zero=False):
    metrics = []
    for output, reference in zip(outputs, expected):
        error, exceeds = compute_error_stats(output, reference, tolerance)
        metrics.append({"passed": not exceeds, "correctness": error.model_dump()})
    exact = all(torch.equal(output, reference) for output, reference in zip(outputs, expected))
    return {"passed": all(row["passed"] for row in metrics) and (not exact_zero or exact),
            "exact": exact, "outputs": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("--indices", default="13,15")
    parser.add_argument("--seeds", default="913")
    parser.add_argument("--variants", default="random")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--entry", choices=("triton", "cute"), default="cute")
    parser.add_argument("--tma", action="store_true")
    parser.add_argument("--cute", action="store_true")
    parser.add_argument("--cute-configs", default="")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--warp-specialize", action="store_true")
    parser.add_argument("--grouped-library", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    definition = Definition(**json.loads((args.problem / "definition.json").read_text()))
    workloads = [Workload(**json.loads(line)) for line in
                 (args.problem / "workload.jsonl").read_text().splitlines() if line.strip()]
    namespace = {}
    exec(definition.reference, namespace)
    reports = []
    for index in map(int, args.indices.split(",")):
        workload = workloads[index]
        for seed in map(int, args.seeds.split(",")):
            for variant in args.variants.split(","):
                set_seed(seed)
                inputs = list(gen_inputs(definition, workload, "cuda"))
                change_inputs(inputs, variant)
                expected = namespace["run"](*inputs)
                outputs = [torch.empty_like(value) for value in expected]
                if variant == "strided":
                    outputs = [torch.empty((*value.shape[:-1], value.shape[-1] * 2),
                                           dtype=value.dtype, device=value.device)[..., ::2] for value in expected]
                if args.entry == "cute":
                    import cute_kernel
                    cute_kernel.run(*inputs, *outputs)
                else:
                    kernel.run(*inputs, *outputs)
                row = {"index": index, "seed": seed, "variant": variant, "axes": workload.axes,
                       **check(outputs, expected, workload.tolerance, variant == "cancel_bias")}
                print(json.dumps(row), flush=True)
                if args.tune and row["passed"]:
                    row["gpu_processes_before"] = gpu_processes()
                    row["layouts"] = {}
                    configs = CONFIGS if not args.tma else [
                        (64, 64, 64, 4, 3, 4), (64, 128, 64, 4, 3, 4),
                        (128, 128, 64, 4, 3, 4), (128, 128, 128, 4, 3, 4),
                        (128, 128, 128, 8, 3, 4), (128, 256, 64, 8, 3, 4),
                    ]
                    if args.persistent:
                        configs = [
                            (128, 128, 64, 4, 3, 4, True, 1),
                            (128, 256, 64, 8, 3, 4, True, 1),
                            (256, 128, 64, 8, 3, 4, True, 2),
                            (256, 256, 64, 8, 3, 4, True, 2),
                            (128, 128, 64, 4, 3, 4, True, 2),
                        ]
                    if args.stages:
                        configs = [
                            (128, 256, 64, 8, 2, 4), (128, 256, 64, 4, 2, 4),
                            (128, 256, 64, 4, 1, 4), (128, 128, 64, 4, 2, 4),
                            (128, 128, 128, 4, 1, 4), (64, 128, 128, 4, 2, 4),
                            (128, 256, 128, 8, 1, 4), (128, 128, 256, 4, 1, 4),
                        ]
                    if args.warp_specialize:
                        configs = [
                            (128, 128, 64, 4, 3, 4, False, 1, True),
                            (128, 128, 128, 4, 3, 4, False, 1, True),
                            (128, 256, 64, 4, 2, 4, False, 1, True),
                            (128, 256, 64, 8, 3, 4, False, 1, True),
                        ]
                    if args.cute:
                        import cute_kernel
                        configs = [
                            (64, 64, 1, 1, False),
                            (128, 128, 1, 1, False),
                            (128, 256, 1, 1, False),
                            (256, 256, 2, 1, True),
                        ]
                    if args.cute_configs:
                        configs = json.loads(args.cute_configs)
                    for config in configs:
                        def call():
                            function = (cute_kernel.configured_run if args.cute else
                                        (kernel.launch_tma if args.tma else kernel.launch))
                            if args.cute and len(config) >= 6:
                                if config[5]:
                                    import experimental_cute_kernel
                                    return experimental_cute_kernel.configured_run(*inputs, *outputs, *config)
                                occupancy = config[6] if len(config) == 7 else 1
                                return function(*inputs, *outputs, *config[:5], occupancy)
                            return function(*inputs, *outputs, *config)
                        try:
                            call()
                            data = check(outputs, expected, workload.tolerance)
                            if data["passed"]:
                                data["latency_us"] = statistics.fmean(bench_gpu_time_with_cupti(
                                    call, warmup=3, rep=args.iterations, cold_l2_cache=True)) * 1000
                        except Exception as error:
                            data = {"passed": False, "error": str(error)}
                        row["layouts"][str(config)] = data
                        print(json.dumps({"config": config, **data}), flush=True)
                    row["gpu_processes_after"] = gpu_processes()
                if args.grouped_library and row["passed"]:
                    import grouped
                    grouped._bridge = grouped.load_bridge()
                    row["gpu_processes_before"] = gpu_processes()
                    grouped.run(*inputs, *outputs)
                    data = check(outputs, expected, workload.tolerance, variant == "cancel_bias")
                    if data["passed"]:
                        data["latency_us"] = statistics.fmean(bench_gpu_time_with_cupti(
                            lambda: grouped.run(*inputs, *outputs), warmup=3,
                            rep=args.iterations, cold_l2_cache=True)) * 1000
                        with collect_cupti_activities(activity_kinds=GPU_TIMING_ACTIVITY_KINDS) as activity:
                            grouped.run(*inputs, *outputs)
                            torch.cuda.synchronize()
                        ordered = sorted(activity.kernels, key=lambda item: item.start)
                        previous = ordered[0].start
                        data["activity_us"] = []
                        for item in ordered:
                            data["activity_us"].append({
                                "name": item.name,
                                "gap_from_previous_us": (item.start - previous) / 1000,
                                "duration_us": (item.end - item.start) / 1000,
                            })
                            previous = item.end
                    row["grouped_library"] = data
                    row["gpu_processes_after"] = gpu_processes()
                    print(json.dumps(data), flush=True)
                reports.append(row)
                if args.output:
                    args.output.write_text(json.dumps(reports, indent=2) + "\n")
                del inputs, outputs, expected
                torch.cuda.empty_cache()
    if any(not row["passed"] for row in reports):
        raise SystemExit("Input correctness check failed")


if __name__ == "__main__":
    main()
