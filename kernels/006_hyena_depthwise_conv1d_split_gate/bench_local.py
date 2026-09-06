"""Validate the fused Hyena convolution and tune launch geometry with CUPTI."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

import torch

import kernel
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti
from sol_execbench.core.data import Definition, Workload

CONFIGS = [(256, 4), (512, 4), (1024, 4), (2048, 4),
           (512, 8), (1024, 8), (2048, 8), (512, 16)]


def gpu_processes():
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
        text=True).splitlines()
    result = []
    for row in rows:
        pid, name, memory = row.split(',', 2)
        result.append({'pid': int(pid), 'process_basename': Path(name.strip()).name,
                       'memory_mib': memory.strip(), 'is_this_process': int(pid) == os.getpid()})
    return result


def mutate(inputs, variant):
    u, weight, bias = inputs
    if variant == 'random':
        return
    if variant == 'zero_input':
        u.zero_()
    elif variant == 'boundary_impulses':
        u.zero_()
        u[..., 0] = 1.25
        u[..., -1] = -0.75
    elif variant == 'cancellation':
        u.fill_(1.0009765625)
        weight[..., 0] = 10000
        weight[..., 1] = -10000
        weight[..., 2] = 1
        bias.fill_(-1.0009765625)
    elif variant == 'strided':
        storage = torch.empty((*u.shape[:-1], u.shape[-1] * 2), dtype=u.dtype, device=u.device)
        storage[..., ::2].copy_(u)
        inputs[0] = storage[..., ::2]
        storage = torch.empty((*weight.shape[:-1], 6), dtype=weight.dtype, device=weight.device)
        storage[..., ::2].copy_(weight)
        inputs[1] = storage[..., ::2]
        storage = torch.empty(bias.numel() * 2, dtype=bias.dtype, device=bias.device)
        storage[::2].copy_(bias)
        inputs[2] = storage[::2]
    else:
        raise ValueError(variant)


def check(outputs, expected, tolerance):
    metrics = []
    for output, reference in zip(outputs, expected):
        error, exceeds = compute_error_stats(output, reference, tolerance)
        metrics.append({'passed': not exceeds, 'correctness': error.model_dump()})
    return {'passed': all(row['passed'] for row in metrics),
            'exact': all(torch.equal(a, b) for a, b in zip(outputs, expected)),
            'outputs': metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problem', type=Path)
    parser.add_argument('--indices', default='13,9,4')
    parser.add_argument('--seeds', default='913')
    parser.add_argument('--variants', default='random')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cuda-scalar', action='store_true')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    definition = Definition(**json.loads((args.problem/'definition.json').read_text()))
    workloads = [Workload(**json.loads(row)) for row in
                 (args.problem/'workload.jsonl').read_text().splitlines() if row.strip()]
    namespace = {}
    exec(definition.reference, namespace)
    selected_kernel = kernel
    configs = CONFIGS
    if args.cuda or args.cuda_scalar:
        import cuda_local
        selected_kernel = cuda_local
        cuda_local.module(experiments=args.cuda_scalar)
        if args.cuda_scalar:
            from types import SimpleNamespace
            selected_kernel = SimpleNamespace(
                run=cuda_local.module(experiments=True).run_scalar,
                configured_run=cuda_local.module(experiments=True).configured_scalar_run)
        configs = [(w,v) for w in (4,8,16) for v in (1,2,4)]
    reports = []
    for index in map(int, args.indices.split(',')):
        workload = workloads[index]
        for seed in map(int, args.seeds.split(',')):
            for variant in args.variants.split(','):
                set_seed(seed)
                inputs = list(gen_inputs(definition, workload, 'cuda'))
                mutate(inputs, variant)
                expected = namespace['run'](*inputs)
                outputs = [torch.empty_like(x, memory_format=torch.contiguous_format) for x in expected]
                if variant == 'strided':
                    outputs = [torch.empty((*x.shape[:-1], x.shape[-1] * 2),
                                           dtype=x.dtype, device=x.device)[..., ::2] for x in expected]
                selected_kernel.run(*inputs, *outputs)
                row = {'index': index, 'seed': seed, 'variant': variant, 'axes': workload.axes,
                       **check(outputs, expected, workload.tolerance)}
                print(json.dumps({k: v for k, v in row.items() if k != 'outputs'}), flush=True)
                if args.tune and row['passed']:
                    row['gpu_processes_before'] = gpu_processes()
                    row['layouts'] = {}
                    for config in configs:
                        function = lambda: selected_kernel.configured_run(*inputs, *outputs, *config)
                        try:
                            function()
                            data = check(outputs, expected, workload.tolerance)
                            if data['passed']:
                                data['latency_us'] = statistics.fmean(bench_gpu_time_with_cupti(
                                    function, warmup=3, rep=args.iterations, cold_l2_cache=True)) * 1000
                        except Exception as error:
                            data = {'passed': False, 'error': str(error)}
                        row['layouts'][str(config)] = data
                        print(json.dumps({'config': config, **{k: v for k, v in data.items() if k != 'outputs'}}), flush=True)
                    row['gpu_processes_after'] = gpu_processes()
                reports.append(row)
                args.output.write_text(json.dumps(reports, indent=2) + '\n')
                del inputs, outputs, expected
                torch.cuda.empty_cache()
    if any(not row['passed'] for row in reports):
        raise SystemExit('Input correctness check failed')


if __name__ == '__main__':
    main()
