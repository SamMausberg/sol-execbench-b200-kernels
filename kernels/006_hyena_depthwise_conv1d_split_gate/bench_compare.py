"""Compare the best mathematically exact implementations in the same GPU window."""
import argparse
import json
from pathlib import Path
import statistics

import torch

import cuda_local
import kernel
from bench_local import check, gpu_processes
from sol_execbench.core.bench.correctness import set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti
from sol_execbench.core.data import Definition, Workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problem', type=Path)
    parser.add_argument('--indices', default='13,9,4')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--stage2', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    definition = Definition(**json.loads((args.problem/'definition.json').read_text()))
    workloads = [Workload(**json.loads(row)) for row in
                 (args.problem/'workload.jsonl').read_text().splitlines() if row.strip()]
    namespace = {}
    exec(definition.reference, namespace)
    module = cuda_local.module(experiments=True)
    candidates = {}
    for warps in (4, 8, 16):
        candidates[f'direct_{warps}'] = lambda a, w=warps: module.configured_direct_run(*a, w, 1)
        candidates[f'vector_{warps}'] = lambda a, w=warps: module.configured_run(*a, w, 1)
    for block, warps in ((256, 4), (512, 8), (512, 16)):
        candidates[f'triton_{block}_{warps}'] = lambda a, b=block, w=warps: kernel.configured_run(*a, b, w)
    if args.stage2:
        candidates = {
            'plain_16': lambda a: module.configured_plain_run(*a, 16, 1),
            'direct_4': lambda a: module.configured_direct_run(*a, 4, 1),
            'direct_8': lambda a: module.configured_direct_run(*a, 8, 1),
            'triton_512_16': lambda a: kernel.configured_run(*a, 512, 16),
        }
    reports = []
    for index in map(int, args.indices.split(',')):
        workload = workloads[index]
        set_seed(914)
        inputs = list(gen_inputs(definition, workload, 'cuda'))
        expected = namespace['run'](*inputs)
        outputs = [torch.empty_like(x, memory_format=torch.contiguous_format) for x in expected]
        arguments = inputs + outputs
        row = {'index': index, 'axes': workload.axes, 'gpu_processes_before': gpu_processes(), 'candidates': {}}
        for name, function in candidates.items():
            call = lambda: function(arguments)
            call()
            data = check(outputs, expected, workload.tolerance)
            if data['passed']:
                data['latency_us'] = statistics.fmean(bench_gpu_time_with_cupti(
                    call, warmup=3, rep=args.iterations, cold_l2_cache=True)) * 1000
            row['candidates'][name] = data
            print(json.dumps({'index': index, 'candidate': name,
                              **{k:v for k,v in data.items() if k != 'outputs'}}), flush=True)
        row['gpu_processes_after'] = gpu_processes()
        reports.append(row)
        args.output.write_text(json.dumps(reports, indent=2) + '\n')
        del arguments, inputs, outputs, expected
        torch.cuda.empty_cache()
    if any(not c['passed'] for r in reports for c in r['candidates'].values()):
        raise SystemExit('Candidate failed correctness')


if __name__ == '__main__':
    main()
