"""Bounded Lt/CuTe comparison with official input generation and error metrics."""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess

import torch

import cute_kernel
import lt_local
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti
from sol_execbench.core.data import Definition, Workload

CUTE_CONFIGS = [(128,128,1,1,False), (128,256,1,1,False),
                (256,128,2,1,True), (256,256,2,1,True), (128,256,1,2,False)]
FIELDS = ['id','tile','split_k','reduction','swizzle','custom','stages','inner_shape','cluster_shape']


def processes():
    rows = subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
         '--format=csv,noheader,nounits'], text=True).splitlines()
    data = []
    for row in rows:
        pid, name, memory = row.split(',', 2)
        data.append({'pid':int(pid), 'process_basename':Path(name.strip()).name,
                     'memory_mib':memory.strip(), 'is_this_process':int(pid)==os.getpid()})
    return {'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'processes':data}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problem', type=Path)
    parser.add_argument('--indices', default='3,6,4,2')
    parser.add_argument('--heuristics', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-cute', action='store_true')
    parser.add_argument('--stage2', action='store_true')
    args = parser.parse_args()
    definition = Definition(**json.loads((args.problem/'definition.json').read_text()))
    workloads = [Workload(**json.loads(row)) for row in
                 (args.problem/'workload.jsonl').read_text().splitlines() if row.strip()]
    namespace = {}
    exec(definition.reference, namespace)
    module = lt_local.module()
    report = {'cublaslt_version': module.cublaslt_version, 'workloads': []}
    for index in map(int, args.indices.split(',')):
        workload = workloads[index]
        set_seed(915)
        inputs = list(gen_inputs(definition, workload, 'cuda',
                      custom_inputs_fn=namespace.get(definition.custom_inputs_entrypoint)))
        hidden, weight = inputs
        expected = namespace['run'](*inputs)
        output = torch.empty_like(expected)
        a, c = hidden.view(-1, hidden.shape[-1]), output.view(-1, weight.shape[0])
        workspace = torch.empty(32*1024*1024, dtype=torch.uint8, device=hidden.device)
        algorithms = module.algorithms(a, weight, c, workspace.numel())
        row = {'index':index, 'axes':workload.axes, 'matrix_rows':a.shape[0],
               'before':processes(), 'heuristics':algorithms, 'candidates':{}}
        candidates = {'torch': lambda: torch.mm(a,weight.t(),out=c)}
        for algorithm in algorithms[:args.heuristics]:
            configuration = [algorithm[field] for field in FIELDS]
            candidates[f"lt_{algorithm['index']}"] = (
                lambda conf=configuration: module.matmul_config(a,weight,c,workspace,conf))
        if not args.skip_cute:
            for config in CUTE_CONFIGS:
                candidates['cute_'+'_'.join(map(str,config))] = (
                    lambda conf=config: cute_kernel.configured_run(hidden,weight,output,*conf))
        if args.stage2:
            candidates = {'torch':lambda: torch.mm(a,weight.t(),out=c)}
            base = [algorithms[0][field] for field in FIELDS]
            candidates['lt_default'] = lambda: module.matmul_config(a,weight,c,workspace,base)
            for custom, cluster in ((0,2),(1,2),(2,3),(3,3),(1,6),(3,6)):
                conf = base.copy()
                conf[5],conf[8] = custom,cluster
                if conf == base:
                    continue
                candidates[f'lt_custom{custom}_cluster{cluster}'] = (
                    lambda cfg=conf: module.matmul_config(a,weight,c,workspace,cfg))
            for config in ((128,128,1,1,False,True,False),
                           (128,256,1,1,False,True,False),
                           (256,256,2,1,True,True,False)):
                candidates['transposed_'+'_'.join(map(str,config))] = (
                    lambda cfg=config: cute_kernel.configured_run(hidden,weight,output,*cfg))
        for name, call in candidates.items():
            try:
                call()
                error, exceeds = compute_error_stats(output, expected, workload.tolerance)
                data = {'passed':not exceeds, 'correctness':error.model_dump()}
                if data['passed']:
                    data['latency_us'] = statistics.fmean(bench_gpu_time_with_cupti(
                        call,warmup=3,rep=args.iterations,cold_l2_cache=True))*1000
            except Exception as error:
                data = {'passed':False,'error':str(error)}
            row['candidates'][name] = data
            print(json.dumps({'index':index, 'candidate':name,
                              **{k:v for k,v in data.items() if k!='correctness'}}),flush=True)
            args.output.write_text(json.dumps({**report,'active_workload':row},indent=2)+'\n')
        row['after'] = processes()
        report['workloads'].append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        del candidates, inputs, hidden, weight, expected, output, a, c, workspace
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
