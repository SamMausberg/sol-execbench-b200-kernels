"""Check fresh values, cancellation, ownership and actual tensor layouts."""
import argparse
import json
from pathlib import Path

import torch

from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.data import Definition, Workload


def change(inputs, variant):
    hidden, weight = inputs
    if variant == 'random':
        pass
    elif variant == 'zero_input':
        hidden.zero_()
    elif variant == 'zero_weights':
        weight.zero_()
    elif variant == 'basis':
        hidden.zero_()
        hidden.view(-1,hidden.shape[-1])[:,17] = 0.5
    elif variant == 'cancellation':
        hidden[...,::2] = 1
        hidden[...,1::2] = -1
        weight[:,::2] = 128
        weight[:,1::2] = 129
    elif variant == 'strided':
        for i, value in enumerate(inputs):
            storage = torch.empty((*value.shape[:-1],value.shape[-1]*2),
                                  dtype=value.dtype,device=value.device)
            storage[...,::2].copy_(value)
            inputs[i] = storage[...,::2]
    elif variant == 'shifted':
        for i, value in enumerate(inputs):
            storage = torch.empty(value.numel()+8,dtype=value.dtype,device=value.device)
            shifted = storage[8:].view(value.shape)
            shifted.copy_(value)
            inputs[i] = shifted
    else:
        raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problem',type=Path)
    parser.add_argument('--backend',choices=['cute','lt'],required=True)
    parser.add_argument('--indices',default='3,6,2')
    parser.add_argument('--variants',default='random,zero_input,zero_weights,basis,cancellation,strided,shifted')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.backend == 'cute':
        import cute_kernel
        function = cute_kernel.run
    else:
        import lt_local
        function = lt_local.module().run
    definition = Definition(**json.loads((args.problem/'definition.json').read_text()))
    workloads = [Workload(**json.loads(row)) for row in
                 (args.problem/'workload.jsonl').read_text().splitlines() if row.strip()]
    namespace = {}
    exec(definition.reference,namespace)
    report = []
    for index in map(int,args.indices.split(',')):
        workload = workloads[index]
        for variant_index, variant in enumerate(args.variants.split(',')):
            seed = 2027 + variant_index
            set_seed(seed)
            inputs = list(gen_inputs(definition,workload,'cuda',
                          custom_inputs_fn=namespace.get(definition.custom_inputs_entrypoint)))
            change(inputs,variant)
            original = [x.clone() for x in inputs]
            expected = namespace['run'](*inputs)
            output = function(*inputs)
            error, exceeds = compute_error_stats(output,expected,workload.tolerance)
            unchanged = all(torch.equal(a,b) for a,b in zip(inputs,original))
            row = {'index':index,'seed':seed,'variant':variant,'axes':workload.axes,
                   'passed':not exceeds and unchanged,'inputs_unchanged':unchanged,
                   'correctness':error.model_dump(),
                   'input_strides':[list(x.stride()) for x in inputs],
                   'input_alignment_mod_256':[x.data_ptr()%256 for x in inputs],
                   'output_shape':list(output.shape),'output_stride':list(output.stride())}
            report.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='correctness'}),flush=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            del inputs,original,expected,output
            torch.cuda.empty_cache()
    if any(not row['passed'] for row in report):
        raise SystemExit('A value or ownership check failed')


if __name__ == '__main__':
    main()
