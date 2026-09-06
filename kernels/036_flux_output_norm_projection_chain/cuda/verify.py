#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Changed-input checks for the complete native Flux chain."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.data import Definition, Workload
from loader import load_native


def strided(tensor):
    storage = torch.empty((*tensor.shape[:-1], tensor.shape[-1] * 2), device=tensor.device, dtype=tensor.dtype)
    view = storage[..., ::2]
    view.copy_(tensor)
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    definition = Definition(**json.loads((root / '.work/problems/36/definition.json').read_text()))
    workloads = [Workload(**json.loads(line)) for line in (root / '.work/problems/36/workload.jsonl').read_text().splitlines()]
    scope = {}; exec(definition.reference, scope)
    native = load_native()
    report = {'source_sha256': native.source_sha256, 'verification_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'records': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in (3, 1):
        for mode in ('random', 'weights_32', 'temb_4', 'zero_weights', 'zero_hidden', 'strided', 'changed_eps', 'near_constant'):
            set_seed(363921 + index)
            inputs = list(gen_inputs(definition, workloads[index], 'cuda'))
            if mode == 'weights_32':
                inputs[2].mul_(32); inputs[4].mul_(32)
            elif mode == 'temb_4': inputs[1].mul_(4)
            elif mode == 'zero_weights': inputs[2].zero_(); inputs[4].zero_()
            elif mode == 'zero_hidden': inputs[0].zero_()
            elif mode == 'strided': inputs = [strided(t) if isinstance(t, torch.Tensor) else t for t in inputs]
            elif mode == 'changed_eps': inputs[-1] = 1e-3
            elif mode == 'near_constant': inputs[0].mul_(1e-4).add_(1.0)
            originals = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
            reference = scope['run'](*inputs)
            output = torch.empty_like(reference)
            if mode == 'strided': output = strided(output)
            output.fill_(float('nan'))
            native.run(*inputs, output)
            error, bad = compute_error_stats(output, reference, workloads[index].tolerance)
            unchanged = all(torch.equal(before, after) for before, after in zip(originals, inputs) if isinstance(before, torch.Tensor))
            row = {'workload': index, 'axes': workloads[index].axes, 'mode': mode, 'passed': not bad and unchanged,
                   'inputs_unchanged': unchanged, 'errors': error.model_dump()}
            report['records'].append(row)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            del inputs, originals, reference, output
            torch.cuda.empty_cache()
    report['passed'] = all(r['passed'] for r in report['records'])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']: raise SystemExit('At least one input variation failed')


if __name__ == '__main__': main()
