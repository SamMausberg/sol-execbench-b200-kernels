#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check changed values, layouts, and output ownership against the pinned reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.data import Definition, Workload


def with_layout(tensor, mode):
    if mode == 'strided':
        storage = torch.empty((*tensor.shape[:-1], 2 * tensor.shape[-1]), device=tensor.device, dtype=tensor.dtype)
        result = storage[..., ::2]
    elif mode == 'padded':
        storage = torch.empty((*tensor.shape[:-1], tensor.shape[-1] + 16), device=tensor.device, dtype=tensor.dtype)
        result = storage[..., :tensor.shape[-1]]
    elif mode == 'unaligned':
        storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
        result = storage[1:].view(tensor.shape)
    else:
        return tensor
    result.copy_(tensor)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    source = (directory / 'binding.cpp').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    snapshot = Path(os.environ['TORCH_EXTENSIONS_DIR']) / 'sol_004_sources' / digest
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / 'binding.cpp').write_bytes(source)
    native = load(name='sol_004_' + digest[:16], sources=[str(snapshot / 'binding.cpp')],
                  extra_cflags=['-O3', '-std=c++17'], with_cuda=True)
    definition = Definition(**json.loads((root / '.work/problems/4/definition.json').read_text()))
    workloads = [Workload(**json.loads(line)) for line in (root / '.work/problems/4/workload.jsonl').read_text().splitlines()]
    scope = {}
    exec(definition.reference, scope)
    # Smallest, intermediate, and largest distinct matrix row counts.
    ordered = sorted(range(len(workloads)), key=lambda i: workloads[i].axes['batch_size'] * workloads[i].axes['seq_len'])
    indices = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    report = {'source_sha256': {'binding.cpp': digest},
              'verification_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(definition.reference.encode()).hexdigest(),
              'scope': 'numerical, input-layout, and fresh-output audit; no timing qualification', 'records': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in indices:
        previous = None
        for mode in ('random', 'zero_weight', 'zero_gradient', 'zero_saved', 'basis', 'signed',
                     'scaled_32', 'strided', 'padded', 'unaligned'):
            set_seed(440219 + index)
            inputs = list(gen_inputs(definition, workloads[index], 'cuda'))
            if mode == 'zero_weight': inputs[2].zero_()
            elif mode == 'zero_gradient': inputs[0].zero_()
            elif mode == 'zero_saved': inputs[1].zero_()
            elif mode == 'basis':
                inputs[0].zero_(); inputs[0][..., ::127] = 1
                inputs[2].copy_(torch.eye(2048, device='cuda', dtype=torch.bfloat16))
            elif mode == 'signed':
                inputs = [value.sign() for value in inputs]
            elif mode == 'scaled_32':
                inputs = [value * 32 for value in inputs]
            inputs = [with_layout(value, mode) for value in inputs]
            originals = [value.clone() for value in inputs]
            expected = scope['run'](*inputs)
            actual = native.run(*inputs)
            outputs = []
            for value, reference in zip(actual, expected):
                error, bad = compute_error_stats(value, reference, workloads[index].tolerance)
                outputs.append({'passed': not bad, 'bitwise_equal': torch.equal(value, reference),
                                'shape': list(value.shape), 'stride': list(value.stride()),
                                'error': error.model_dump()})
            unchanged = all(torch.equal(value, original) for value, original in zip(inputs, originals))
            prior_preserved = previous is None or all(torch.equal(value, saved) for value, saved in zip(*previous))
            row = {'workload': index, 'axes': workloads[index].axes, 'mode': mode,
                   'inputs_unchanged': unchanged, 'previous_outputs_preserved': prior_preserved,
                   'outputs': outputs,
                   'passed': unchanged and prior_preserved and all(value['passed'] for value in outputs)}
            report['records'].append(row)
            previous = (actual, [value.clone() for value in actual])
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'workload': index, 'mode': mode, 'passed': row['passed']}), flush=True)
            del inputs, originals, expected
    report['passed'] = all(row['passed'] for row in report['records'])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('At least one variation failed')


if __name__ == '__main__':
    main()
