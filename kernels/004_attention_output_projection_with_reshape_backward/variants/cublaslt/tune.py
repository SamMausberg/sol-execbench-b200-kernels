#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded cuBLASLt search for both BF16 gradient projections."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import threading
import time

import torch
from torch.utils.cpp_extension import load
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data.workload import ToleranceSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--rep', type=int, default=10)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[3]
    sys.path.insert(0, str(root / 'tools'))
    from qualify_gpu import snapshot
    source = (directory / 'binding.cpp').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    target = Path(os.environ['TORCH_EXTENSIONS_DIR']) / 'sol_004_sources' / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / 'binding.cpp').write_bytes(source)
    cuda_package = Path(sysconfig.get_paths()['purelib']) / 'nvidia/cu13'
    native = load(name='sol_004_' + digest[:16], sources=[str(target / 'binding.cpp')],
                  extra_cflags=['-O3', '-std=c++17'], extra_include_paths=[str(cuda_package / 'include')],
                  extra_ldflags=[f'-L{cuda_package / "lib"}', '-l:libcublasLt.so.13', '-l:libcudart.so.13'], with_cuda=True)
    result = {'source_sha256': {'binding.cpp': digest}, 'scope': 'bounded component tuning; no complete-problem ranking claim',
              'records': [], 'process_samples': [], 'selected': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    family = {}

    def monitor():
        while not stop.is_set():
            result['process_samples'].append(snapshot(os.getpid(), family))
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    tolerance = ToleranceSpec(max_atol=0.00032, max_rtol=0.05)
    try:
        torch.manual_seed(400481)
        for rows in (256, 512, 1024, 2048, 4096, 8192):
            gradient = torch.randn((rows, 2048), device='cuda', dtype=torch.bfloat16)
            saved = torch.randn_like(gradient)
            weight = torch.randn((2048, 2048), device='cuda', dtype=torch.bfloat16)
            scratch = torch.empty(32 * 1024 * 1024, device='cuda', dtype=torch.uint8)
            algorithms = native.algorithms(gradient)
            selected = {}
            for kind, right, transposed in (('weight', saved, True), ('input', weight, False)):
                reference = (gradient.float().T @ right.float() if transposed else gradient.float() @ right.float()).to(torch.bfloat16)
                output = torch.empty_like(reference)
                records = []
                for candidate in algorithms[kind][:args.limit]:
                    index = candidate['index']

                    def invoke(a, b, c):
                        native.component(a, b, c, scratch, transposed, index)

                    record = {'rows': rows, 'kind': kind, 'algorithm': candidate, 'started_at': time.time()}
                    try:
                        output.fill_(float('nan'))
                        invoke(gradient, right, output)
                        error, bad = compute_error_stats(output, reference, tolerance)
                        record['passed'] = not bad
                        record['error'] = error.model_dump()
                        if not bad:
                            record['latency_ms'] = time_runnable(invoke, [gradient, right], [output], 'cuda', warmup=3, rep=args.rep)
                    except Exception as exc:
                        record['passed'] = False
                        record['exception'] = f'{type(exc).__name__}: {exc}'
                    record['finished_at'] = time.time()
                    records.append(record); result['records'].append(record)
                    args.output.write_text(json.dumps(result, indent=2) + '\n')
                valid = [record for record in records if record.get('passed') and 'latency_ms' in record]
                if valid:
                    best = min(valid, key=lambda record: record['latency_ms'])
                    selected[kind] = best
                    print(json.dumps({'rows': rows, 'kind': kind, 'best_index': best['algorithm']['index'],
                                      'latency_us': best['latency_ms'] * 1000}), flush=True)
            result['selected'][str(rows)] = selected
            del gradient, saved, weight, scratch
    finally:
        stop.set(); thread.join(timeout=10)
        result['foreign_samples'] = sum(any(not context['campaign_descendant'] for context in sample['contexts'])
                                        for sample in result['process_samples'])
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
