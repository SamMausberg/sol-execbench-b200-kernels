#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded test of register pressure in the FP32 projection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import torch
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data import Definition, Workload
from kernel import launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    sys.path.insert(0, str(root / 'tools'))
    from qualify_gpu import snapshot
    definition = Definition(**json.loads((root / '.work/problems/35/definition.json').read_text()))
    workloads = [Workload(**json.loads(line)) for line in (root / '.work/problems/35/workload.jsonl').read_text().splitlines()]
    result = {'source_sha256': {'kernel.py': hashlib.sha256((directory / 'kernel.py').read_bytes()).hexdigest()},
              'scope': 'four configurations at two shapes; register-spill investigation', 'records': [], 'process_samples': []}
    family = {}
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            result['process_samples'].append(snapshot(os.getpid(), family))
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    configs = [dict(bm=64, bn=64, bk=64, stages=3, warps=4),
               dict(bm=64, bn=64, bk=32, stages=2, warps=8),
               dict(bm=32, bn=64, bk=32, stages=2, warps=4),
               dict(bm=64, bn=64, bk=32, stages=1, warps=4)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index in (6, 7):
            set_seed(350511)
            emb, weight, bias = gen_inputs(definition, workloads[index], 'cuda')
            for scale in (1, 32):
                left, right = emb * scale, weight * scale
                expected = left @ right.T + bias
                output = torch.empty_like(expected)
                for config in configs:
                    row = {'workload': index, 'axes': workloads[index].axes, 'scale': scale, 'config': config,
                           'started_at': time.time()}

                    def invoke(e, w, b, o):
                        launch(e, w, b, o, **config)

                    try:
                        invoke(left, right, bias, output)
                        error, bad = compute_error_stats(output, expected, workloads[index].tolerance)
                        row['passed'] = not bad; row['error'] = error.model_dump()
                        if not bad and scale == 1:
                            row['latency_ms'] = time_runnable(invoke, [left, right, bias], [output], 'cuda', warmup=3, rep=10)
                    except Exception as exc:
                        row['passed'] = False; row['exception'] = f'{type(exc).__name__}: {exc}'
                    row['finished_at'] = time.time(); result['records'].append(row)
                    args.output.write_text(json.dumps(result, indent=2) + '\n')
                    print(json.dumps({key: value for key, value in row.items() if key != 'error'}), flush=True)
    finally:
        stop.set(); thread.join(timeout=10)
        result['foreign_samples'] = sum(any(not c['campaign_descendant'] for c in s['contexts']) for s in result['process_samples'])
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
