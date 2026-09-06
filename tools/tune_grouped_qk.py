#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded grouped-QK geometry sweep with official input and correctness helpers."""

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import statistics

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti
from sol_execbench.core.data import Definition, Workload

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--persistent", action="store_true")
parser.add_argument("--tma", action="store_true")
parser.add_argument("--library", action="store_true")
parser.add_argument("--stages", type=int, choices=(1, 2, 3, 4))
parser.add_argument("--indices", default="5,4,0,3")
parser.add_argument("--resume", type=Path)
args = parser.parse_args()
if sum((args.tma, args.persistent, args.library)) > 1:
    parser.error("Select one of --tma, --persistent, and --library")
SOURCE = ROOT / "kernels/049_attention_qk_matmul_with_gqa_repeat_and_scaling/kernel.py"
spec = importlib.util.spec_from_file_location("grouped_qk", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if args.persistent:
    pspec = importlib.util.spec_from_file_location("persistent_qk", SOURCE.with_name("persistent.py"))
    persistent = importlib.util.module_from_spec(pspec)
    pspec.loader.exec_module(persistent)
if args.tma:
    tspec = importlib.util.spec_from_file_location("tma_qk", SOURCE.with_name("tma.py"))
    tma = importlib.util.module_from_spec(tspec)
    tspec.loader.exec_module(tma)
if args.library:
    lspec = importlib.util.spec_from_file_location("library_qk", SOURCE.with_name("library.py"))
    library = importlib.util.module_from_spec(lspec)
    lspec.loader.exec_module(library)
problem = ROOT / ".work/problems/49"
definition = Definition(**json.loads((problem / "definition.json").read_text()))
workloads = [Workload(**json.loads(row)) for row in (problem / "workload.jsonl").read_text().splitlines()]
namespace = {}
exec(definition.reference, namespace)
reference = namespace["run"]
configs = [
    (16, 32, 64, 4), (32, 32, 64, 4), (32, 64, 64, 4),
    (64, 64, 64, 4), (64, 128, 64, 4), (128, 64, 64, 4),
    (128, 128, 64, 4), (128, 128, 64, 8), (128, 256, 64, 8),
    (256, 128, 64, 8), (64, 64, 128, 4), (64, 128, 128, 4),
    (128, 128, 128, 8), (128, 256, 128, 8),
    (64, 64, 256, 4), (64, 128, 256, 4),
    (128, 128, 256, 8), (128, 256, 256, 8),
]
if args.library:
    configs = [(0, 0, 0, 0, 0, False)]
elif args.tma:
    configs = [(bm, bn, bk, warps, programs, True)
               for bm, bn, bk, warps in [(64, 128, 64, 4), (128, 128, 64, 4),
                                          (128, 256, 64, 8), (128, 128, 128, 8),
                                          (128, 256, 128, 8), (128, 128, 256, 8)]
               for programs in (0, 296)]
elif args.persistent:
    configs = [(bm, bn, bk, warps, programs, stream)
               for bm, bn, bk, warps in [(128, 128, 64, 4), (128, 256, 64, 8),
                                          (128, 128, 128, 8), (256, 128, 64, 8)]
               for programs in (148, 296) for stream in (False, True)]
else:
    configs = [(*c, 0, False) for c in configs]
results = []
completed = set()
def config_key(row):
    return (row["workload"], row["bm"], row["bn"], row["bk"], row["num_warps"],
            row.get("programs", 0), row.get("stream", False), row.get("num_stages", 3))
if args.resume:
    for line in args.resume.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("passed") and "latency_ms" in row:
            results.append(row)
            completed.add(config_key(row))
lock = Path(os.environ.get("SOL_GPU_LOCK", "/workspace/sol-execbench-b200-kernels/.work/gpu.lock"))
with lock.open("a") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    for index in map(int, args.indices.split(",")):
        set_seed(4900 + index)
        workload = workloads[index]
        q, k, scale = gen_inputs(definition, workload, "cuda")
        ref = reference(q, k, scale)
        out = torch.empty_like(ref)
        batch, heads, seq, dim = q.shape
        kv_heads = k.shape[1]
        group = heads // kv_heads
        for bm, bn, bk, warps, programs, stream in configs:
            stages = min(3, 232000 // (2 * (bm + bn) * bk)) if not args.library else 0
            if args.stages is not None and not args.library:
                stages = args.stages
            row = {"workload": index, "axes": workload.axes, "bm": bm, "bn": bn,
                   "bk": bk, "num_warps": warps, "programs": programs, "stream": stream,
                   "num_stages": stages}
            if config_key(row) in completed:
                continue
            if args.tma:
                assert q.is_contiguous() and k.is_contiguous()
                qdesc = TensorDescriptor(q, [batch * heads * seq, dim], [dim, 1], [bm, bk])
                kdesc = TensorDescriptor(k, [batch * kv_heads * seq, dim], [dim, 1], [bn, bk])
            def launch():
                if args.library:
                    library.run(q, k, scale, out)
                elif args.tma:
                    tiles = triton.cdiv(group * seq, bm) * triton.cdiv(seq, bn) * batch * kv_heads
                    grid = min(programs, tiles) if programs else tiles
                    tma.grouped_qk_tma[(grid,)](
                        qdesc, kdesc, out, scale, seq, dim, group, kv_heads, batch,
                        bm, bn, bk, grid, stream, num_warps=warps, num_stages=stages,
                    )
                elif programs:
                    tiles = triton.cdiv(group * seq, bm) * triton.cdiv(seq, bn) * batch * kv_heads
                    persistent.grouped_qk_persistent[(min(programs, tiles),)](
                        q, k, out, scale, seq, dim, group, kv_heads, batch,
                        *q.stride(), *k.stride(), bm, bn, bk, stream,
                        num_warps=warps, num_stages=stages,
                    )
                else:
                    module._grouped_qk[(triton.cdiv(group * seq, bm), triton.cdiv(seq, bn), batch * kv_heads)](
                        q, k, out, scale, seq, dim, group, kv_heads,
                        *q.stride(), *k.stride(), bm, bn, bk,
                        num_warps=warps, num_stages=stages,
                    )
            try:
                out.fill_(float("nan"))
                launch()
                error, exceeds = compute_error_stats(out, ref, workload.tolerance)
                row["correctness"] = error.model_dump()
                row["passed"] = not exceeds
                if not exceeds:
                    times = bench_gpu_time_with_cupti(launch, warmup=3, rep=15)
                    row["latency_ms"] = statistics.median(times)
            except Exception as error:
                row["passed"] = False
                row["error"] = f"{type(error).__name__}: {error}"
            print(json.dumps(row), flush=True)
            results.append(row)
        del q, k, out, ref
        torch.cuda.empty_cache()
output = ROOT / (".work/tuning/49-library.json" if args.library else ".work/tuning/49-tma.json" if args.tma else
                 ".work/tuning/49-persistent.json" if args.persistent else ".work/tuning/49-grouped.json")
if args.stages is not None:
    output = output.with_stem(f"{output.stem}-stages{args.stages}")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + "\n")
