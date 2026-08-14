#!/usr/bin/env python3
"""Simulate the evaluator's build_ext.py for a packaged solution.json.

Usage: eval_build_sim.py dist/solution_X.json [--arch 120a]
With --arch, the gencode flag is rewritten so the result can run locally.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("solution")
ap.add_argument("--arch", default=None)
args = ap.parse_args()

sol = json.load(open(args.solution))
stem = Path(args.solution).stem + (f"_{args.arch}" if args.arch else "")
stage = Path("local/stage") / stem
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(parents=True)
for s in sol["sources"]:
    (stage / s["path"]).write_text(s["content"])

co = sol["spec"].get("compile_options") or {}
cuda_cflags = list(co.get("cuda_cflags", ["-O3", "--use_fast_math"]))
if args.arch:
    cuda_cflags = [f.replace("compute_100a", f"compute_{args.arch}").replace(
        "sm_100a", f"sm_{args.arch}") for f in cuda_cflags]
ld_flags = list(co.get("ld_flags", ["-lcuda"])) + ["-L/usr/lib/wsl/lib"]

import torch.utils.cpp_extension as ext
cutlass = os.environ.get("CUTLASS_DIR", "/home/sam/cutlass-v441")
t0 = time.time()
ext.load(
    name="benchmark_kernel",
    sources=[str(p) for p in stage.iterdir() if p.suffix in (".cu", ".cpp")],
    extra_cuda_cflags=cuda_cflags,
    extra_cflags=co.get("cflags", []),
    extra_ldflags=ld_flags,
    extra_include_paths=[str(stage), f"{cutlass}/include", f"{cutlass}/tools/util/include"],
    build_directory=str(stage),
    verbose=False,
)
print(f"build OK in {time.time()-t0:.1f}s")
sos = [f for f in stage.glob("benchmark_kernel*.so")]
sos[0].rename(stage / "benchmark_kernel.so") if sos and sos[0].name != "benchmark_kernel.so" else None
print(stage / "benchmark_kernel.so")
