#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise BF16 rounding, changed values, cancellation, and noncontiguous inputs."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import torch
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.data import Definition, Workload

from cute_kernel import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    problem = root / ".work/problems/30"
    definition = Definition(**json.loads((problem / "definition.json").read_text()))
    tolerance = Workload(**json.loads((problem / "workload.jsonl").read_text().splitlines()[0])).tolerance
    namespace = {}
    exec(definition.reference, namespace)
    report = {"tolerance": tolerance.model_dump(), "cases": [], "source_sha256": {
        n: hashlib.sha256((directory / n).read_bytes()).hexdigest()
        for n in ("cute_kernel.py", "cute_residual.py", "kernel.py")}}
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for seq in (128, 131, 541, 2048):
            set_seed(30371 + seq)
            a = torch.randn((1, seq, 2560), device="cuda", dtype=torch.bfloat16)
            w = torch.randn((2560, 2560), device="cuda", dtype=torch.bfloat16)
            r = torch.randn_like(a)
            for case in ("fresh", "reused_storage", "cancellation", "scaled_values",
                         "zero_weight", "zero_input", "strided", "strided_output"):
                a.normal_()
                w.normal_()
                r.normal_()
                if case == "cancellation":
                    r.copy_(-torch.matmul(a, w.t()))
                elif case == "scaled_values":
                    a.mul_(16)
                    w.mul_(-0.5)
                elif case == "zero_weight":
                    w.zero_()
                elif case == "zero_input":
                    a.zero_()
                x, weight, residual = a, w, r
                if case == "strided":
                    x = torch.empty((1, seq, 5120), device=a.device, dtype=a.dtype)[..., ::2]
                    residual = torch.empty_like(x, memory_format=torch.contiguous_format)
                    weight = torch.empty((2560, 5120), device=a.device, dtype=a.dtype)[:, ::2]
                    x.copy_(a)
                    residual.copy_(r)
                    weight.copy_(w)
                originals = [tensor.clone() for tensor in (x, residual, weight)]
                reference = namespace["run"](x, residual, weight)
                if case == "strided_output":
                    output = torch.empty((1, seq, 5120), device=a.device, dtype=a.dtype)[..., ::2]
                else:
                    output = torch.empty_like(reference)
                output.fill_(float("nan"))
                run(x, residual, weight, output)
                error, exceeds = compute_error_stats(output, reference, tolerance)
                delta = (output.float() - reference.float()).abs()
                match = delta <= tolerance.max_atol + tolerance.max_rtol * reference.float().abs()
                unchanged = all(torch.equal(tensor, old) for tensor, old in zip((x, residual, weight), originals))
                record = {"seq": seq, "case": case, "passed": not exceeds and unchanged,
                          "inputs_unchanged": unchanged, "strict_allclose": bool(match.all()),
                          "matched_ratio": float(match.float().mean()),
                          "fully_written": bool(torch.isfinite(output).all()),
                          "correctness": error.model_dump()}
                report["cases"].append(record)
                print(json.dumps(record), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                if not record["passed"]:
                    raise RuntimeError(f"Failed {seq}/{case}")


if __name__ == "__main__":
    main()
