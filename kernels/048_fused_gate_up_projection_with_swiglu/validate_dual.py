#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check changed values and a strided layout against the pinned reference."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
if os.environ.get("SOL_48_VALIDATION_LOCKED") != "1":
    lock = os.environ.get("SOL_GPU_LOCK", str(ROOT / ".work/gpu.lock"))
    print(f"Waiting for GPU lock before CUDA imports: {lock}", flush=True)
    os.execvpe("flock", ["flock", "--exclusive", lock, sys.executable, __file__],
               {**os.environ, "SOL_48_VALIDATION_LOCKED": "1"})

import torch
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from dual_kernel import run


def main():
    problem = ROOT / ".work/problems/48"
    definition = Definition.model_validate_json((problem / "definition.json").read_text())
    workloads = [Workload.model_validate_json(line)
                 for line in (problem / "workload.jsonl").read_text().splitlines()]
    reference = runpy.run_path(str(problem / "reference.py"))["run"]
    target = ROOT / ".work/tuning/48-dual-validation.json"
    report = {
        "scope": "changed inputs and noncontiguous fallback, original tolerance; no timing claims",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_sha256": {name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
                          for name in ("cute_dual.py", "dual_kernel.py", "validate_dual.py")},
        "records": [],
    }
    cases = [(index, scale, "dense") for index in (6, 4, 2) for scale in (1.0, 0.03125, 32.0)]
    cases += [(13, 1.0, "strided"), (13, 1.0, "zero_x")]
    with torch.no_grad():
        for index, scale, layout in cases:
            torch.manual_seed(481700 + index)
            x, gate, up = gen_inputs(definition, workloads[index], "cuda")
            gate.mul_(scale)
            up.mul_(scale)
            if layout == "zero_x":
                x.zero_()
            if layout == "strided":
                def strided(value):
                    storage = torch.empty((*value.shape[:-1], value.shape[-1] * 2),
                                          dtype=value.dtype, device=value.device)
                    view = storage[..., ::2]
                    view.copy_(value)
                    return view
                x, gate, up = map(strided, (x, gate, up))
            expected = reference(x, gate, up)
            out = torch.full_like(expected, float("nan"))
            run(x, gate, up, out)
            errors, failed = compute_error_stats(out, expected, workloads[index].tolerance)
            tolerance = workloads[index].tolerance
            matched = ((out.float() - expected.float()).abs()
                       <= tolerance.max_atol + tolerance.max_rtol * expected.float().abs()).float().mean().item()
            record = {"workload": index, "axes": workloads[index].axes,
                      "weight_scale": scale, "layout": layout, "passed": not failed,
                      "matched_ratio": matched, "errors": errors.model_dump(),
                      "tolerance": tolerance.model_dump()}
            report["records"].append(record)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if failed:
                raise RuntimeError("Changed-input correctness failed")
    report["passed"] = all(row["passed"] for row in report["records"])
    report["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
