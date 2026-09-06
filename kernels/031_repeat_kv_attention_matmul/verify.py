#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check DPS behavior and varied BF16 data against the pinned reference."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.data.workload import ToleranceSpec

from kernel import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).parent
    root = directory.parents[1]
    spec = importlib.util.spec_from_file_location(
        "pinned_reference_031", root / ".work/problems/31/reference.py",
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    workload = json.loads(
        (root / ".work/problems/31/workload.jsonl").read_text().splitlines()[0],
    )
    tolerance = ToleranceSpec.model_validate(workload["tolerance"])
    lock = Path(os.environ.get(
        "SOL_GPU_LOCK", "/workspace/sol-execbench-b200-kernels/.work/gpu.lock",
    ))
    lock.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("kernel.py", "tma.py")
        },
        "reference_sha256": hashlib.sha256(
            Path(spec.origin).read_bytes(),
        ).hexdigest(),
        "tolerance": tolerance.model_dump(),
        "checks": [],
    }
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        print("GPU lock acquired", flush=True)
        patterns = [
            "seed0", "seed1", "seed2", "small", "large",
            "zero_query", "ones", "cancel", "distinct_heads", "equal_qk",
        ]
        for batch, seq in ((1, 128), (1, 131), (1, 449)):
            for index, pattern in enumerate(patterns):
                torch.manual_seed(315710 + index)
                shape = (batch, 32, seq, 128)
                q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                if pattern == "small":
                    q.mul_(0.01)
                    k.mul_(0.01)
                elif pattern == "large":
                    q.mul_(16)
                    k.mul_(16)
                elif pattern == "zero_query":
                    q.zero_()
                elif pattern == "ones":
                    q.fill_(1)
                    k.fill_(1)
                elif pattern == "cancel":
                    q.fill_(1)
                    q[..., 1::2] = -1
                    k.fill_(1)
                elif pattern == "distinct_heads":
                    factor = torch.arange(1, 33, device="cuda").view(1, 32, 1, 1)
                    q.copy_(factor.expand(shape))
                    k.copy_((33 - factor).expand(shape))
                elif pattern == "equal_qk":
                    k.copy_(q)
                original_q, original_k = q.clone(), k.clone()
                expected = reference.run(q, k)
                output = torch.full_like(expected, float("nan"))
                returned = run(q, k, output)
                if returned is not None:
                    raise AssertionError("DPS entry point must return None")
                errors, exceeds = compute_error_stats(output, expected, tolerance)
                absolute_error = (output.float() - expected.float()).abs()
                bounds = tolerance.max_atol + tolerance.max_rtol * expected.float().abs()
                mismatches = int((absolute_error > bounds).sum().item())
                torch.testing.assert_close(q, original_q, atol=0, rtol=0)
                torch.testing.assert_close(k, original_k, atol=0, rtol=0)
                check = {
                    "batch": batch, "seq": seq, "pattern": pattern,
                    "passed": not exceeds, "errors": errors.model_dump(),
                    "mismatched_elements": mismatches,
                    "total_elements": output.numel(),
                    "matched_ratio": 1 - mismatches / output.numel(),
                    "strict_all_elements_passed": mismatches == 0,
                }
                result["checks"].append(check)
                print(json.dumps(check), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                if exceeds:
                    raise AssertionError(f"Official tolerance failed: {check}")


if __name__ == "__main__":
    main()
