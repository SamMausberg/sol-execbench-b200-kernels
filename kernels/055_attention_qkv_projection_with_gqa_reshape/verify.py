#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit new values, concrete output views, and input preservation."""

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


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="cute_kernel.py")
    parser.add_argument("--label", default="cute")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    candidate = load(directory / args.module, "candidate_55")
    reference = load(root / ".work/problems/55/reference.py", "reference_55")
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    destination = root / f".work/tuning/55-variation-{args.label}.json"
    result = {
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in (args.module, "cute_qkv.py", "verify.py")},
        "reference_sha256": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
        "scope": "input-variation audit; separate official workloads and timing required",
        "records": [],
    }
    tolerance = ToleranceSpec(max_atol=1e-5, max_rtol=0.05)
    names = ("random", "zero_input", "ones", "negative", "alternating",
             "small", "large", "basis", "zero_q", "changed_kv")
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for batch, seq in ((1, 131), (2, 211), (2, 1024)):
            torch.manual_seed(550000 + batch * 1000 + seq)
            base = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for shape in (
                (batch, seq, 2048), (2048, 2048), (512, 2048), (512, 2048),
            )]
            retained = None
            retained_values = None
            for name in names:
                inputs = [value.clone() for value in base]
                if name == "zero_input":
                    inputs[0].zero_()
                elif name == "ones":
                    inputs[0].fill_(1)
                elif name == "negative":
                    inputs[0].copy_(-inputs[0].abs())
                elif name == "alternating":
                    inputs[0][..., ::2] = 1
                    inputs[0][..., 1::2] = -1
                elif name == "small":
                    inputs[0].mul_(2 ** -8)
                elif name == "large":
                    inputs[0].mul_(32)
                elif name == "basis":
                    inputs[0].zero_()
                    inputs[0][..., 1023] = 1
                elif name == "zero_q":
                    inputs[1].zero_()
                elif name == "changed_kv":
                    inputs[2].neg_()
                    inputs[3].mul_(4)
                originals = [value.clone() for value in inputs]
                expected = reference.run(*inputs)
                actual = candidate.run(*inputs)
                if not isinstance(actual, tuple) or len(actual) != 3:
                    raise AssertionError("expected three concrete output tensors")
                rows = []
                for out, ref in zip(actual, expected):
                    if not isinstance(out, torch.Tensor):
                        raise AssertionError("outputs must be ordinary concrete tensors")
                    assert out.shape == ref.shape and out.dtype == ref.dtype
                    assert out.device == ref.device and out.stride() == ref.stride()
                    error, failed = compute_error_stats(out, ref, tolerance)
                    unmatched = int(((out - ref).abs() >
                                     tolerance.max_atol + tolerance.max_rtol * ref.abs()).sum().item())
                    rows.append({"passed": not failed, "error": error.model_dump(),
                                 "unmatched": unmatched, "elements": out.numel(),
                                 "bitwise_equal": torch.equal(out, ref)})
                for value, original in zip(inputs, originals):
                    torch.testing.assert_close(value, original, rtol=0, atol=0)
                if retained is not None:
                    for out, snapshot in zip(retained, retained_values):
                        torch.testing.assert_close(out, snapshot, atol=0, rtol=0)
                retained = actual
                retained_values = tuple(value.clone() for value in actual)
                record = {"batch": batch, "seq": seq, "pattern": name,
                          "passed": all(row["passed"] for row in rows), "outputs": rows}
                result["records"].append(record)
                destination.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(record), flush=True)
    if not all(row["passed"] for row in result["records"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
