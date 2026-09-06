#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check changed MLA values, scalar epsilon, concrete views, and ownership."""

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
    parser.add_argument("--module", default="kernel.py")
    parser.add_argument("--label", default="library")
    parser.add_argument("--patterns", nargs="+")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    candidate = load(directory / args.module, "candidate_mla_43")
    reference_path = root / ".work/problems/43/reference.py"
    reference = load(reference_path, "reference_mla_43")
    output = root / f".work/tuning/43-variation-{args.label}.json"
    report = {
        "scope": "changed input/scalar and output-ownership audit; separate timing evidence required",
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in directory.glob("*.py")},
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "records": [],
    }
    tolerance = ToleranceSpec(max_atol=0.0044, max_rtol=0.05)
    patterns = ("random", "zero_input", "ones", "basis", "negative", "small", "large",
                "changed_qa", "changed_norm", "zero_qb", "changed_kv", "epsilon_and_strides", "unaligned_all")
    if args.patterns:
        if not set(args.patterns).issubset(patterns):
            parser.error("unknown input pattern")
        patterns = args.patterns
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for batch, sequence in ((1, 131), (2, 149), (2, 512)):
            torch.manual_seed(433000 + sequence)
            base = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for shape in (
                (batch, sequence, 7168), (1536, 7168), (1536,), (24576, 1536), (576, 7168),
            )]
            retained = None
            retained_values = None
            for pattern in patterns:
                inputs = [tensor.clone() for tensor in base]
                epsilon = 1e-6
                if pattern == "zero_input":
                    inputs[0].zero_()
                elif pattern == "ones":
                    inputs[0].fill_(1)
                elif pattern == "basis":
                    inputs[0].zero_()
                    inputs[0][..., 17] = 1
                elif pattern == "negative":
                    inputs[0].copy_(-inputs[0].abs())
                elif pattern == "small":
                    inputs[0].mul_(2 ** -8)
                elif pattern == "large":
                    inputs[0].mul_(32)
                elif pattern == "changed_qa":
                    inputs[1].neg_()
                elif pattern == "changed_norm":
                    inputs[2].mul_(32)
                elif pattern == "zero_qb":
                    inputs[3].zero_()
                elif pattern == "changed_kv":
                    inputs[4].mul_(-2)
                elif pattern == "epsilon_and_strides":
                    epsilon = 0.1
                    inputs[1].mul_(2 ** -8)
                    norm = torch.empty(3072, dtype=torch.bfloat16, device="cuda")[::2]
                    norm.copy_(inputs[2])
                    inputs[2] = norm
                    hidden = torch.empty((batch, sequence, 14336), dtype=torch.bfloat16, device="cuda")[..., ::2]
                    hidden.copy_(inputs[0])
                    inputs[0] = hidden
                elif pattern == "unaligned_all":
                    for index, tensor in enumerate(inputs):
                        storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
                        value = storage[1:].view(tensor.shape)
                        value.copy_(tensor)
                        inputs[index] = value
                originals = [tensor.clone() for tensor in inputs]
                expected = reference.run(*inputs, epsilon)
                actual = candidate.run(*inputs, epsilon)
                assert type(actual) is tuple and len(actual) == 4
                rows = []
                for tensor, ref in zip(actual, expected):
                    assert type(tensor) is torch.Tensor
                    assert tensor.shape == ref.shape and tensor.dtype == ref.dtype and tensor.device == ref.device
                    assert all(tensor.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
                               for value in inputs)
                    error, failed = compute_error_stats(tensor, ref, tolerance)
                    unmatched = int(((tensor - ref).abs() > tolerance.max_atol + tolerance.max_rtol * ref.abs()).sum().item())
                    rows.append({"passed": not failed, "bitwise_equal": torch.equal(tensor, ref),
                                 "unmatched": unmatched, "elements": tensor.numel(),
                                 "strides": tensor.stride(), "error": error.model_dump()})
                assert all(torch.equal(tensor, original) for tensor, original in zip(inputs, originals))
                if retained is not None:
                    assert all(torch.equal(tensor, value) for tensor, value in zip(retained, retained_values))
                    assert all(tensor.untyped_storage().data_ptr() != old.untyped_storage().data_ptr()
                               for tensor in actual for old in retained)
                snapshots = tuple(tensor.clone() for tensor in actual)
                # Query components may share a storage object, but their logical
                # elements must be independent when a caller mutates one result.
                actual[0].zero_()
                assert all(torch.equal(tensor, value) for tensor, value in zip(actual[1:], snapshots[1:]))
                actual[0].copy_(snapshots[0])
                retained, retained_values = actual, snapshots
                record = {"batch": batch, "sequence": sequence, "pattern": pattern,
                          "epsilon": epsilon, "passed": all(row["passed"] for row in rows), "outputs": rows}
                report["records"].append(record)
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(record), flush=True)
    if not all(row["passed"] for row in report["records"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
