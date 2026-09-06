#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit changed attention values, same-shape layout changes, and output ownership."""
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
    parser.add_argument("--label", default="selected")
    parser.add_argument("--shapes", nargs="+", default=["1,131", "2,256", "1,997"])
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    candidate = load(directory / args.module, "candidate_av_32")
    reference_path = root / ".work/problems/32/reference.py"
    reference = load(reference_path, "reference_av_32")
    path = root / f".work/tuning/32-variation-{args.label}.json"
    report = {
        "scope": "changed-input, current-value, shape/layout, and output-ownership checks; no timing claims",
        "candidate_module": args.module,
        "source_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                          for source in directory.glob("*.py")},
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "records": [],
    }
    tolerance = ToleranceSpec(max_atol=1e-5, max_rtol=0.05)
    patterns = ["random", "zero_attention", "zero_value", "ones", "identity_attention",
                "cancellation", "large_attention", "large_value", "small", "distinct_heads",
                "unaligned_attention", "unaligned_value", "strided_attention", "strided_value",
                "transposed_attention", "dense_after_layouts"]
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for shape in args.shapes:
            batch, sequence = map(int, shape.split(","))
            torch.manual_seed(323107 + sequence)
            base = [torch.randn((batch, 40, sequence, width), device="cuda", dtype=torch.bfloat16)
                    for width in (sequence, 128)]
            retained = retained_values = None
            for pattern in patterns:
                inputs = [tensor.clone() for tensor in base]
                if pattern == "zero_attention":
                    inputs[0].zero_()
                elif pattern == "zero_value":
                    inputs[1].zero_()
                elif pattern == "ones":
                    for tensor in inputs:
                        tensor.fill_(1)
                elif pattern == "identity_attention":
                    inputs[0].zero_()
                    inputs[0].diagonal(dim1=-2, dim2=-1).fill_(1)
                elif pattern == "cancellation":
                    inputs[0].fill_(1)
                    inputs[0][..., 1::2] = -1
                    inputs[1].fill_(1)
                elif pattern == "large_attention":
                    inputs[0].mul_(32)
                elif pattern == "large_value":
                    inputs[1].mul_(32)
                elif pattern == "small":
                    for tensor in inputs:
                        tensor.mul_(2 ** -8)
                elif pattern == "distinct_heads":
                    factors = torch.arange(1, batch * 40 + 1, device="cuda").view(batch, 40, 1, 1)
                    inputs[0].copy_(factors.expand_as(inputs[0]))
                    inputs[1].copy_((1 - factors).expand_as(inputs[1]))
                elif pattern.startswith("unaligned"):
                    index = 0 if pattern.endswith("attention") else 1
                    tensor = inputs[index]
                    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
                    value = storage[1:].view(tensor.shape)
                    value.copy_(tensor)
                    inputs[index] = value
                elif pattern.startswith("strided"):
                    index = 0 if pattern.endswith("attention") else 1
                    tensor = inputs[index]
                    dims = list(tensor.shape)
                    dims[-1] *= 2
                    value = torch.empty(dims, dtype=tensor.dtype, device=tensor.device)[..., ::2]
                    value.copy_(tensor)
                    inputs[index] = value
                elif pattern == "transposed_attention":
                    inputs[0] = inputs[0].transpose(-1, -2)
                originals = [tensor.clone() for tensor in inputs]
                expected = reference.run(*inputs)
                actual = candidate.run(*inputs)
                assert type(actual) is torch.Tensor
                assert actual.shape == expected.shape and actual.dtype == expected.dtype
                assert actual.device == expected.device and actual.is_contiguous()
                assert all(actual.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
                           for tensor in inputs)
                error, failed = compute_error_stats(actual, expected, tolerance)
                bounds = tolerance.max_atol + tolerance.max_rtol * expected.float().abs()
                unmatched = int(((actual.float() - expected.float()).abs() > bounds).sum().item())
                assert all(torch.equal(tensor, original) for tensor, original in zip(inputs, originals))
                if retained is not None:
                    assert torch.equal(retained, retained_values)
                    assert actual.untyped_storage().data_ptr() != retained.untyped_storage().data_ptr()
                retained, retained_values = actual, actual.clone()
                record = {"batch": batch, "sequence": sequence, "pattern": pattern,
                          "passed": not failed, "bitwise_equal": torch.equal(actual, expected),
                          "unmatched": unmatched, "elements": actual.numel(), "error": error.model_dump(),
                          "input_strides": [list(tensor.stride()) for tensor in inputs],
                          "input_alignment_mod16": [tensor.data_ptr() % 16 for tensor in inputs],
                          "output_stride": list(actual.stride()), "ownership_passed": True}
                report["records"].append(record)
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(record), flush=True)
    if not all(row["passed"] for row in report["records"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
