#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit layout changes at a fixed shape, output ownership, and exact values."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.data.workload import ToleranceSpec

from cute_kernel_v2 import run, supports_tma


def layout_copy(tensor, layout):
    if layout == "column_stride":
        shape = (*tensor.shape[:-1], tensor.shape[-1] * 2)
        result = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)[..., ::2]
    elif layout == "row_padding":
        shape = (*tensor.shape[:-1], tensor.shape[-1] + 8)
        result = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)[..., :tensor.shape[-1]]
    elif layout == "transposed":
        result = torch.empty_like(tensor.transpose(-1, -2)).contiguous().transpose(-1, -2)
    elif layout == "unaligned":
        result = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)[1:].view(tensor.shape)
    else:
        result = torch.empty_like(tensor)
    result.copy_(tensor)
    return result


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    ref_path = root / ".work/problems/55/reference.py"
    spec = importlib.util.spec_from_file_location("reference_55_layout", ref_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    report = {
        "scope": "layout-changing value and ownership audit; no timing measurement",
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in ("cute_kernel_v2.py", "cute_kernel.py", "cute_qkv.py", "verify_layouts.py")},
        "reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
        "records": [],
    }
    output = root / ".work/tuning/55-layout-v2.json"
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    tolerance = ToleranceSpec(max_atol=1e-5, max_rtol=0.05)
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for batch, sequence in ((1, 131), (2, 211)):
            torch.manual_seed(551000 + sequence)
            base = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for shape in (
                (batch, sequence, 2048), (2048, 2048), (512, 2048), (512, 2048),
            )]
            cases = [("dense_initial", "dense", None)]
            cases += [(f"{layout}_{index}", layout, index)
                      for layout in ("column_stride", "row_padding", "transposed", "unaligned")
                      for index in range(4)]
            cases += [("dense_changed", "dense", None), ("all_column_stride", "column_stride", -1)]
            retained = None
            snapshots = None
            for case_number, (name, layout, changed) in enumerate(cases):
                inputs = [layout_copy(tensor, layout if changed in (index, -1) else "dense")
                          for index, tensor in enumerate(base)]
                # Every call uses changed values as well as current strides.
                inputs[0].mul_(1 + case_number / 16)
                originals = [tensor.clone() for tensor in inputs]
                actual = run(*inputs)
                expected = reference.run(*inputs)
                assert type(actual) is tuple and len(actual) == 3
                outputs = []
                for tensor, ref in zip(actual, expected):
                    assert type(tensor) is torch.Tensor
                    assert tensor.shape == ref.shape and tensor.dtype == ref.dtype
                    assert tensor.device == ref.device and tensor.stride() == ref.stride()
                    assert all(tensor.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
                               for value in inputs)
                    error, failed = compute_error_stats(tensor, ref, tolerance)
                    outputs.append({"passed": not failed, "bitwise_equal": torch.equal(tensor, ref),
                                    "error": error.model_dump()})
                assert all(torch.equal(tensor, original) for tensor, original in zip(inputs, originals))
                if retained is not None:
                    assert all(torch.equal(tensor, old) for tensor, old in zip(retained, snapshots))
                    assert all(tensor.untyped_storage().data_ptr() != old.untyped_storage().data_ptr()
                               for tensor in actual for old in retained)
                retained = actual
                snapshots = tuple(tensor.clone() for tensor in actual)
                record = {
                    "batch": batch, "sequence": sequence, "case": name,
                    "uses_tma": supports_tma(inputs), "strides": [tensor.stride() for tensor in inputs],
                    "alignment_mod16": [tensor.data_ptr() % 16 for tensor in inputs],
                    "passed": all(row["passed"] for row in outputs), "outputs": outputs,
                }
                assert record["uses_tma"] == (changed is None)
                report["records"].append(record)
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(record), flush=True)
    if not all(row["passed"] for row in report["records"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
