#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a frozen grouped-QK package with changed inputs, scalars, and strides."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.data.workload import ToleranceSpec

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("package", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
payload = args.package.read_bytes()
package = json.loads(payload)
records = []
with tempfile.TemporaryDirectory(prefix="sol-qk-variation-") as temporary:
    directory = Path(temporary)
    for source in package["sources"]:
        path = Path(source["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Package source path must stay within its staging directory")
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source["content"])
    entry, function = package["spec"]["entry_point"].split("::")
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("verified_qk", directory / entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, function)
    tolerance = ToleranceSpec(max_atol=1e-5, max_rtol=0.05)
    lock = Path(os.environ.get("SOL_GPU_LOCK", "/workspace/sol-execbench-b200-kernels/.work/gpu.lock"))
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        torch.manual_seed(94749)
        for batch, seq, strided in [(1, 128, False), (1, 293, False),
                                    (2, 773, False), (1, 1024, False), (1, 293, True)]:
            shape = (batch, 4, 256, seq) if strided else (batch, 4, seq, 256)
            key_shape = (batch, 1, 256, seq) if strided else (batch, 1, seq, 256)
            q = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            k = torch.empty(key_shape, device="cuda", dtype=torch.bfloat16)
            if strided:
                q, k = q.transpose(-1, -2), k.transpose(-1, -2)
            out = torch.empty((batch, 4, seq, seq), device="cuda", dtype=torch.bfloat16)
            for scale in (0.0, 0.13, -0.5):
                for refresh in range(2):
                    # Keep the same storage while replacing every input value.
                    q.copy_(torch.randn(q.shape, device="cuda", dtype=q.dtype))
                    k.copy_(torch.randn(k.shape, device="cuda", dtype=k.dtype))
                    ref = (torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale).bfloat16()
                    out.fill_(float("nan"))
                    candidate(q, k, scale, out)
                    error, exceeds = compute_error_stats(out, ref, tolerance)
                    record = {"batch": batch, "seq": seq, "strided": strided,
                              "scale": scale, "refresh": refresh, "passed": not exceeds,
                              "correctness": error.model_dump()}
                    records.append(record)
                    print(json.dumps(record), flush=True)
        report = {"submission_sha256": hashlib.sha256(payload).hexdigest(),
                  "passed": all(row["passed"] for row in records), "cases": records}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise SystemExit("Input variation check failed")
