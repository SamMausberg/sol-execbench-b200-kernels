#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reduced-shape routing and derivative checks before the full MoE contract."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import torch

from kernel import run


def main():
    directory = Path(__file__).parent
    root = directory.parents[1]
    spec = importlib.util.spec_from_file_location(
        "reference_119", root / ".work/problems/119/reference.py",
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    lock = Path(os.environ.get(
        "SOL_GPU_LOCK", "/workspace/sol-execbench-b200-kernels/.work/gpu.lock",
    ))
    result = {
        "scope": "reduced-shape audit; official workloads require separate validation",
        "source_sha256": hashlib.sha256((directory / "kernel.py").read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest(),
        "records": [],
    }
    target = root / ".work/tuning/119-small-verification.json"
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        result["gpu_processes_before"] = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv",
        ], text=True)
        torch.manual_seed(119031)
        for tokens, hidden, intermediate, experts, top, concentrated in (
            (17, 128, 64, 8, 2, False),
            (37, 256, 128, 8, 4, True),
            (97, 512, 256, 16, 4, False),
        ):
            values = reference.get_inputs({
                "num_tokens": tokens, "hidden_size": hidden,
                "moe_intermediate_size": intermediate,
                "n_routed_experts": experts, "num_experts_per_tok": top,
            }, torch.device("cuda"))
            if concentrated:
                values["topk_indices"].copy_(
                    torch.arange(top, device="cuda").expand(tokens, top),
                )
            inputs = list(values.values())
            originals = [value.clone() for value in inputs]
            expected = reference.run(*inputs)
            outputs = [torch.full_like(value, float("nan")) for value in expected]
            returned = run(*inputs, *outputs)
            if returned is not None:
                raise AssertionError("DPS entry point must return None")
            errors = []
            for actual, correct in zip(outputs, expected):
                torch.testing.assert_close(actual, correct, atol=1e-4, rtol=1e-4)
                errors.append(float((actual - correct).abs().max().item()))
            for actual, original in zip(inputs, originals):
                torch.testing.assert_close(actual, original, atol=0, rtol=0)
            record = {
                "tokens": tokens, "hidden": hidden, "intermediate": intermediate,
                "experts": experts, "top": top, "concentrated": concentrated,
                "passed": True, "max_absolute_errors": errors,
            }
            result["records"].append(record)
            print(json.dumps(record), flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2) + "\n")
        result["gpu_processes_after"] = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv",
        ], text=True)
        target.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
