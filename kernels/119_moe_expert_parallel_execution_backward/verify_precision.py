#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check real hidden dimensions with fewer experts against the pinned reference."""

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
    parser.add_argument("--module", default="tma.py")
    parser.add_argument("--label", default="tma")
    parser.add_argument("--config", type=json.loads, default={})
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    source = directory / args.module
    kernel = load(source, "candidate_119")
    for key, value in args.config.items():
        if key not in {"FORWARD_PRECISION", "BACKWARD_PRECISION", "PROJECT_TILE", "WEIGHT_TILE", "INPUT_TILE", "GEMM_ARCH"}:
            parser.error(f"Unsupported tuning parameter: {key}")
        setattr(kernel, key, value)
    reference_path = root / ".work/problems/119/reference.py"
    reference = load(reference_path, "reference_119")
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    output_path = root / f".work/tuning/119-precision-{args.label}.json"
    report = {
        "scope": "full hidden dimensions with fewer experts; separate official validation required",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dependency_sha256": {"kernel.py": hashlib.sha256((directory / "kernel.py").read_bytes()).hexdigest()},
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "config": args.config,
        "tolerance": {"max_atol": 0.11, "max_rtol": 1e-5, "required_matched_ratio": 0.99},
        "records": [],
    }
    tolerance = ToleranceSpec(**report["tolerance"])
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for tokens, experts, concentrated, seed in (
            (64, 8, False, 119321),
            (137, 16, False, 119654),
            (64, 16, True, 119987),
        ):
            torch.manual_seed(seed)
            values = reference.get_inputs({
                "num_tokens": tokens, "hidden_size": 4096,
                "moe_intermediate_size": 2048,
                "n_routed_experts": experts, "num_experts_per_tok": 8,
            }, torch.device("cuda"))
            if concentrated:
                values["topk_indices"].copy_(torch.arange(8, device="cuda").expand(tokens, 8))
            inputs = list(values.values())
            before = [value.clone() for value in inputs]
            expected = reference.run(*inputs)
            outputs = [torch.full_like(value, float("nan")) for value in expected]
            returned = kernel.run(*inputs, *outputs)
            if returned is not None:
                raise AssertionError("DPS entry point must return None")
            rows = []
            for actual, correct in zip(outputs, expected):
                error, failed = compute_error_stats(actual, correct, tolerance)
                unmatched = int(((actual - correct).abs() >
                                 tolerance.max_atol + tolerance.max_rtol * correct.abs()).sum().item())
                rows.append({"passed": not failed, "errors": error.model_dump(),
                             "unmatched": unmatched, "elements": actual.numel()})
            for actual, original in zip(inputs, before):
                torch.testing.assert_close(actual, original, atol=0, rtol=0)
            record = {"tokens": tokens, "experts": experts, "concentrated": concentrated,
                      "seed": seed, "outputs": rows, "passed": all(row["passed"] for row in rows)}
            report["records"].append(record)
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            del values, inputs, before, expected, outputs
            torch.cuda.empty_cache()
    if not all(row["passed"] for row in report["records"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
