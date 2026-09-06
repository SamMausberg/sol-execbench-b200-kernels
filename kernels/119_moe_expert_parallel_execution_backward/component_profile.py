#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CUPTI component profile of a full-size MoE backward invocation."""

import argparse
from collections import defaultdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import threading

import torch

from sol_execbench.core.bench.cupti_utils import collect_cupti_activities
from sol_execbench.core.bench.timing import (
    GPU_TIMING_ACTIVITY_KINDS,
    _clear_cache,
    _get_empty_cache_for_benchmark,
    _reset_persisting_l2_cache,
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--module", default="kernel.py")
    parser.add_argument("--label", default="initial")
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
    sys.path.insert(0, str(root / "tools"))
    from qualify_gpu import snapshot

    target = root / f".work/tuning/119-profile-{args.label}.json"
    lock = Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock")))
    record = {
        "scope": "diagnostic component profile; fixed addresses and unlocked GPU",
        "tokens": args.tokens,
        "source": str(source.relative_to(root)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dependency_sha256": {"kernel.py": hashlib.sha256((directory / "kernel.py").read_bytes()).hexdigest()},
        "config": args.config,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "process_samples": [],
        "trials": [],
    }
    finished = threading.Event()
    family = {}

    def monitor():
        while not finished.is_set():
            sample = snapshot(os.getpid(), family)
            for row in sample["contexts"]:
                row["process_name"] = Path(row["process_name"]).name
            record["process_samples"].append(sample)
            finished.wait(0.5)

    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        try:
            torch.manual_seed(119119)
            values = reference.get_inputs({
                "num_tokens": args.tokens, "hidden_size": 4096,
                "moe_intermediate_size": 2048,
                "n_routed_experts": 256, "num_experts_per_tok": 8,
            }, torch.device("cuda"))
            outputs = [torch.empty_like(values[name]) for name in (
                "hidden_states", "topk_weights", "gate_weights", "up_weights", "down_weights",
            )]
            inputs = list(values.values())
            cache = _get_empty_cache_for_benchmark("cuda")
            print("Inputs ready; warming kernels", flush=True)
            kernel.run(*inputs, *outputs)
            torch.cuda.synchronize()
            for trial in range(3):
                _reset_persisting_l2_cache("cuda")
                _clear_cache(cache)
                torch.cuda.synchronize()
                with collect_cupti_activities(
                    activity_kinds=GPU_TIMING_ACTIVITY_KINDS,
                ) as activities:
                    kernel.run(*inputs, *outputs)
                    torch.cuda.synchronize()
                rows = [{"name": item.name, "duration_ms": (item.end - item.start) / 1e6}
                        for item in sorted(activities.kernels, key=lambda item: item.start)]
                span = (max(item.end for item in activities.kernels)
                        - min(item.start for item in activities.kernels)) / 1e6
                result = {"trial": trial, "span_ms": span, "activities": rows}
                record["trials"].append(result)
                print(json.dumps(result), flush=True)
            components = defaultdict(list)
            for trial in record["trials"]:
                ordinal = defaultdict(int)
                for row in trial["activities"]:
                    index = ordinal[row["name"]]
                    ordinal[row["name"]] += 1
                    components[f"{row['name']}:{index}"].append(row["duration_ms"])
            record["component_medians_ms"] = {
                name: statistics.median(values) for name, values in components.items()
            }
            record["span_median_ms"] = statistics.median(
                trial["span_ms"] for trial in record["trials"]
            )
            record["max_allocated_bytes"] = torch.cuda.max_memory_allocated()
        finally:
            finished.set()
            worker.join(timeout=10)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Profile: {target}", flush=True)


if __name__ == "__main__":
    main()
