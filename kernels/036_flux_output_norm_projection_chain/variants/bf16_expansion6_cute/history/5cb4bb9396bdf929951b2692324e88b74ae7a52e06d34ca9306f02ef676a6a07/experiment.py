# SPDX-License-Identifier: Apache-2.0
"""Bounded original/32x accuracy and component timing at M8192,N64,K3072."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import threading
import time

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[3]
if os.environ.get("SOL_36_CUTE_EXPANSION_LOCKED") != "1":
    lock = os.environ.get("SOL_GPU_LOCK", str(ROOT / ".work/gpu.lock"))
    print(f"Waiting for GPU lock before CUDA imports: {lock}", flush=True)
    os.execvpe("flock", ["flock", "--exclusive", lock, "timeout", "--signal=TERM", "240", sys.executable, __file__],
               {**os.environ, "SOL_36_CUTE_EXPANSION_LOCKED": "1"})

import torch
import torch.nn.functional as F
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from kernel import launch


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = DIRECTORY / "reports" / f"shared-fragments-{stamp}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    hashes = {name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
              for name in ("kernel.py", "experiment.py")}
    source_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    frozen = DIRECTORY / "history" / source_id
    frozen.mkdir(parents=True, exist_ok=True)
    for name in hashes:
        shutil.copy2(DIRECTORY / name, frozen / name)
    report = {"scope": "component experiment; unchanged output-projection input values; no problem score",
              "started_at": stamp, "source_sha256": hashes, "source_id": source_id,
              "records": [], "process_samples": [],
              "clock_mode": "unlocked", "timing_caveat": "Exploratory timing, separate from qualification wrapper"}
    sys.path.insert(0, str(ROOT / "tools"))
    from qualify_gpu import snapshot
    family, stop = {}, threading.Event()
    def monitor():
        while not stop.is_set():
            report["process_samples"].append(snapshot(os.getpid(), family))
            stop.wait(0.5)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    def record(value):
        report["records"].append(value)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(value), flush=True)
    try:
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "device": torch.cuda.get_device_name(),
                                 "allow_tf32": torch.backends.cuda.matmul.allow_tf32}
        problem = ROOT / ".work/problems/36"
        definition = Definition.model_validate_json((problem / "definition.json").read_text())
        workloads = [Workload.model_validate_json(line)
                     for line in (problem / "workload.jsonl").read_text().splitlines()]
        reference = runpy.run_path(str(problem / "reference.py"))["run"]
        with torch.no_grad():
            for index, weight_scale in [(i, s) for i in (1, 9, 13) for s in (1.0, 32.0)]:
                workload = workloads[index]
                torch.manual_seed(36000 + index)
                inputs = gen_inputs(definition, workload, "cuda")
                hidden, temb, linear_weight, linear_bias, output_weight, output_bias, eps = inputs
                linear_weight.mul_(weight_scale)
                output_weight.mul_(weight_scale)
                expected = reference(*inputs)
                norm = (hidden - hidden.mean(-1, keepdim=True)) / torch.sqrt(hidden.var(-1, keepdim=True, unbiased=False) + eps)
                shift, scale = F.linear(temb * torch.sigmoid(temb), linear_weight, linear_bias).chunk(2, -1)
                adapted = (norm * (1.0 + scale[:, None, :]) + shift[:, None, :]).flatten(0, 1)
                out = torch.empty((8192, 64), device="cuda", dtype=torch.float32)
                partials = torch.empty((4, 6, 8192, 64), device="cuda", dtype=torch.float32)
                out.fill_(float("nan"))
                partials.fill_(float("nan"))
                launch(adapted, output_weight, output_bias, out, partials)
                errors, failed = compute_error_stats(out.view_as(expected), expected, workload.tolerance)
                matched = ((out.view_as(expected) - expected).abs()
                           <= workload.tolerance.max_atol + workload.tolerance.max_rtol * expected.abs()).float().mean().item()
                record({"kind": "end_to_end_accuracy", "workload": index, "axes": workload.axes,
                        "weight_scale": weight_scale, "passed": not failed,
                        "matched_ratio": matched, "errors": errors.model_dump(),
                        "tolerance": workload.tolerance.model_dump()})
                if failed:
                    report["passed"] = False
                    record({"kind": "stopped", "reason": "Original numerical tolerance failed; timing omitted."})
                    return

            report["passed"] = True
            report["timing_inputs"] = {"workload": index, "weight_scale": weight_scale}
            spec = importlib.util.spec_from_file_location("frozen_expansion6_loader", DIRECTORY.parent / "bf16_expansion6/loader.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            bridge = module.load_bridge()
            report["baseline_source_sha256"] = bridge.source_sha256
            workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            for algorithm in bridge.algorithms(8192, 64, 3072, True, True, 0, 1)[:2]:
                index = algorithm["index"]
                latency = time_runnable(
                    lambda a, b, bias, out: bridge.baseline(a, b, bias, out, workspace, index),
                    [adapted, output_weight, output_bias], [out], "cuda", warmup=2, rep=10,
                )
                record({"kind": "timing", "path": "frozen_emulated_bf16x9", "algorithm": algorithm,
                        "latency_ms": latency})
            latency = time_runnable(launch, [adapted, output_weight, output_bias], [out, partials],
                                    "cuda", warmup=2, rep=10)
            record({"kind": "timing", "path": "shared_conversion_six_products_split4_and_reduction",
                    "latency_ms": latency})
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        thread.join(timeout=5)
        report["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {target}", flush=True)


if __name__ == "__main__":
    main()
