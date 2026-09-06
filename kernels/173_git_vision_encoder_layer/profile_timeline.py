# SPDX-License-Identifier: Apache-2.0
"""Inspect user-kernel execution and launch gaps using the pinned CUPTI collector."""

import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess

import torch
from sol_execbench.core.bench.cupti_utils import collect_cupti_activities
from sol_execbench.core.bench.io import gen_inputs
from sol_execbench.core.bench.timing import GPU_TIMING_ACTIVITY_KINDS
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload

from kernel import run


def processes():
    return subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader,nounits"], text=True).strip()


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    problem = root / ".work/problems/173"
    definition = Definition.model_validate_json((problem / "definition.json").read_text())
    workloads = [Workload.model_validate_json(line) for line in (problem / "workload.jsonl").read_text().splitlines()]
    reference = runpy.run_path(str(problem / "reference.py"))["run"]
    records = []
    with Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock"))).open("a+") as lock:
        print("Waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        for index in [0, 2]:
            inputs = gen_inputs(definition, workloads[index], "cuda")
            output = torch.empty_like(inputs[0])
            for label, call in [("candidate", lambda: run(*inputs, output)), ("reference", lambda: reference(*inputs))]:
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                before = processes()
                for trial in range(2):
                    cache.zero_()
                    torch.cuda.synchronize()
                    with collect_cupti_activities(activity_kinds=GPU_TIMING_ACTIVITY_KINDS) as activities:
                        call()
                        torch.cuda.synchronize()
                    kernels = sorted(activities.kernels, key=lambda event: event.start)
                    start = previous = kernels[0].start
                    timeline = []
                    for event in kernels:
                        timeline.append({"name": event.name, "duration_us": (event.end - event.start) / 1000,
                                         "start_us": (event.start - start) / 1000,
                                         "preceding_gap_us": (event.start - previous) / 1000})
                        previous = event.end
                    record = {"workload": index, "implementation": label, "trial": trial,
                              "total_us": (kernels[-1].end - start) / 1000,
                              "active_us": sum(event["duration_us"] for event in timeline),
                              "processes_before": before, "processes_after": processes(),
                              "timeline": timeline}
                    records.append(record)
                    print(json.dumps(record), flush=True)
            del inputs, output
    target = root / ".work/validation/173"
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"profile-{stamp}.json"
    path.write_text(json.dumps({"kernel_sha256": hashlib.sha256((directory / "kernel.py").read_bytes()).hexdigest(),
                               "torch_matmul_precision": torch.get_float32_matmul_precision(),
                               "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "records": records}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
