"""Show the CUPTI activity timeline for the attention implementation."""

import argparse
import fcntl
import json
from pathlib import Path
import runpy

import torch
from sol_execbench.core.bench.cupti_utils import collect_cupti_activities
from sol_execbench.core.bench.timing import GPU_TIMING_ACTIVITY_KINDS

from kernel import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=int, action="append")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    reference = runpy.run_path(str(root / ".work/problems/121/reference.py"))
    workloads = [json.loads(line) for line in (root / ".work/problems/121/workload.jsonl").read_text().splitlines()]
    records = []
    with (root / ".work/gpu.lock").open("a+") as lock:
        print("Waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        for index in args.workload or [6, 3]:
            inputs = reference["get_inputs"](workloads[index]["axes"], torch.device("cuda"))
            output = torch.empty_like(inputs["hidden_states"])
            for _ in range(3):
                run(*inputs.values(), output)
            for trial in range(2):
                cache.zero_()
                torch.cuda.synchronize()
                with collect_cupti_activities(activity_kinds=GPU_TIMING_ACTIVITY_KINDS) as activities:
                    run(*inputs.values(), output)
                    torch.cuda.synchronize()
                kernels = sorted(activities.kernels, key=lambda event: event.start)
                start = kernels[0].start
                previous = start
                timeline = []
                for event in kernels:
                    timeline.append({"name": event.name, "duration_us": (event.end - event.start) / 1000,
                                     "start_us": (event.start - start) / 1000,
                                     "preceding_gap_us": (event.start - previous) / 1000})
                    previous = event.end
                record = {"workload": index, "trial": trial,
                          "total_us": (kernels[-1].end - start) / 1000,
                          "active_us": sum(event["duration_us"] for event in timeline),
                          "timeline": timeline}
                print(json.dumps(record), flush=True)
                records.append(record)
            del inputs, output
    path = root / ".work/validation/121/cupti-profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
