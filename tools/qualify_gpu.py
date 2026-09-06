#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run separate official trials while observing competing CUDA processes.

This is a CPU-side experiment helper. It does not alter kernels, the evaluator,
GPU clocks, other sessions, or the campaign's shared-lock behavior.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_table():
    """Read identities, parent PIDs, and Linux start ticks without command lines."""
    result = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            result[int(path.name)] = (int(fields[1]), int(fields[19]))
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
    return result


def update_family(table, root_pid, family):
    """Track descendants, retaining identity through reparenting and PID reuse."""
    if root_pid in table and root_pid not in family:
        family[root_pid] = table[root_pid][1]
    known = {pid for pid, identity in family.items()
             if pid in table and table[pid][1] == identity}
    while True:
        children = {pid for pid, (parent, _) in table.items()
                    if parent in known and pid not in known}
        if not children:
            break
        for pid in children:
            family[pid] = table[pid][1]
        known.update(children)


def snapshot(root_pid=None, family=None):
    started = now()
    before = process_table() if root_pid is not None else {}
    if root_pid is not None:
        update_family(before, root_pid, family)
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
               "--format=csv,noheader,nounits"]
    error = None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        if result.returncode:
            error = f"nvidia-smi exit {result.returncode}: {result.stderr.strip()}"
        output = result.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        error, output = str(exc), ""
    after = process_table() if root_pid is not None else {}
    if root_pid is not None:
        update_family(after, root_pid, family)
    contexts = []
    for fields in csv.reader(output.splitlines()):
        if not fields or not fields[0].strip():
            continue
        try:
            pid = int(fields[0].strip())
        except ValueError:
            error = f"Unrecognized compute-process record: {fields}"
            continue
        identity = after.get(pid, before.get(pid, (None, None)))[1]
        belongs = (root_pid is not None and pid in family
                   and (identity is None or family[pid] == identity))
        contexts.append({"pid": pid, "process_name": Path(fields[1].strip()).name if len(fields) > 1 else "",
                         "memory_mib": fields[2].strip() if len(fields) > 2 else "",
                         "start_ticks": identity, "campaign_descendant": belongs})
    return {"timestamp": started, "finished_at": now(), "root_pid": root_pid,
            "contexts": contexts, "error": error}


def timestamp(value):
    return dt.datetime.fromisoformat(value).timestamp()


def audit_window(samples, started_at, finished_at, max_sample_gap):
    """Conservatively qualify the recorded trial window, excluding lock waits."""
    start, finish = timestamp(started_at), timestamp(finished_at)
    selected = [sample for sample in samples
                if timestamp(sample["finished_at"]) >= start
                and timestamp(sample["timestamp"]) <= finish]
    before = [sample for sample in samples if timestamp(sample["timestamp"]) <= start]
    after = [sample for sample in samples if timestamp(sample["finished_at"]) >= finish]
    errors = []
    if not selected or not before or not after:
        errors.append("monitor does not cover the entire trial window")
    covering = ([before[-1]] if before else []) + selected + ([after[0]] if after else [])
    times = sorted(set(timestamp(sample["timestamp"]) for sample in covering))
    gap = max((b - a for a, b in zip(times, times[1:])), default=0.0)
    if gap > max_sample_gap:
        errors.append(f"maximum sample gap {gap:.3f}s exceeds {max_sample_gap:.3f}s")
    failures = [sample for sample in covering if sample["error"]]
    if failures:
        errors.append("GPU process query failed during the trial window")
    foreign = [{"timestamp": sample["timestamp"], **context}
               for sample in selected for context in sample["contexts"]
               if not context["campaign_descendant"]]
    return {"clean": not foreign and not errors, "foreign_context_observations": foreign,
            "monitor_errors": errors, "samples_in_window": len(selected),
            "max_sample_gap_seconds": gap,
            "qualification": "No foreign CUDA context observed in the sampled trial window. "
                             "Sub-sample overlap cannot be excluded." if not foreign and not errors
                             else "Excluded from performance qualification."}


def record_sample(handle, sample, phase):
    handle.write(json.dumps({"phase": phase, **sample}, separators=(",", ":")) + "\n")
    handle.flush()


def wait_idle(args, handle, deadline):
    wait_deadline = min(deadline, time.monotonic() + args.idle_timeout)
    quiet_since = None
    next_status = 0.0
    while time.monotonic() < wait_deadline:
        sample = snapshot()
        record_sample(handle, sample, "idle_wait")
        if sample["error"] or sample["contexts"]:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= args.idle_seconds:
            return True
        if time.monotonic() >= next_status:
            pids = [row["pid"] for row in sample["contexts"]]
            print(f"Waiting for idle GPU; observed CUDA PIDs: {pids}", flush=True)
            next_status = time.monotonic() + 30
        time.sleep(args.poll_interval)
    return False


def run_attempt(args, directory, number):
    log_path = directory / f"attempt-{number}.log"
    monitor_path = directory / f"attempt-{number}-processes.jsonl"
    label = f"{args.label}-{directory.name.split('-')[0]}-a{number}"
    command = [sys.executable, str(ROOT / "tools/campaign.py"), "bench", str(args.problem_id),
               "--solution", str(args.solution), "--trials", "1", "--label", label,
               "--timeout", str(args.timeout), "--compile-timeout", str(args.compile_timeout)]
    if args.keep_staging:
        command.append("--keep-staging")
    samples, family = [], {}
    print(f"Starting attempt {number}; log: {log_path}", flush=True)
    with log_path.open("w") as output, monitor_path.open("w") as monitor:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        next_sample = time.monotonic()
        while True:
            sample = snapshot(process.pid, family)
            samples.append(sample)
            record_sample(monitor, sample, "campaign")
            if process.poll() is not None:
                break
            next_sample += args.poll_interval
            time.sleep(max(0.0, next_sample - time.monotonic()))
    matches = re.findall(r"^Run artifacts: (.+)$", log_path.read_text(), re.MULTILINE)
    record = {"attempt": number, "command": command, "returncode": process.returncode,
              "campaign_root_pid": process.pid, "campaign_process_identities": family,
              "log_path": str(log_path.relative_to(ROOT)), "log_sha256": sha(log_path),
              "monitor_path": str(monitor_path.relative_to(ROOT)), "monitor_sha256": sha(monitor_path)}
    if not matches:
        return {**record, "status": "evaluation_failed", "reason": "campaign did not record a run directory"}
    run_dir = Path(matches[-1])
    record["run_directory"] = str(run_dir.relative_to(ROOT))
    if not (run_dir / "summary.json").exists():
        return {**record, "status": "evaluation_failed", "reason": "campaign did not finish a summary"}
    summary = json.loads((run_dir / "summary.json").read_text())
    run = json.loads((run_dir / "run.json").read_text())
    record.update({"summary_sha256": sha(run_dir / "summary.json"),
                   "submission_sha256": summary["submission_sha256"],
                   "passed": summary["passed"], "local_sol_score": summary["local_mean_sol_score"]})
    if process.returncode or not summary["passed"] or not summary["complete_problem"] or len(run["trials"]) != 1:
        return {**record, "status": "evaluation_failed", "reason": "official full-trial validation failed"}
    trial = run["trials"][0]
    audit = audit_window(samples, trial["started_at"], trial["finished_at"], args.poll_interval * 3 + 0.25)
    record.update({"trial_started_at": trial["started_at"], "trial_finished_at": trial["finished_at"],
                   "audit": audit, "status": "qualified" if audit["clean"] else "contaminated"})
    return record


def aggregate(attempts):
    qualified = [a for a in attempts if a["status"] == "qualified"]
    if not qualified:
        return None
    if len({a["submission_sha256"] for a in qualified}) != 1:
        raise RuntimeError("solution changed between qualified trials")
    summaries = [json.loads((ROOT / a["run_directory"] / "summary.json").read_text()) for a in qualified]
    if len({s["contract_sha256"] for s in summaries}) != 1:
        raise RuntimeError("workload contract changed between qualified trials")
    rows = []
    for first in summaries[0]["workloads"]:
        observations = [next(w for w in s["workloads"] if w["uuid"] == first["uuid"]) for s in summaries]
        if any(w["baseline_latency_ms"] != first["baseline_latency_ms"] or w["sol_ms"] != first["sol_ms"] for w in observations):
            raise RuntimeError("scoring constants changed between qualified trials")
        times = [o["latency_ms"] for w in observations for o in w["observations"]]
        median = statistics.median(times)
        baseline, sol = first["baseline_latency_ms"], first["sol_ms"]
        rows.append({"uuid": first["uuid"], "axes": first["axes"], "latencies_ms": times,
                     "median_latency_ms": median, "baseline_latency_ms": baseline, "sol_ms": sol,
                     "score": (baseline - sol) / (median + baseline - 2 * sol),
                     "relative_spread": (max(times) - min(times)) / median})
    return {"qualified_trials": len(qualified), "submission_sha256": qualified[0]["submission_sha256"],
            "contract_sha256": summaries[0]["contract_sha256"], "clock_mode": summaries[0]["clock_mode"],
            "local_sol_score": statistics.mean(w["score"] for w in rows),
            "trial_sol_scores": [a["local_sol_score"] for a in qualified], "workloads": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem_id", type=int)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--idle-timeout", type=float, default=180)
    parser.add_argument("--idle-seconds", type=float, default=2)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--deadline", type=float, default=1800, help="stop starting attempts after this many seconds")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--label", default="qualified")
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.max_attempts < args.trials or min(args.poll_interval, args.deadline, args.idle_timeout) <= 0:
        parser.error("use positive bounds with max-attempts at least trials")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("label must use letters, digits, underscores or hyphens")
    args.solution = args.solution.resolve()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / ".work/qualification" / str(args.problem_id) / f"{stamp}-{args.label}"
    directory.mkdir(parents=True)
    deadline = time.monotonic() + args.deadline
    report = {"schema_version": 1, "problem_id": args.problem_id, "started_at": now(),
              "requested_clean_trials": args.trials, "max_attempts": args.max_attempts,
              "sampling_seconds": args.poll_interval, "tool_sha256": sha(__file__),
              "scope": "local timing qualification from sampled CUDA process identities; no hosted evaluation",
              "attempts": [], "complete": False}
    write_json(directory / "qualification.json", report)
    with (directory / "idle-processes.jsonl").open("w") as idle_log:
        for attempt in range(1, args.max_attempts + 1):
            if not wait_idle(args, idle_log, deadline):
                report["stop_reason"] = "idle wait or campaign deadline reached"
                break
            result = run_attempt(args, directory, attempt)
            report["attempts"].append(result)
            print(f"Attempt {attempt}: {result['status']}; local score {result.get('local_sol_score')}", flush=True)
            report["aggregate"] = aggregate(report["attempts"])
            report["complete"] = sum(a["status"] == "qualified" for a in report["attempts"]) >= args.trials
            write_json(directory / "qualification.json", report)
            if result["status"] == "evaluation_failed":
                report["stop_reason"] = "evaluation failed; automatic retries are limited to contention or monitoring failures"
                break
            if report["complete"]:
                report["stop_reason"] = "requested qualified trials complete"
                break
        else:
            report["stop_reason"] = "attempt limit reached"
    report["finished_at"] = now()
    write_json(directory / "qualification.json", report)
    print(f"Qualification: {directory / 'qualification.json'}", flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
