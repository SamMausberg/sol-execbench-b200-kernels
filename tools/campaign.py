#!/usr/bin/env python3
"""Fetch pinned SOL contracts, package sources, and record native B200 evaluations."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work"
EVALUATOR = ROOT / "third_party/sol-execbench"
API = "https://research.nvidia.com/benchmarks/sol-execbench/api"
COLLECTIONS = ("L1", "L2", "Quant", "FlashInfer-Bench")


class CampaignError(RuntimeError):
    pass


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def encoded(value, *, sort_keys=True):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=sort_keys, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path, value, *, sort_keys=True):
    write(path, encoded(value, sort_keys=sort_keys))


def write_jsonl(path, rows):
    write(path, b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode() for row in rows))


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.PIPE).strip()


def lock_data():
    lock = read_json(ROOT / "benchmark.lock.json")
    extra = read_json(ROOT / "kernels/benchmark.lock.json")
    for key in ("dataset", "evaluator"):
        if lock[key]["revision"] != extra[key]["revision"]:
            raise CampaignError(f"Repository {key} revision locks disagree")
    return lock


def verify_evaluator():
    lock = lock_data()
    if git("rev-parse", "HEAD", cwd=EVALUATOR) != lock["evaluator"]["revision"]:
        raise CampaignError("Evaluator checkout differs from benchmark.lock.json")
    if git("status", "--porcelain", "--untracked-files=no", cwd=EVALUATOR):
        raise CampaignError("Evaluator checkout has tracked modifications")
    return lock


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "sol-native-campaign/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def kernel_snapshot(problem_id):
    url = f"{API}/kernels/{problem_id}"
    payload = download(url)
    data = json.loads(payload)["data"]
    if data.get("id") != problem_id or not data.get("workloads"):
        raise CampaignError(f"Invalid kernel API response for {problem_id}")
    return {"url": url, "retrieved_at": now(), "response_sha256": sha(payload), "response": json.loads(payload)}


def contract_digest(definition, workloads):
    payload = json.dumps({"definition": definition, "workloads": workloads}, ensure_ascii=False, separators=(",", ":"))
    return sha(payload.encode())


def load_problem(problem_id):
    directory = WORK / "problems" / str(problem_id)
    definition = read_json(directory / "definition.json")
    workloads = read_jsonl(directory / "workload.jsonl")
    metadata = read_json(directory / "provenance.json")
    if metadata["website_id"] != problem_id:
        raise CampaignError("Problem provenance has a different website ID")
    if metadata["dataset_revision"] != lock_data()["dataset"]["revision"]:
        raise CampaignError("Extracted problem belongs to a different dataset revision")
    if contract_digest(definition, workloads) != metadata["contract_sha256"]:
        raise CampaignError("Extracted problem contract changed; fetch the pinned contract again")
    return directory, definition, workloads, metadata


def command_fetch(args):
    import pyarrow.parquet as parquet

    lock = lock_data()
    revision = lock["dataset"]["revision"]
    repo = lock["dataset"]["repository"]
    tree_url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/data"
    tree = json.loads(download(tree_url))
    write_json(WORK / "data" / f"manifest-{revision}.json", {"url": tree_url, "files": tree})
    files = {row["path"]: row for row in tree}
    rows = []
    for collection in COLLECTIONS:
        relative = f"data/{collection}.parquet"
        expected = files[relative]["lfs"]["oid"]
        path = WORK / relative
        if not path.exists():
            payload = download(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{relative}?download=true")
            if sha(payload) != expected:
                raise CampaignError(f"Downloaded dataset SHA-256 mismatch: {collection}")
            write(path, payload)
        if sha(path.read_bytes()) != expected:
            raise CampaignError(f"Dataset SHA-256 mismatch: {path}")
        rows.extend((collection, expected, row) for row in parquet.read_table(path).to_pylist())

    known = read_json(ROOT / "kernels/benchmark.lock.json")["problems"]
    known[str(lock["problem"]["id"])] = {"contract_sha256": lock["contract"]["sha256"]}
    for problem_id in args.ids:
        snapshot = kernel_snapshot(problem_id)
        live = snapshot["response"]["data"]
        matches = [(collection, digest, row) for collection, digest, row in rows if row["name"] == live["name"]]
        if len(matches) != 1:
            raise CampaignError(f"Website problem {problem_id} ({live['name']}) has {len(matches)} pinned dataset matches")
        collection, dataset_sha, row = matches[0]
        definition = {
            "name": row["name"], "hf_id": row.get("hf_id"), "description": row["description"],
            "axes": json.loads(row["axes"]), "custom_inputs_entrypoint": row.get("custom_inputs_entrypoint"),
            "inputs": json.loads(row["inputs"]), "outputs": json.loads(row["outputs"]), "reference": row["reference"],
        }
        workloads = json.loads(row["workloads"])
        digest = contract_digest(definition, workloads)
        if str(problem_id) in known and digest != known[str(problem_id)]["contract_sha256"]:
            raise CampaignError(f"Problem {problem_id} disagrees with its committed contract hash")
        match_scores(definition, workloads, live)
        directory = WORK / "problems" / str(problem_id)
        write_json(directory / "definition.json", definition, sort_keys=False)
        write(directory / "reference.py", row["reference"].encode())
        write_jsonl(directory / "workload.jsonl", workloads)
        write_json(directory / "score-source.json", snapshot)
        write_json(directory / "provenance.json", {
            "website_id": problem_id, "name": row["name"], "collection": collection,
            "dataset_repository": repo, "dataset_revision": revision, "dataset_sha256": dataset_sha,
            "dataset_metadata_url": tree_url, "contract_sha256": digest, "workload_count": len(workloads),
        })
        print(f"{problem_id}: {row['name']}, {len(workloads)} workloads, {directory}")


def package(problem_id, source):
    from sol_execbench.core import Solution

    _, definition, _, _ = load_problem(problem_id)
    source = Path(source).resolve()
    manifest = read_json(source)
    if manifest.get("definition") != definition["name"]:
        raise CampaignError(f"Manifest definition must be {definition['name']}")
    manifest["spec"]["target_hardware"] = ["B200"]
    seen = set()
    for entry in manifest["sources"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise CampaignError(f"Invalid or duplicate source path: {relative}")
        seen.add(str(relative))
        if "content" not in entry:
            source_file = (source.parent / relative).resolve()
            if not source_file.is_relative_to(source.parent):
                raise CampaignError(f"Source symlink escapes manifest directory: {relative}")
            entry["content"] = source_file.read_text()
    Solution(**manifest)
    return encoded(manifest)


def command_package(args):
    verify_evaluator()
    payload = package(args.problem_id, args.solution)
    output = Path(args.output or ROOT / "dist" / f"{args.problem_id}-b200.json").resolve()
    write(output, payload)
    write(output.with_suffix(output.suffix + ".sha256"), f"{sha(payload)}  {output.name}\n".encode())
    print(f"{output}\nSHA-256: {sha(payload)}")


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=30)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": command, "error": str(error)}


def gpu_snapshot():
    return capture(["nvidia-smi", "--query-gpu=uuid,name,driver_version,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total", "--format=csv"])


def environment():
    keys = ("CUDA_HOME", "CUDACXX", "CUTLASS_DIR", "CPLUS_INCLUDE_PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "PATH", "PYTHONPATH", "MAX_JOBS", "SOL_GPU_LOCK", "CUDA_VISIBLE_DEVICES", "SOL_EXECBENCH_CLOCKS_LOCKED", "SOL_EXECBENCH_GPU_CLK_MHZ", "SOL_EXECBENCH_DRAM_CLK_MHZ", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "CUTE_DSL_CACHE_DIR", "TMPDIR")
    return {
        "recorded_at": now(), "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions() if dist.metadata["Name"])),
        "environment": {key: os.environ[key] for key in keys if key in os.environ},
        "gpu": gpu_snapshot(), "gpu_processes": capture(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]),
        "nvcc": capture([os.environ.get("CUDACXX", "nvcc"), "--version"]), "gcc": capture(["gcc", "--version"]),
        "repository_commit": git("rev-parse", "HEAD"), "repository_status": git("status", "--porcelain"),
        "evaluator_commit": git("rev-parse", "HEAD", cwd=EVALUATOR),
        "evaluator_status": git("status", "--porcelain", "--untracked-files=no", cwd=EVALUATOR),
        "cutlass": capture(["git", "-C", os.environ.get("CUTLASS_DIR", "/workspace/upstream/cutlass"), "rev-parse", "HEAD"]),
        "file_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in (ROOT / "benchmark.lock.json", ROOT / "kernels/benchmark.lock.json", EVALUATOR / "uv.lock", Path(__file__), ROOT / "tools/native_env.sh", ROOT / "tools/bootstrap_native.sh")},
    }


def command_info(args):
    snapshot = environment()
    if args.output:
        write_json(args.output, snapshot)
    print(json.dumps(snapshot, indent=2))


def match_scores(definition, workloads, live):
    if live["name"] != definition["name"] or len(live["workloads"]) != len(workloads):
        raise CampaignError("Live score source and pinned contract differ in name or workload count")
    constants = {key: axis["value"] for key, axis in definition["axes"].items() if axis["type"] == "const"}
    used = set()
    matches = {}
    for workload in workloads:
        axes = {**constants, **workload["axes"]}
        found = [i for i, item in enumerate(live["workloads"]) if all(item["axes"].get(key) == value for key, value in axes.items())]
        if len(found) != 1 or found[0] in used:
            raise CampaignError(f"Cannot uniquely match workload axes to score source: {axes}")
        used.add(found[0])
        item = live["workloads"][found[0]]
        if not all(isinstance(item.get(key), (int, float)) and math.isfinite(item[key]) and item[key] >= 0 for key in ("baseline_latency_ms", "sol_ms")):
            raise CampaignError(f"Missing or invalid scoring baseline for {axes}")
        if workload["uuid"] in matches:
            raise CampaignError("Pinned workload UUIDs are not unique")
        matches[workload["uuid"]] = item
    return matches


def workload_score(latency, baseline, sol):
    # Exact formula from the pinned evaluator's sol_score.py. Do not score a mean latency.
    gap = baseline - sol
    if gap <= 0:
        return 1.0 if latency <= sol else 0.0
    denominator = 1.0 + (latency - sol) / gap
    if denominator <= 0:
        raise CampaignError("SOL formula is undefined for this below-bound observation")
    return 1.0 / denominator


@contextlib.contextmanager
def gpu_lock(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        print(f"Waiting for exclusive GPU lock: {path}", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        print("GPU lock acquired", flush=True)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def summarize(run_dir):
    run_dir = Path(run_dir)
    run = read_json(run_dir / "run.json")
    solution_bytes = (run_dir / "solution.json").read_bytes()
    if sha(solution_bytes) != run["submission_sha256"]:
        raise CampaignError("Archived solution bytes differ from recorded package hash")
    manifest = json.loads(solution_bytes)
    definition = read_json(run_dir / "definition.json")
    full_workloads = read_jsonl(run_dir / "all-workloads.jsonl")
    if contract_digest(definition, full_workloads) != run["contract"]["contract_sha256"]:
        raise CampaignError("Archived contract differs from recorded contract hash")
    score_source = read_json(run_dir / "score-source.json")
    if sha((run_dir / "score-source.json").read_bytes()) != run["score_source_sha256"]:
        raise CampaignError("Archived scoring source differs from its recorded hash")
    live = score_source["response"]["data"]
    if live["id"] != run["website_id"]:
        raise CampaignError("Archived scoring source belongs to a different website ID")
    matched = match_scores(definition, full_workloads, live)
    expected = {full_workloads[index]["uuid"]: full_workloads[index] for index in run["workload_indices"]}
    observations = {uuid: [] for uuid in expected}
    trace_hashes = {}
    for trial in range(1, run["requested_trials"] + 1):
        trace = run_dir / f"trial-{trial}.jsonl"
        rows = read_jsonl(trace) if trace.exists() else []
        if trace.exists():
            trace_hashes[trace.name] = sha(trace.read_bytes())
        seen = set()
        for row in rows:
            uuid = row["workload"]["uuid"]
            if uuid not in expected or uuid in seen:
                raise CampaignError(f"Unexpected or duplicate workload in {trace.name}: {uuid}")
            if row["definition"] != definition["name"] or row.get("solution") != manifest["name"] or row["workload"]["axes"] != expected[uuid]["axes"]:
                raise CampaignError(f"Trace identity differs from archived inputs: {trace.name}, {uuid}")
            seen.add(uuid)
            evaluation = row.get("evaluation") or {}
            latency = (evaluation.get("performance") or {}).get("latency_ms")
            status = evaluation.get("status", "MISSING_EVALUATION")
            if status == "PASSED" and (not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0):
                raise CampaignError(f"PASSED trace lacks a positive finite latency: {trace.name}")
            observations[uuid].append({"trial": trial, "status": status, "latency_ms": latency if status == "PASSED" else None, "environment": evaluation.get("environment")})
        for uuid in expected.keys() - seen:
            observations[uuid].append({"trial": trial, "status": "MISSING_TRACE", "latency_ms": None})
    metrics = []
    for uuid, workload in expected.items():
        rows = observations[uuid]
        passed = all(row["status"] == "PASSED" for row in rows)
        latencies = [row["latency_ms"] for row in rows if row["status"] == "PASSED"]
        median = statistics.median(latencies) if passed else None
        scoring = matched[uuid]
        scores = [workload_score(value, scoring["baseline_latency_ms"], scoring["sol_ms"]) for value in latencies] if passed else []
        metrics.append({
            "uuid": uuid, "axes": workload["axes"], "passed": passed, "observations": rows,
            "baseline_latency_ms": scoring["baseline_latency_ms"], "sol_ms": scoring["sol_ms"],
            "latency_ms_median": median, "trial_sol_scores": scores,
            "median_latency_sol_score": workload_score(median, scoring["baseline_latency_ms"], scoring["sol_ms"]) if passed else None,
            "relative_spread": (max(latencies) - min(latencies)) / median if passed else None,
            "below_sol_bound": bool(passed and median < scoring["sol_ms"]),
        })
    completed_trials = len(run["trials"]) == run["requested_trials"] and all(
        trial.get("returncode") == 0 for trial in run["trials"]
    )
    all_passed = bool(metrics) and all(item["passed"] for item in metrics) and completed_trials
    complete = all_passed and len(expected) == len(full_workloads)
    mean_score = statistics.fmean(item["median_latency_sol_score"] for item in metrics) if all_passed else None
    summary = {
        "schema_version": 1, "recorded_at": now(), "website_id": run["website_id"], "problem_name": definition["name"],
        "submission_sha256": run["submission_sha256"], "contract_sha256": run["contract"]["contract_sha256"],
        "evaluator_revision": run["evaluator_revision"], "evaluation_stack": run["evaluation_stack"],
        "requested_trials": run["requested_trials"], "passed": all_passed, "complete_problem": complete,
        "selected_workloads": len(expected), "total_workloads": len(full_workloads),
        "local_mean_sol_score": mean_score, "problem_mean_sol_score": mean_score if complete else None,
        "trial_mean_sol_scores": [statistics.fmean(item["trial_sol_scores"][trial] for item in metrics) for trial in range(run["requested_trials"])] if all_passed else [],
        "geometric_mean_latency_ms": statistics.geometric_mean(item["latency_ms_median"] for item in metrics) if all_passed else None,
        "score_aggregation": "arithmetic mean of per-workload SOL scores; per-workload latency is the median across trials",
        "score_formula": "1 / (1 + (latency_ms - sol_ms) / (baseline_latency_ms - sol_ms))",
        "score_source_url": score_source["url"], "score_source_retrieved_at": score_source["retrieved_at"],
        "score_source_sha256": sha((run_dir / "score-source.json").read_bytes()), "sol_is_dummy": live.get("sol_is_dummy"),
        "clock_mode": run["clock_mode"], "result_scope": "local estimate; official leaderboard requires server evaluation",
        "environment_file": "environment.json", "environment_sha256": sha((run_dir / "environment.json").read_bytes()),
        "trace_sha256": trace_hashes, "workloads": metrics,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def command_summarize(args):
    summary = summarize(args.run_dir)
    print(json.dumps({key: summary[key] for key in ("passed", "complete_problem", "problem_mean_sol_score", "local_mean_sol_score", "clock_mode")}, indent=2))


def execute(command, log_path, timeout, env):
    with log_path.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=ROOT)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def command_bench(args):
    lock = verify_evaluator()
    _, definition, workloads, provenance = load_problem(args.problem_id)
    payload = package(args.problem_id, args.solution)
    indices = sorted(set(args.workload)) if args.workload else list(range(len(workloads)))
    if not indices or min(indices) < 0 or max(indices) >= len(workloads):
        raise CampaignError(f"Workload index must be between 0 and {len(workloads) - 1}")
    snapshot = kernel_snapshot(args.problem_id)
    match_scores(definition, workloads, snapshot["response"]["data"])
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", args.label or json.loads(payload)["name"])[:70]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = WORK / "runs" / str(args.problem_id) / f"{stamp}-{label}-{sha(payload)[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write(run_dir / "solution.json", payload)
    write_json(run_dir / "definition.json", definition, sort_keys=False)
    write_jsonl(run_dir / "all-workloads.jsonl", workloads)
    write_jsonl(run_dir / "workload.jsonl", [workloads[index] for index in indices])
    write_json(run_dir / "score-source.json", snapshot)
    run = {
        "schema_version": 1, "started_at": now(), "website_id": args.problem_id,
        "submission_sha256": sha(payload), "contract": provenance, "workload_indices": indices,
        "source_sha256": {item["path"]: sha(item["content"].encode()) for item in json.loads(payload)["sources"]},
        "score_source_sha256": sha((run_dir / "score-source.json").read_bytes()),
        "requested_trials": args.trials, "evaluator_revision": lock["evaluator"]["revision"],
        "evaluation_stack": lock["evaluation_stack"], "clock_mode": "locked" if args.lock_clocks else "unlocked",
        "gpu_lock": str(Path(args.gpu_lock).resolve()), "trials": [],
    }
    write_json(run_dir / "run.json", run)
    print(f"Run artifacts: {run_dir}", flush=True)
    clock_lock_owned = False
    try:
        with gpu_lock(args.gpu_lock):
            env = dict(os.environ)
            env["SOL_EXECBENCH_CLOCKS_LOCKED"] = "0"
            if args.lock_clocks:
                from sol_execbench.core.bench.clock_lock import lock_clocks, unlock_clocks

                device = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
                clock_lock_owned = lock_clocks(device)
                if not clock_lock_owned:
                    raise CampaignError("GPU clock locking failed; use the default unlocked mode for this host")
                env["SOL_EXECBENCH_CLOCKS_LOCKED"] = "1"
            try:
                snapshot_env = environment()
                snapshot_env["environment"]["SOL_EXECBENCH_CLOCKS_LOCKED"] = env["SOL_EXECBENCH_CLOCKS_LOCKED"]
                write_json(run_dir / "environment.json", snapshot_env)
                for trial in range(1, args.trials + 1):
                    command = [sys.executable, "-m", "sol_execbench.cli.main", str(run_dir), "--solution", str(run_dir / "solution.json"), "--compile-timeout", str(args.compile_timeout), "--timeout", str(args.timeout), "--verbose", "-o", str(run_dir / f"trial-{trial}.jsonl")]
                    if args.lock_clocks:
                        command.append("--lock-clocks")
                    if args.keep_staging:
                        command.append("--keep-staging")
                    print(f"Official evaluator trial {trial}/{args.trials}; log: {run_dir / f'trial-{trial}.log'}", flush=True)
                    trial_record = {"trial": trial, "command": command, "started_at": now(), "gpu_before": gpu_snapshot()}
                    run["trials"].append(trial_record)
                    write_json(run_dir / "run.json", run)
                    start = time.monotonic()
                    trial_record["returncode"] = execute(command, run_dir / f"trial-{trial}.log", args.timeout + args.compile_timeout + 120, env)
                    trial_record.update({"finished_at": now(), "elapsed_seconds": time.monotonic() - start, "gpu_after": gpu_snapshot()})
                    write_json(run_dir / "run.json", run)
                    if trial_record["returncode"] != 0:
                        break
            finally:
                if clock_lock_owned:
                    unlock_clocks()
    except (Exception, KeyboardInterrupt) as error:
        run["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run["finished_at"] = now()
        write_json(run_dir / "run.json", run)
        if (run_dir / "environment.json").exists():
            summary = summarize(run_dir)
            print(f"Summary: {run_dir / 'summary.json'}", flush=True)
    print(f"Passed: {summary['passed']}; local mean SOL score: {summary['local_mean_sol_score']}; clock mode: {summary['clock_mode']}")
    if not summary["passed"]:
        raise CampaignError(f"Evaluation incomplete or failed; inspect {run_dir}")


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="extract website IDs from the pinned dataset")
    fetch.add_argument("ids", nargs="+", type=positive_int)
    fetch.set_defaults(function=command_fetch)
    package_cmd = commands.add_parser("package", aliases=["stage"], help="embed a source manifest in deterministic B200 submission JSON")
    package_cmd.add_argument("problem_id", type=positive_int)
    package_cmd.add_argument("--solution", required=True)
    package_cmd.add_argument("--output")
    package_cmd.set_defaults(function=command_package)
    info = commands.add_parser("info", help="record software, toolchain, GPU and clock state")
    info.add_argument("--output")
    info.set_defaults(function=command_info)
    bench = commands.add_parser("bench", help="run the unmodified official evaluator while holding the shared GPU lock")
    bench.add_argument("problem_id", type=positive_int)
    bench.add_argument("--solution", required=True)
    bench.add_argument("--trials", type=positive_int, default=1)
    bench.add_argument("--workload", type=int, action="append", help="zero-based index; repeat to select multiple workloads")
    bench.add_argument("--label")
    bench.add_argument("--compile-timeout", type=positive_int, default=600)
    bench.add_argument("--timeout", type=positive_int, default=3600)
    bench.add_argument("--lock-clocks", action="store_true", help="require official GPU and memory clock presets; needs host permission")
    bench.add_argument("--gpu-lock", default=os.environ.get("SOL_GPU_LOCK", "/workspace/sol-execbench-b200-kernels/.work/gpu.lock"))
    bench.add_argument("--keep-staging", action="store_true")
    bench.set_defaults(function=command_bench)
    summarize_cmd = commands.add_parser("summarize", help="recompute a run summary from archived inputs and raw traces")
    summarize_cmd.add_argument("run_dir")
    summarize_cmd.set_defaults(function=command_summarize)
    return result


def main():
    args = parser().parse_args()
    try:
        args.function(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except (CampaignError, OSError, ValueError, KeyError, ImportError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
