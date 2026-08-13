#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 sol-038-flux-rmsnorm contributors
# SPDX-License-Identifier: Apache-2.0

"""Repository automation for SOL-ExecBench kernel 38."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "benchmark.lock.json"
KERNEL_DIR = ROOT / "kernels" / "038_flux_multi_head_rmsnorm_qk"
SOLUTION_PATH = KERNEL_DIR / "solution.json"
DATA_PATH = ROOT / ".work" / "data" / "L1.parquet"
PROBLEM_DIR = ROOT / ".work" / "problem-038"
EVALUATOR_DIR = ROOT / "third_party" / "sol-execbench"
RESULTS_PATH = ROOT / "results" / "results.jsonl"

PROBLEM_ID = 38
PROBLEM_NAME = "038_flux_multi_head_rmsnorm_qk"
WORKLOAD_COUNT = 16
DATASET_REVISION = "63699402f003496acc3af4eb534a5304a8ac1ea9"
DATASET_SHA256 = "dbaf8476812194920e405118cc5f4abc65035b79a4307b5ee88fb07666ceed48"
EVALUATOR_REVISION = "a9fa0804c793d438e70850c33fe34426e66d53dd"
CONTRACT_SHA256 = "c80cf9574612ec24ba6e67d6142ab67ab2ef3ff71d1cf7c29ba8224535665d4f"
EVALUATION_STACK = "v1.1"
DATASET_URL = (
    "https://huggingface.co/datasets/nvidia/SOL-ExecBench/resolve/"
    f"{DATASET_REVISION}/data/L1.parquet?download=true"
)
LEADERBOARD_URL = (
    "https://research.nvidia.com/benchmarks/sol-execbench/api/leaderboard/"
    f"kernel/{PROBLEM_ID}/B200?evaluation_stack_version={EVALUATION_STACK}"
)
SOURCE_PATHS = ("binding.cpp", "kernel.cu", "kernel.cuh")


class RepoError(RuntimeError):
    """An expected repository validation error."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RepoError(f"missing file: {path.relative_to(ROOT)}") from error
    except json.JSONDecodeError as error:
        raise RepoError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise RepoError(f"expected a JSON object in {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def ordered_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def ordered_compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(*arguments: str, cwd: Path = ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=cwd, text=True, stderr=subprocess.PIPE
        ).strip()
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or str(error)
        raise RepoError(f"git command failed: {message}") from error


def contract_digest(definition: dict[str, Any], workloads: list[Any]) -> str:
    payload = ordered_compact_json({"definition": definition, "workloads": workloads})
    return sha256_bytes(payload.encode("utf-8"))


def read_workloads(path: Path) -> list[dict[str, Any]]:
    workloads: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RepoError(f"invalid workload JSON on line {line_number}: {error}") from error
        if not isinstance(value, dict):
            raise RepoError(f"workload line {line_number} is not an object")
        workloads.append(value)
    return workloads


def validate_lock() -> dict[str, Any]:
    lock = load_json(LOCK_PATH)
    expected = {
        ("schema_version",): 1,
        ("problem", "id"): PROBLEM_ID,
        ("problem", "name"): PROBLEM_NAME,
        ("problem", "collection"): "L1",
        ("problem", "workloads"): WORKLOAD_COUNT,
        ("dataset", "repository"): "nvidia/SOL-ExecBench",
        ("dataset", "revision"): DATASET_REVISION,
        ("dataset", "path"): "data/L1.parquet",
        ("dataset", "sha256"): DATASET_SHA256,
        ("evaluator", "repository"): "https://github.com/NVIDIA/SOL-ExecBench.git",
        ("evaluator", "revision"): EVALUATOR_REVISION,
        ("evaluator", "version"): "1.0.2",
        ("evaluation_stack",): EVALUATION_STACK,
        ("contract", "sha256"): CONTRACT_SHA256,
    }
    for keys, expected_value in expected.items():
        current: Any = lock
        try:
            for key in keys:
                current = current[key]
        except (KeyError, TypeError) as error:
            raise RepoError(f"benchmark lock is missing {'.'.join(keys)}") from error
        if current != expected_value:
            raise RepoError(
                f"benchmark lock mismatch at {'.'.join(keys)}: "
                f"expected {expected_value!r}, found {current!r}"
            )
    return lock


def validate_solution(
    manifest: dict[str, Any], *, require_embedded: bool | None = False
) -> None:
    if manifest.get("definition") != PROBLEM_NAME:
        raise RepoError("solution definition does not match problem 38")
    if not manifest.get("name") or not manifest.get("author"):
        raise RepoError("solution name and author must be nonempty")
    spec = manifest.get("spec")
    if not isinstance(spec, dict):
        raise RepoError("solution spec must be an object")
    expected_spec = {
        "languages": ["cuda_cpp"],
        "entry_point": "binding.cpp::run",
        "destination_passing_style": True,
        "binding": "torch",
        "dependencies": [],
    }
    for key, expected_value in expected_spec.items():
        if spec.get(key) != expected_value:
            raise RepoError(f"invalid solution spec field {key}")
    compile_options = spec.get("compile_options")
    if compile_options != {
        "cflags": ["-O3", "-std=c++17"],
        "cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"],
        "ld_flags": ["-lcuda"],
    }:
        raise RepoError("solution compile options do not match the repository contract")
    targets = spec.get("target_hardware")
    if not isinstance(targets, list) or not targets or not set(targets) <= {"B200", "LOCAL"}:
        raise RepoError("target_hardware must contain B200, LOCAL, or both")

    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise RepoError("solution sources must be a list")
    paths = [source.get("path") for source in sources if isinstance(source, dict)]
    if paths != list(SOURCE_PATHS):
        raise RepoError(f"solution sources must be ordered as {SOURCE_PATHS}")
    for source in sources:
        if not isinstance(source, dict):
            raise RepoError("each source must be an object")
        content = source.get("content")
        if require_embedded is True and not isinstance(content, str):
            raise RepoError(f"packaged source {source['path']} has no content")
        if require_embedded is False and "content" in source:
            raise RepoError(
                "kernels/038_flux_multi_head_rmsnorm_qk/solution.json "
                "must not embed source content"
            )


def validate_abi() -> None:
    binding = (KERNEL_DIR / "binding.cpp").read_text(encoding="utf-8")
    normalized = " ".join(binding.split())
    signature = re.compile(
        r"void run\( const torch::Tensor& query, const torch::Tensor& key, "
        r"const torch::Tensor& weight_q, const torch::Tensor& weight_k, "
        r"double eps, torch::Tensor query_norm, torch::Tensor key_norm\)"
    )
    if not signature.search(normalized):
        raise RepoError("binding.cpp does not expose the required destination-passing ABI")
    if "at::cuda::getCurrentCUDAStream(query.get_device())" not in binding:
        raise RepoError("binding.cpp must launch on PyTorch's current CUDA stream")
    if 'module.def("run", &run' not in binding:
        raise RepoError("binding.cpp does not export run through pybind11")


def verify_local_contract() -> None:
    if DATA_PATH.exists():
        actual = sha256_file(DATA_PATH)
        if actual != DATASET_SHA256:
            raise RepoError(f"dataset SHA-256 mismatch: expected {DATASET_SHA256}, found {actual}")

    definition_path = PROBLEM_DIR / "definition.json"
    workload_path = PROBLEM_DIR / "workload.jsonl"
    if definition_path.exists() != workload_path.exists():
        raise RepoError("local problem extraction is incomplete")
    if not definition_path.exists():
        return
    definition = load_json(definition_path)
    workloads = read_workloads(workload_path)
    if definition.get("name") != PROBLEM_NAME:
        raise RepoError("extracted definition has the wrong problem name")
    if len(workloads) != WORKLOAD_COUNT:
        raise RepoError(f"expected {WORKLOAD_COUNT} workloads, found {len(workloads)}")
    actual = contract_digest(definition, workloads)
    if actual != CONTRACT_SHA256:
        raise RepoError(f"contract SHA-256 mismatch: expected {CONTRACT_SHA256}, found {actual}")


def command_fetch(_: argparse.Namespace) -> None:
    validate_lock()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists() or sha256_file(DATA_PATH) != DATASET_SHA256:
        print(f"fetching {DATASET_URL}")
        temporary = DATA_PATH.with_suffix(".parquet.download")
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "sol-038-repo/1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = sha256_file(temporary)
        if actual != DATASET_SHA256:
            temporary.unlink(missing_ok=True)
            raise RepoError(
                f"downloaded dataset SHA-256 mismatch: expected {DATASET_SHA256}, found {actual}"
            )
        os.replace(temporary, DATA_PATH)
    print(f"dataset SHA-256: {DATASET_SHA256}")

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RepoError("PyArrow is required; run this command through the pinned image") from error

    table = parquet.read_table(DATA_PATH)
    matches = [row for row in table.to_pylist() if row.get("name") == PROBLEM_NAME]
    if len(matches) != 1:
        raise RepoError(f"expected one row for {PROBLEM_NAME}, found {len(matches)}")
    row = matches[0]
    definition = {
        "name": row["name"],
        "hf_id": row.get("hf_id"),
        "description": row["description"],
        "axes": json.loads(row["axes"]),
        "custom_inputs_entrypoint": row.get("custom_inputs_entrypoint"),
        "inputs": json.loads(row["inputs"]),
        "outputs": json.loads(row["outputs"]),
        "reference": row["reference"],
    }
    workloads = json.loads(row["workloads"])
    actual_contract = contract_digest(definition, workloads)
    if actual_contract != CONTRACT_SHA256:
        raise RepoError(
            f"extracted contract SHA-256 mismatch: expected {CONTRACT_SHA256}, "
            f"found {actual_contract}"
        )
    atomic_write(PROBLEM_DIR / "definition.json", ordered_json_bytes(definition))
    workload_data = "".join(
        f"{ordered_compact_json(workload)}\n" for workload in workloads
    )
    atomic_write(PROBLEM_DIR / "workload.jsonl", workload_data.encode("utf-8"))
    atomic_write(PROBLEM_DIR / "contract.sha256", f"{actual_contract}\n".encode("ascii"))
    verify_local_contract()
    print(f"extracted problem {PROBLEM_ID}: {len(workloads)} workloads")
    print(f"contract SHA-256: {actual_contract}")


def lint_json_format(path: Path) -> None:
    value = load_json(path)
    if path.read_bytes() != canonical_json_bytes(value):
        raise RepoError(f"{path.relative_to(ROOT)} is not canonical formatted JSON")


def repository_files() -> list[Path]:
    output = run_git("ls-files", "--cached", "--others", "--exclude-standard")
    files: list[Path] = []
    for item in output.splitlines():
        path = ROOT / item
        if path.is_file() and not path.is_symlink():
            files.append(path)
    return files


def command_lint(_: argparse.Namespace) -> None:
    validate_lock()
    manifest = load_json(SOLUTION_PATH)
    validate_solution(manifest)
    validate_abi()
    verify_local_contract()

    required = [
        "README.md",
        "LICENSE",
        "Makefile",
        "benchmark.lock.json",
        "kernels/038_flux_multi_head_rmsnorm_qk/binding.cpp",
        "kernels/038_flux_multi_head_rmsnorm_qk/kernel.cu",
        "kernels/038_flux_multi_head_rmsnorm_qk/kernel.cuh",
        "kernels/038_flux_multi_head_rmsnorm_qk/solution.json",
        "tools/repo.py",
        "results/results.jsonl",
        "third_party/sol-execbench",
        ".github/workflows/ci.yml",
        ".gitignore",
        ".gitmodules",
    ]
    missing = [item for item in required if not (ROOT / item).exists()]
    if missing:
        raise RepoError(f"missing required repository paths: {', '.join(missing)}")

    lint_json_format(LOCK_PATH)
    lint_json_format(SOLUTION_PATH)
    if len((ROOT / "README.md").read_text(encoding="utf-8").splitlines()) > 80:
        raise RepoError("README.md must stay at or below 80 lines")

    evaluator_head = run_git("rev-parse", "HEAD", cwd=EVALUATOR_DIR)
    if evaluator_head != EVALUATOR_REVISION:
        raise RepoError(
            f"evaluator checkout mismatch: expected {EVALUATOR_REVISION}, found {evaluator_head}"
        )
    modules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    if "https://github.com/NVIDIA/SOL-ExecBench.git" not in modules:
        raise RepoError(".gitmodules does not use the official evaluator repository")

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    required_targets = (
        "setup", "info", "lint", "test", "bench", "profile", "status",
        "package", "verify-package",
    )
    for target in required_targets:
        if not re.search(rf"^{re.escape(target)}\s*:(?!=)", makefile, re.MULTILINE):
            raise RepoError(f"Makefile is missing target {target}")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RepoError("LICENSE is not Apache-2.0")
    kernel_prefix = "kernels/038_flux_multi_head_rmsnorm_qk"
    for relative in (
        "Makefile",
        "tools/repo.py",
        *[f"{kernel_prefix}/{path}" for path in SOURCE_PATHS],
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "SPDX-License-Identifier: Apache-2.0" not in text:
            raise RepoError(f"missing Apache-2.0 SPDX header in {relative}")

    # Treat an unstaged deletion as absent so contributors can validate the
    # exact fix for accidentally tracked generated artifacts before staging it.
    tracked = {
        path
        for path in run_git("ls-files").splitlines()
        if (ROOT / path).exists()
    }
    forbidden_patterns = (
        re.compile(r"(^|/)definition\.json$"),
        re.compile(r"(^|/)workload\.jsonl$"),
        re.compile(r"\.parquet$"),
        re.compile(r"(^|/)dist/"),
        re.compile(r"\.ncu-rep$"),
        re.compile(r"\.nsys-rep$"),
        re.compile(r"\.zip$"),
    )
    forbidden = sorted(
        path for path in tracked if any(pattern.search(path) for pattern in forbidden_patterns)
    )
    if forbidden:
        raise RepoError(f"generated or benchmark data is tracked: {', '.join(forbidden)}")

    text_suffixes = {".md", ".py", ".cpp", ".cu", ".cuh", ".json", ".jsonl", ".yml", ".yaml", ".txt"}
    for path in repository_files():
        if path.suffix not in text_suffixes and path.name not in {
            "Makefile", "LICENSE", ".gitignore", ".gitmodules"
        }:
            continue
        text = path.read_text(encoding="utf-8")
        if "\u2014" in text:
            raise RepoError(f"Unicode em dash found in {path.relative_to(ROOT)}")

    compile((ROOT / "tools" / "repo.py").read_text(encoding="utf-8"), "tools/repo.py", "exec")
    for line_number, line in enumerate(RESULTS_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RepoError(f"invalid results JSON on line {line_number}: {error}") from error
        if line != compact_json(record):
            raise RepoError(f"results line {line_number} is not compact canonical JSON")

    print("lint: repository structure, locks, manifest, ABI, and licenses are valid")
    if (PROBLEM_DIR / "definition.json").exists():
        print("lint: pinned local contract is valid")


def copy_sources(destination: Path) -> None:
    for source_name in SOURCE_PATHS:
        source = KERNEL_DIR / source_name
        if source.is_symlink() or not source.is_file():
            raise RepoError(f"invalid kernel source: {source_name}")
        atomic_write(destination / source_name, source.read_bytes())


def command_stage(arguments: argparse.Namespace) -> None:
    manifest = load_json(SOLUTION_PATH)
    validate_solution(manifest)
    target = arguments.target.lower()
    hardware = "LOCAL" if target == "local" else "B200"
    output = Path(arguments.output).resolve()
    if ROOT not in output.parents:
        raise RepoError("staging output must be inside the repository")
    output.mkdir(parents=True, exist_ok=True)
    staged = json.loads(json.dumps(manifest))
    staged["spec"]["target_hardware"] = [hardware]
    copy_sources(output)
    atomic_write(output / "solution.json", canonical_json_bytes(staged))
    print(f"staged {hardware} manifest at {output.relative_to(ROOT)}")


def command_select_workload(arguments: argparse.Namespace) -> None:
    problem = Path(arguments.problem).resolve()
    output = Path(arguments.output).resolve()
    workloads = read_workloads(problem / "workload.jsonl")
    index = arguments.index
    if index < 0 or index >= len(workloads):
        raise RepoError(f"workload index must be between 0 and {len(workloads) - 1}")
    if ROOT not in output.parents:
        raise RepoError("profile problem output must be inside the repository")
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / "definition.json", (problem / "definition.json").read_bytes())
    atomic_write(
        output / "workload.jsonl",
        f"{ordered_compact_json(workloads[index])}\n".encode("utf-8"),
    )
    print(f"selected workload {index}: {workloads[index].get('uuid', '<no uuid>')}")


def source_digest() -> str:
    digest = hashlib.sha256()
    for source_name in SOURCE_PATHS:
        digest.update(source_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((KERNEL_DIR / source_name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_package_bytes() -> bytes:
    manifest = load_json(SOLUTION_PATH)
    validate_solution(manifest)
    packaged = json.loads(json.dumps(manifest))
    packaged["spec"]["target_hardware"] = ["B200"]
    for source in packaged["sources"]:
        source_path = KERNEL_DIR / source["path"]
        if source_path.is_symlink() or not source_path.is_file():
            raise RepoError(f"invalid package source: {source['path']}")
        source["content"] = source_path.read_text(encoding="utf-8")
    validate_solution(packaged, require_embedded=True)
    return canonical_json_bytes(packaged)


def write_package(output: Path) -> str:
    data = build_package_bytes()
    digest = sha256_bytes(data)
    atomic_write(output, data)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    atomic_write(sidecar, f"{digest}  {output.name}\n".encode("ascii"))
    return digest


def command_package(arguments: argparse.Namespace) -> None:
    output = Path(arguments.output).resolve()
    digest = write_package(output)
    print(f"package: {output.relative_to(ROOT) if ROOT in output.parents else output}")
    print(f"SHA-256: {digest}")


def command_verify_package(arguments: argparse.Namespace) -> None:
    artifact = Path(arguments.artifact).resolve()
    expected = build_package_bytes()
    actual = artifact.read_bytes()
    if actual != expected:
        raise RepoError("package is stale or not deterministic; run make package")
    manifest = load_json(artifact)
    validate_solution(manifest, require_embedded=True)
    if manifest["spec"]["target_hardware"] != ["B200"]:
        raise RepoError("submission package must target only B200")
    digest = sha256_bytes(actual)
    sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
    expected_sidecar = f"{digest}  {artifact.name}\n"
    if not sidecar.exists() or sidecar.read_text(encoding="ascii") != expected_sidecar:
        raise RepoError("package SHA-256 sidecar is missing or stale")
    if canonical_json_bytes(json.loads(actual)) != actual:
        raise RepoError("package does not round-trip through canonical JSON")
    print(f"package verified: {digest}")


def load_trace_file(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RepoError(f"trace {path}:{line_number} is not an object")
        traces.append(value)
    if len(traces) != WORKLOAD_COUNT:
        raise RepoError(f"trace {path} has {len(traces)} workloads, expected {WORKLOAD_COUNT}")
    return traces


def rounded(value: float) -> float:
    return round(value, 9)


def command_summarize(arguments: argparse.Namespace) -> None:
    target = arguments.target.lower()
    trials = [load_trace_file(Path(item)) for item in arguments.traces]
    if len(trials) != 3:
        raise RepoError("exactly three trial trace files are required")
    uuids: list[str] = []
    hardware: set[str] = set()
    latency_by_workload: list[list[float]] = [[] for _ in range(WORKLOAD_COUNT)]
    for trial_index, traces in enumerate(trials):
        current_uuids: list[str] = []
        for workload_index, trace in enumerate(traces):
            if trace.get("definition") != PROBLEM_NAME:
                raise RepoError("trace definition does not match problem 38")
            workload = trace.get("workload") or {}
            current_uuids.append(str(workload.get("uuid", "")))
            evaluation = trace.get("evaluation") or {}
            if evaluation.get("status") != "PASSED":
                raise RepoError(
                    f"trial {trial_index + 1}, workload {workload_index} did not pass: "
                    f"{evaluation.get('status')}"
                )
            environment = evaluation.get("environment") or {}
            hardware.add(str(environment.get("hardware", "unknown")))
            performance = evaluation.get("performance") or {}
            latency = performance.get("latency_ms")
            if not isinstance(latency, (int, float)) or latency <= 0:
                raise RepoError("passed trace is missing a positive latency")
            latency_by_workload[workload_index].append(float(latency))
        if trial_index == 0:
            uuids = current_uuids
        elif current_uuids != uuids:
            raise RepoError("trial workload order or UUIDs differ")
    if target == "b200" and not all("B200" in item.upper() for item in hardware):
        raise RepoError(f"B200 promotion requires B200 traces, found {sorted(hardware)}")

    workload_summaries: list[dict[str, Any]] = []
    spreads: list[float] = []
    for uuid, latencies in zip(uuids, latency_by_workload, strict=True):
        median = statistics.median(latencies)
        spread = (max(latencies) - min(latencies)) / median if median else 0.0
        spreads.append(spread)
        workload_summaries.append(
            {
                "latency_ms_mean": rounded(statistics.fmean(latencies)),
                "latency_ms_median": rounded(median),
                "relative_spread": rounded(spread),
                "uuid": uuid,
            }
        )
    trial_means = [
        statistics.fmean(latency_by_workload[index][trial] for index in range(WORKLOAD_COUNT))
        for trial in range(3)
    ]
    package_sha = sha256_bytes(build_package_bytes())
    summary = {
        "contract_sha256": CONTRACT_SHA256,
        "dataset_revision": DATASET_REVISION,
        "evaluation_stack": EVALUATION_STACK,
        "evaluator_revision": EVALUATOR_REVISION,
        "hardware": sorted(hardware),
        "max_relative_spread": rounded(max(spreads)),
        "passed_observations": WORKLOAD_COUNT * 3,
        "problem_id": PROBLEM_ID,
        "problem_name": PROBLEM_NAME,
        "recorded_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "schema_version": 1,
        "source_sha256": source_digest(),
        "submission_sha256": package_sha,
        "target": target,
        "trial_mean_latency_ms": [rounded(value) for value in trial_means],
        "trials": 3,
        "workloads": workload_summaries,
    }
    output = Path(arguments.output).resolve()
    atomic_write(output, canonical_json_bytes(summary))
    if arguments.promote:
        promote = Path(arguments.promote).resolve()
        if target != "b200" or promote != RESULTS_PATH:
            raise RepoError("only B200 summaries may be promoted to results/results.jsonl")
        for line in promote.read_text(encoding="utf-8").splitlines():
            if line:
                json.loads(line)
        with promote.open("a", encoding="utf-8") as handle:
            handle.write(f"{compact_json(summary)}\n")
        print("promoted B200 summary to results/results.jsonl")
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_info(_: argparse.Namespace) -> None:
    lock = validate_lock()
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        gpu = f"unavailable ({error})"
    evaluator = run_git("rev-parse", "HEAD", cwd=EVALUATOR_DIR)
    print(f"GPU, driver, compute capability: {gpu}")
    print(f"evaluator revision: {evaluator}")
    print(f"evaluator version: {lock['evaluator']['version']}")
    print(f"dataset revision: {lock['dataset']['revision']}")
    print(f"evaluation stack: {lock['evaluation_stack']}")
    print(f"contract SHA-256: {lock['contract']['sha256']}")
    print(f"dataset cache: {'verified' if DATA_PATH.exists() and sha256_file(DATA_PATH) == DATASET_SHA256 else 'not fetched'}")
    try:
        import torch

        print(f"container PyTorch: {torch.__version__}")
        print(f"container CUDA: {torch.version.cuda}")
    except ImportError:
        pass


def command_status(_: argparse.Namespace) -> None:
    request = urllib.request.Request(LEADERBOARD_URL, headers={"User-Agent": "sol-038-repo/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    data = payload.get("data") or {}
    if data.get("kernel_id") != PROBLEM_ID:
        raise RepoError("leaderboard response does not describe kernel 38")
    print(f"{data.get('kernel_name')} | {data.get('gpu_type')} | {data.get('evaluation_stack_version')}")
    print("rank  SOL score  latency ms  fast workloads  user")
    rankings = [row for row in data.get("rankings", []) if isinstance(row.get("rank"), int)]
    for row in rankings:
        fast = f"{row.get('fast_1_count', 0)}/{row.get('fast_1_total', 0)}"
        print(
            f"{row['rank']:>4}  {row.get('sol_score', 0):>9.6f}  "
            f"{row.get('latency_ms', 0):>10.6f}  {fast:>14}  {row.get('username', '')}"
        )
    print("https://research.nvidia.com/benchmarks/sol-execbench/kernel/38")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="fetch and extract the pinned problem")
    fetch.set_defaults(function=command_fetch)

    lint = subparsers.add_parser("lint", help="validate repository invariants")
    lint.set_defaults(function=command_lint)

    info = subparsers.add_parser("info", help="print environment and lock information")
    info.set_defaults(function=command_info)

    stage = subparsers.add_parser("stage", help="stage a source manifest for one target")
    stage.add_argument("--target", required=True, choices=("local", "b200"))
    stage.add_argument("--output", required=True)
    stage.set_defaults(function=command_stage)

    select = subparsers.add_parser("select-workload", help="extract one workload for profiling")
    select.add_argument("--index", type=int, required=True)
    select.add_argument("--problem", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(function=command_select_workload)

    summarize = subparsers.add_parser("summarize", help="summarize three official-harness traces")
    summarize.add_argument("--target", required=True, choices=("local", "b200"))
    summarize.add_argument("--output", required=True)
    summarize.add_argument("--promote")
    summarize.add_argument("traces", nargs=3)
    summarize.set_defaults(function=command_summarize)

    package = subparsers.add_parser("package", help="build deterministic B200 submission JSON")
    package.add_argument("--output", required=True)
    package.set_defaults(function=command_package)

    verify = subparsers.add_parser("verify-package", help="verify submission JSON and sidecar")
    verify.add_argument("--artifact", required=True)
    verify.set_defaults(function=command_verify_package)

    status = subparsers.add_parser("status", help="print the live kernel leaderboard")
    status.set_defaults(function=command_status)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        arguments.function(arguments)
    except (RepoError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
