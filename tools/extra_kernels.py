#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 sol-038-flux-rmsnorm contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce, stage, benchmark, and package the additional SOL kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "kernels" / "benchmark.lock.json"
WORK_ROOT = ROOT / ".work" / "extra-kernels"
DATA_ROOT = ROOT / ".work" / "data"
SUBMISSION_ROOT = WORK_ROOT / "submissions"

ALLOWED_LANGUAGES = {
    "cuda_cpp",
    "cublas",
    "cudnn",
    "cutlass",
    "cute_dsl",
    "pytorch",
    "triton",
}
ALLOWED_DEPENDENCIES = {"cublas", "cudnn", "cutlass", "torch", "triton"}
CPP_LANGUAGES = {"cuda_cpp", "cublas", "cudnn", "cutlass"}
PYTHON_LANGUAGES = {"cute_dsl", "pytorch", "triton"}

# Shape dispatch, persistent scratch, and PyTorch's current CUDA stream are all
# legitimate. These fragments are narrowly limited to APIs that inspect or
# interfere with evaluation. NVIDIA's runtime reward-hack checks remain the
# authoritative gate; this audit catches obvious source regressions earlier.
FORBIDDEN_SOURCE_FRAGMENTS = {
    "benchmark or evaluator timing": (
        "cudaeventcreate",
        "cudaeventrecord",
        "cudaeventelapsedtime",
        "clock_gettime",
        "std::chrono",
        "time.perf_counter",
        "torch.cuda.event",
    ),
    "non-current CUDA stream or graph": (
        "cudastreamcreate",
        "getdefaultcudastream",
        "getstreamfrompool",
        "cudagraph",
    ),
    "thread injection": ("std::thread", "pthread_create", "import threading"),
    "process, network, or filesystem access": (
        "std::getenv",
        "getenv(",
        "std::system",
        "system(",
        "popen(",
        "subprocess",
        "urllib",
        "requests.",
        "socket(",
        "fopen(",
        "ifstream",
        "/proc/",
    ),
    "deliberate delay": ("sleep(", "usleep("),
    "Python runtime mutation": ("monkeypatch", "setattr(torch"),
    "profiler access": ("cupti", "nvtxrange"),
    "persistent numeric workspace": (
        "thread_local at::tensor",
        "thread_local torch::tensor",
    ),
}


class RepoError(RuntimeError):
    """An expected validation or packaging failure."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def ordered_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def ordered_compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


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


def read_workloads(path: Path) -> list[dict[str, Any]]:
    workloads: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RepoError(f"workload line {line_number} is not an object")
        workloads.append(value)
    return workloads


def load_lock() -> dict[str, Any]:
    lock = load_json(LOCK_PATH)
    if lock.get("schema_version") != 1:
        raise RepoError("unsupported additional-kernel lock schema")
    if lock.get("evaluation_stack") != "v1.1":
        raise RepoError("additional kernels must use evaluation stack v1.1")
    evaluator = lock.get("evaluator") or {}
    if evaluator.get("revision") != "a9fa0804c793d438e70850c33fe34426e66d53dd":
        raise RepoError("evaluator revision is not pinned to the repository submodule")
    return lock


def select_ids(arguments: argparse.Namespace, lock: dict[str, Any]) -> list[str]:
    raw = getattr(arguments, "ids", None) or list(lock["problems"])
    ids = [str(item) for item in raw]
    unknown = sorted(set(ids) - set(lock["problems"]))
    if unknown:
        raise RepoError(f"unknown problem id(s): {', '.join(unknown)}")
    return ids


def problem_config(lock: dict[str, Any], problem_id: str) -> dict[str, Any]:
    return lock["problems"][problem_id]


def kernel_dir(config: dict[str, Any]) -> Path:
    return ROOT / "kernels" / config["directory"]


def variant_dir(config: dict[str, Any], variant: str | None = None) -> Path:
    directory = kernel_dir(config)
    if variant is None:
        return directory
    variants_root = (directory / "variants").resolve()
    candidate = (variants_root / variant).resolve()
    if variants_root not in candidate.parents or not candidate.is_dir():
        raise RepoError(f"unknown source variant: {variant}")
    return candidate


def problem_dir(problem_id: str) -> Path:
    return WORK_ROOT / "problems" / problem_id


def contract_digest(definition: dict[str, Any], workloads: list[Any]) -> str:
    payload = ordered_compact_json({"definition": definition, "workloads": workloads})
    return sha256_bytes(payload.encode("utf-8"))


def verify_contract(problem_id: str, config: dict[str, Any]) -> None:
    directory = problem_dir(problem_id)
    definition_path = directory / "definition.json"
    workload_path = directory / "workload.jsonl"
    if not definition_path.exists() or not workload_path.exists():
        raise RepoError(f"problem {problem_id} is not extracted; run fetch")
    definition = load_json(definition_path)
    workloads = read_workloads(workload_path)
    if definition.get("name") != config["name"]:
        raise RepoError(f"problem {problem_id} definition name mismatch")
    if len(workloads) != config["workloads"]:
        raise RepoError(
            f"problem {problem_id} expected {config['workloads']} workloads, "
            f"found {len(workloads)}"
        )
    actual = contract_digest(definition, workloads)
    if actual != config["contract_sha256"]:
        raise RepoError(
            f"problem {problem_id} contract mismatch: expected "
            f"{config['contract_sha256']}, found {actual}"
        )


def fetch_dataset(lock: dict[str, Any], config: dict[str, Any]) -> Path:
    collection = config["collection"]
    destination = DATA_ROOT / f"{collection}.parquet"
    expected = config["dataset_sha256"]
    if destination.exists() and sha256_file(destination) == expected:
        return destination
    revision = lock["dataset"]["revision"]
    url = (
        "https://huggingface.co/datasets/nvidia/SOL-ExecBench/resolve/"
        f"{revision}/data/{collection}.parquet?download=true"
    )
    print(f"fetching {collection} from pinned dataset revision")
    request = urllib.request.Request(url, headers={"User-Agent": "sol-extra-kernels/1"})
    temporary = destination.with_suffix(".parquet.download")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual = sha256_file(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise RepoError(f"dataset SHA-256 mismatch: expected {expected}, found {actual}")
    os.replace(temporary, destination)
    return destination


def command_fetch(arguments: argparse.Namespace) -> None:
    lock = load_lock()
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RepoError("PyArrow is required; run fetch through the evaluator image") from error

    for problem_id in select_ids(arguments, lock):
        config = problem_config(lock, problem_id)
        dataset = fetch_dataset(lock, config)
        table = parquet.read_table(dataset)
        matches = [row for row in table.to_pylist() if row.get("name") == config["name"]]
        if len(matches) != 1:
            raise RepoError(
                f"expected one {config['name']} row in {config['collection']}, "
                f"found {len(matches)}"
            )
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
        actual = contract_digest(definition, workloads)
        if actual != config["contract_sha256"]:
            raise RepoError(
                f"problem {problem_id} extracted contract mismatch: "
                f"expected {config['contract_sha256']}, found {actual}"
            )
        output = problem_dir(problem_id)
        atomic_write(output / "definition.json", ordered_json_bytes(definition))
        workload_bytes = "".join(
            f"{ordered_compact_json(workload)}\n" for workload in workloads
        ).encode("utf-8")
        atomic_write(output / "workload.jsonl", workload_bytes)
        atomic_write(output / "contract.sha256", f"{actual}\n".encode("ascii"))
        verify_contract(problem_id, config)
        print(f"problem {problem_id}: {len(workloads)} workloads, contract {actual}")


def load_source_manifest(
    config: dict[str, Any], variant: str | None = None
) -> dict[str, Any]:
    directory = variant_dir(config, variant)
    manifest_path = directory / "solution.json"
    manifest = load_json(manifest_path)
    if manifest_path.read_bytes() != canonical_json_bytes(manifest):
        raise RepoError(f"{manifest_path.relative_to(ROOT)} is not canonical JSON")
    allowed_top_level = {"author", "definition", "description", "name", "sources", "spec"}
    if not set(manifest).issubset(allowed_top_level):
        unknown = ", ".join(sorted(set(manifest) - allowed_top_level))
        raise RepoError(f"unsupported solution field(s): {unknown}")
    if manifest.get("definition") != config["name"]:
        raise RepoError(f"solution definition does not match {config['name']}")
    if not manifest.get("name") or not manifest.get("author"):
        raise RepoError("solution name and author must be nonempty")
    spec = manifest.get("spec")
    if not isinstance(spec, dict):
        raise RepoError("solution spec must be an object")
    expected_spec_fields = {
        "binding",
        "compile_options",
        "dependencies",
        "destination_passing_style",
        "entry_point",
        "languages",
        "target_hardware",
    }
    if set(spec) != expected_spec_fields:
        missing = sorted(expected_spec_fields - set(spec))
        extra = sorted(set(spec) - expected_spec_fields)
        raise RepoError(f"solution spec fields mismatch (missing={missing}, extra={extra})")
    languages = spec.get("languages")
    if (
        not isinstance(languages, list)
        or not languages
        or any(language not in ALLOWED_LANGUAGES for language in languages)
    ):
        raise RepoError("solution languages are missing or unsupported")
    uses_cpp = any(language in CPP_LANGUAGES for language in languages)
    uses_python = any(language in PYTHON_LANGUAGES for language in languages)
    if uses_cpp and uses_python:
        raise RepoError("C++ and Python solution languages cannot be mixed")
    expected_binding = "torch" if uses_cpp else None
    if spec.get("binding") != expected_binding:
        raise RepoError(
            f"solution binding must be {expected_binding!r} for its languages"
        )
    dependencies = spec.get("dependencies")
    if (
        not isinstance(dependencies, list)
        or any(dependency not in ALLOWED_DEPENDENCIES for dependency in dependencies)
    ):
        raise RepoError("solution dependencies are malformed or unsupported")
    if not isinstance(spec.get("destination_passing_style"), bool):
        raise RepoError("destination_passing_style must be a boolean")
    entry_point = spec.get("entry_point")
    if (
        not isinstance(entry_point, str)
        or entry_point.count("::") != 1
        or not all(entry_point.split("::", 1))
    ):
        raise RepoError("solution entry point must be <source>::<symbol>")
    compile_options = spec.get("compile_options")
    expected_compile_fields = {"cflags", "cuda_cflags", "ld_flags"} if uses_cpp else set()
    if not isinstance(compile_options, dict) or set(compile_options) != expected_compile_fields:
        raise RepoError(
            f"compile_options fields must be {sorted(expected_compile_fields)}"
        )
    for option_name, values in compile_options.items():
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise RepoError(f"compile option {option_name} must be a list of strings")
    targets = spec.get("target_hardware")
    if not isinstance(targets, list) or "B200" not in targets:
        raise RepoError("source solution must include B200 target hardware")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RepoError("solution sources must be a nonempty list")
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"path"}:
            raise RepoError("source manifest entries must contain only path")
        source_path = Path(source["path"])
        if source_path.is_absolute() or ".." in source_path.parts:
            raise RepoError(f"unsafe source path: {source_path}")
        if source_path.as_posix() in seen:
            raise RepoError(f"duplicate source path: {source_path}")
        seen.add(source_path.as_posix())
        local_path = directory / source_path
        if local_path.is_symlink() or not local_path.is_file():
            raise RepoError(f"missing kernel source: {local_path.relative_to(ROOT)}")
        lowered = local_path.read_text(encoding="utf-8").lower()
        for category, fragments in FORBIDDEN_SOURCE_FRAGMENTS.items():
            for fragment in fragments:
                if fragment in lowered:
                    raise RepoError(
                        f"forbidden {category} API {fragment!r} in "
                        f"{local_path.relative_to(ROOT)}"
                    )
    entry_file = str(spec.get("entry_point", "")).split("::", 1)[0]
    if entry_file not in seen:
        raise RepoError("entry point file is not listed in sources")
    return manifest


def build_artifact(
    config: dict[str, Any],
    target: str,
    variant: str | None = None,
    extra_cuda_cflags: list[str] | None = None,
    cute_policy: str | None = None,
    cute_static_max_clusters: int | None = None,
    cute_execution_policy: str | None = None,
) -> bytes:
    manifest = load_source_manifest(config, variant)
    packaged = json.loads(json.dumps(manifest))
    packaged["spec"]["target_hardware"] = [target]
    languages = set(packaged["spec"]["languages"])
    if (
        config["directory"] == "179_fp8_mlp_gate_up_projection"
        and languages & CPP_LANGUAGES
    ):
        architecture_define = (
            "-DSOL_TARGET_SM120=1" if target == "LOCAL" else "-DSOL_TARGET_SM100=1"
        )
        packaged["spec"]["compile_options"]["cuda_cflags"].append(
            architecture_define
        )
        if target == "LOCAL":
            # CUTLASS blockwise FP8 MMA is an architecture-conditional SM120
            # feature.  The generic sm_120 target compiles its fallback assert;
            # sm_120a enables the feature set used by the official 87b example.
            packaged["spec"]["compile_options"]["cuda_cflags"].append(
                "-gencode=arch=compute_120a,code=sm_120a"
            )
    if extra_cuda_cflags:
        if not languages & CPP_LANGUAGES:
            raise RepoError("extra CUDA flags require a C++/CUDA source variant")
        packaged["spec"]["compile_options"]["cuda_cflags"].extend(
            extra_cuda_cflags
        )
    directory = variant_dir(config, variant)
    for source in packaged["sources"]:
        content = (directory / source["path"]).read_text(encoding="utf-8")
        if cute_policy is not None:
            if (
                config["directory"] != "179_fp8_mlp_gate_up_projection"
                or variant is not None
                or source["path"] != "kernel.py"
            ):
                raise RepoError(
                    "CuTe schedule policies require the canonical problem 179 source"
                )
            needle = '_SCHEDULE_POLICY = "1cta_n4"'
            replacement = f'_SCHEDULE_POLICY = "{cute_policy}"'
            if content.count(needle) != 1:
                raise RepoError("problem 179 CuTe policy sentinel is missing")
            content = content.replace(needle, replacement)
            packaged["name"] = f"{packaged['name']}_{cute_policy}"
        if cute_static_max_clusters is not None:
            if (
                config["directory"] != "179_fp8_mlp_gate_up_projection"
                or variant is not None
                or source["path"] != "kernel.py"
            ):
                raise RepoError(
                    "a static CuTe cluster count requires canonical problem 179"
                )
            if cute_static_max_clusters <= 0:
                raise RepoError("the static CuTe cluster count must be positive")
            needle = (
                "value = cutlass_utils.HardwareInfo().get_max_active_clusters("
                "cluster_size)"
            )
            replacement = f"value = {cute_static_max_clusters}"
            if content.count(needle) != 1:
                raise RepoError("problem 179 occupancy-query sentinel is missing")
            content = content.replace(needle, replacement)
            packaged["name"] = (
                f"{packaged['name']}_compilecheck_clusters"
                f"{cute_static_max_clusters}"
            )
        if cute_execution_policy is not None:
            if (
                config["directory"] != "179_fp8_mlp_gate_up_projection"
                or variant is not None
                or source["path"] != "kernel.py"
            ):
                raise RepoError(
                    "a CuTe execution policy requires canonical problem 179"
                )
            needle = '_EXECUTION_POLICY = "separate"'
            replacement = f'_EXECUTION_POLICY = "{cute_execution_policy}"'
            if content.count(needle) != 1:
                raise RepoError("problem 179 execution-policy sentinel is missing")
            content = content.replace(needle, replacement)
            packaged["name"] = f"{packaged['name']}_{cute_execution_policy}"
        source["content"] = content
    return canonical_json_bytes(packaged)


def command_lint(arguments: argparse.Namespace) -> None:
    lock = load_lock()
    if LOCK_PATH.read_bytes() != canonical_json_bytes(lock):
        raise RepoError("kernels/benchmark.lock.json is not canonical JSON")
    for problem_id in select_ids(arguments, lock):
        config = problem_config(lock, problem_id)
        load_source_manifest(config)
        if problem_dir(problem_id).exists():
            verify_contract(problem_id, config)
        print(f"problem {problem_id}: source manifest and files are valid")


def command_stage(arguments: argparse.Namespace) -> None:
    lock = load_lock()
    config = problem_config(lock, str(arguments.problem_id))
    hardware = "LOCAL" if arguments.target == "local" else "B200"
    output = Path(arguments.output).resolve()
    if ROOT not in output.parents:
        raise RepoError("staged artifacts must remain inside the repository")
    data = build_artifact(
        config,
        hardware,
        arguments.variant,
        arguments.cuda_cflag,
        arguments.cute_policy,
        arguments.cute_static_max_clusters,
        arguments.cute_execution_policy,
    )
    atomic_write(output, data)
    print(f"staged {hardware} artifact: {output.relative_to(ROOT)}")
    print(f"SHA-256: {sha256_bytes(data)}")


def command_compile(arguments: argparse.Namespace) -> None:
    """Compile an embedded artifact through the pinned official packager only."""
    try:
        from sol_execbench.core import BenchmarkConfig, Definition, Solution, Workload
        from sol_execbench.driver import ProblemPackager
    except ImportError as error:
        raise RepoError("compile must run inside the pinned evaluator image") from error

    lock = load_lock()
    problem_id = str(arguments.problem_id)
    config = problem_config(lock, problem_id)
    verify_contract(problem_id, config)
    hardware = "LOCAL" if arguments.target == "local" else "B200"
    solution = Solution(
        **json.loads(
            build_artifact(
                config,
                hardware,
                arguments.variant,
                arguments.cuda_cflag,
                arguments.cute_policy,
                arguments.cute_static_max_clusters,
                arguments.cute_execution_policy,
            )
        )
    )
    definition = Definition(**load_json(problem_dir(problem_id) / "definition.json"))
    workloads = read_workloads(problem_dir(problem_id) / "workload.jsonl")
    workload = Workload(**workloads[0])

    if not any(language.value in CPP_LANGUAGES for language in solution.spec.languages):
        print(
            f"problem {problem_id}: official {hardware} Python/CuTe DSL "
            "schema validation passed (device code compiles JIT on the target)"
        )
        return

    build_root = WORK_ROOT / "compile"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{problem_id}-{arguments.target}-", dir=build_root
    ) as temporary:
        output_dir = Path(temporary)
        packager = ProblemPackager(
            definition=definition,
            workloads=[workload],
            solution=solution,
            config=BenchmarkConfig(),
            output_dir=output_dir,
            keep_output_dir=True,
        )
        command, artifact_path = packager.compile()
        environment = {
            **os.environ,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        result = subprocess.run(
            command,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=arguments.timeout,
            env=environment,
        )
        log_text = (
            f"command: {' '.join(command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )
        if result.returncode != 0:
            log_path = build_root / f"{problem_id}-{arguments.target}.log"
            atomic_write(log_path, log_text.encode("utf-8"))
            raise RepoError(
                f"problem {problem_id} {hardware} compile failed; see "
                f"{log_path.relative_to(ROOT)}"
            )
        artifact = Path(artifact_path)
        if not artifact.is_file():
            raise RepoError("official packager did not produce benchmark_kernel.so")
        cubin_listing = subprocess.run(
            ["cuobjdump", "--list-elf", str(artifact)],
            capture_output=True,
            text=True,
            check=True,
        )
        log_text += f"cuobjdump:\n{cubin_listing.stdout}\n{cubin_listing.stderr}\n"
        resource_listing = subprocess.run(
            ["cuobjdump", "--dump-resource-usage", str(artifact)],
            capture_output=True,
            text=True,
            check=True,
        )
        log_text += (
            "resource usage:\n"
            f"{resource_listing.stdout}\n{resource_listing.stderr}\n"
        )
        log_path = build_root / f"{problem_id}-{arguments.target}.log"
        atomic_write(log_path, log_text.encode("utf-8"))
        expected_arch = "sm_120" if arguments.target == "local" else "sm_100"
        if expected_arch not in cubin_listing.stdout:
            raise RepoError(
                f"compiled artifact does not contain expected {expected_arch} code"
            )
        print(
            f"problem {problem_id}: official {hardware} compile passed "
            f"({sha256_file(artifact)})"
        )
        print(f"compile log: {log_path.relative_to(ROOT)}")


def write_package(config: dict[str, Any]) -> tuple[Path, str]:
    output = SUBMISSION_ROOT / config["artifact"]
    data = build_artifact(config, "B200")
    digest = sha256_bytes(data)
    atomic_write(output, data)
    atomic_write(
        output.with_suffix(output.suffix + ".sha256"),
        f"{digest}  {output.name}\n".encode("ascii"),
    )
    return output, digest


def command_package(arguments: argparse.Namespace) -> None:
    lock = load_lock()
    for problem_id in select_ids(arguments, lock):
        output, digest = write_package(problem_config(lock, problem_id))
        print(f"problem {problem_id}: {output.relative_to(ROOT)}")
        print(f"SHA-256: {digest}")


def command_verify_package(arguments: argparse.Namespace) -> None:
    lock = load_lock()
    for problem_id in select_ids(arguments, lock):
        config = problem_config(lock, problem_id)
        artifact = SUBMISSION_ROOT / config["artifact"]
        expected = build_artifact(config, "B200")
        actual = artifact.read_bytes()
        if actual != expected:
            raise RepoError(f"problem {problem_id} package is missing or stale")
        manifest = json.loads(actual)
        if manifest["spec"]["target_hardware"] != ["B200"]:
            raise RepoError(f"problem {problem_id} package is not B200-only")
        digest = sha256_bytes(actual)
        sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
        expected_sidecar = f"{digest}  {artifact.name}\n"
        if not sidecar.exists() or sidecar.read_text(encoding="ascii") != expected_sidecar:
            raise RepoError(f"problem {problem_id} SHA-256 sidecar is stale")
        if canonical_json_bytes(manifest) != actual:
            raise RepoError(f"problem {problem_id} package is not canonical JSON")
        print(f"problem {problem_id}: package verified ({digest})")


def command_select_workload(arguments: argparse.Namespace) -> None:
    lock = load_lock()
    problem_id = str(arguments.problem_id)
    config = problem_config(lock, problem_id)
    verify_contract(problem_id, config)
    source = problem_dir(problem_id)
    workloads = read_workloads(source / "workload.jsonl")
    if arguments.index < 0 or arguments.index >= len(workloads):
        raise RepoError(f"workload index must be between 0 and {len(workloads) - 1}")
    output = Path(arguments.output).resolve()
    if ROOT not in output.parents:
        raise RepoError("selected workload directory must remain inside the repository")
    atomic_write(output / "definition.json", (source / "definition.json").read_bytes())
    atomic_write(
        output / "workload.jsonl",
        f"{ordered_compact_json(workloads[arguments.index])}\n".encode("utf-8"),
    )
    print(f"problem {problem_id}: selected workload {arguments.index}")


def command_summarize(arguments: argparse.Namespace) -> None:
    lock = load_lock()
    problem_id = str(arguments.problem_id)
    config = problem_config(lock, problem_id)
    trace_paths = [Path(item) for item in arguments.traces]
    if len(trace_paths) != 3:
        raise RepoError("exactly three trace files are required")
    trials = [read_workloads(path) for path in trace_paths]
    workload_count = config["workloads"]
    if any(len(trial) != workload_count for trial in trials):
        raise RepoError(f"each trace must contain {workload_count} workloads")
    workload_latencies: list[list[float]] = [[] for _ in range(workload_count)]
    hardware: set[str] = set()
    for trial_index, trial in enumerate(trials):
        for workload_index, trace in enumerate(trial):
            evaluation = trace.get("evaluation") or {}
            if evaluation.get("status") != "PASSED":
                raise RepoError(
                    f"trial {trial_index + 1}, workload {workload_index} did not pass: "
                    f"{evaluation.get('status')}"
                )
            environment = evaluation.get("environment") or {}
            hardware.add(str(environment.get("hardware", "unknown")))
            latency = (evaluation.get("performance") or {}).get("latency_ms")
            if not isinstance(latency, (float, int)) or latency <= 0:
                raise RepoError("passed trace is missing a positive latency")
            workload_latencies[workload_index].append(float(latency))
    trial_means = [
        statistics.fmean(workload_latencies[index][trial] for index in range(workload_count))
        for trial in range(3)
    ]
    output = {
        "contract_sha256": config["contract_sha256"],
        "hardware": sorted(hardware),
        "problem_id": int(problem_id),
        "problem_name": config["name"],
        "trial_mean_latency_ms": [round(value, 9) for value in trial_means],
        "workloads": [
            {
                "latency_ms_mean": round(statistics.fmean(values), 9),
                "latency_ms_median": round(statistics.median(values), 9),
            }
            for values in workload_latencies
        ],
    }
    destination = Path(arguments.output).resolve()
    atomic_write(destination, canonical_json_bytes(output))
    print(json.dumps(output, indent=2, sort_keys=True))


def command_status(arguments: argparse.Namespace) -> None:
    lock = load_lock()
    stack = lock["evaluation_stack"]
    for problem_id in select_ids(arguments, lock):
        url = (
            "https://research.nvidia.com/benchmarks/sol-execbench/api/leaderboard/"
            f"kernel/{problem_id}/B200?evaluation_stack_version={stack}"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "sol-extra-kernels/1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = (json.load(response).get("data") or {})
        if data.get("kernel_id") != int(problem_id):
            raise RepoError(f"leaderboard response does not describe problem {problem_id}")
        print(f"{problem_id} {data.get('kernel_name')} | {stack}")
        print("rank  SOL score  latency ms  fast workloads  user")
        for row in data.get("rankings", []):
            if not isinstance(row.get("rank"), int):
                continue
            fast = f"{row.get('fast_1_count', 0)}/{row.get('fast_1_total', 0)}"
            print(
                f"{row['rank']:>4}  {row.get('sol_score', 0):>9.6f}  "
                f"{row.get('latency_ms', 0):>10.6f}  {fast:>14}  "
                f"{row.get('username', '')}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, function, help_text in (
        ("fetch", command_fetch, "fetch and extract pinned problem contracts"),
        ("lint", command_lint, "validate source manifests and contracts"),
        ("package", command_package, "build deterministic B200 JSON packages"),
        ("verify-package", command_verify_package, "verify B200 JSON packages"),
        ("status", command_status, "print current B200 leaderboards"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("ids", nargs="*", choices=("29", "179"))
        command.set_defaults(function=function)

    stage = subparsers.add_parser("stage", help="build one self-contained test artifact")
    stage.add_argument("problem_id", choices=("29", "179"))
    stage.add_argument("--target", required=True, choices=("local", "b200"))
    stage.add_argument("--output", required=True)
    stage.add_argument("--variant")
    stage.add_argument("--cuda-cflag", action="append", default=[])
    stage.add_argument(
        "--cute-policy",
        choices=(
            "1cta_n4",
            "2cta_256_even_n4",
            "2cta_128_even_n4",
            "2cta_256_all_n4",
        ),
    )
    stage.add_argument(
        "--cute-static-max-clusters",
        type=int,
        help="replace B200 occupancy discovery for a local compile-only check",
    )
    stage.add_argument(
        "--cute-execution-policy",
        choices=("separate", "dual", "dual_cute_silu"),
    )
    stage.set_defaults(function=command_stage)

    compile_command = subparsers.add_parser(
        "compile", help="compile one artifact with the pinned official packager"
    )
    compile_command.add_argument("problem_id", choices=("29", "179"))
    compile_command.add_argument("--target", required=True, choices=("local", "b200"))
    compile_command.add_argument("--timeout", type=int, default=1200)
    compile_command.add_argument("--variant")
    compile_command.add_argument("--cuda-cflag", action="append", default=[])
    compile_command.add_argument(
        "--cute-policy",
        choices=(
            "1cta_n4",
            "2cta_256_even_n4",
            "2cta_128_even_n4",
            "2cta_256_all_n4",
        ),
    )
    compile_command.add_argument(
        "--cute-static-max-clusters",
        type=int,
        help="replace B200 occupancy discovery for a local compile-only check",
    )
    compile_command.add_argument(
        "--cute-execution-policy",
        choices=("separate", "dual", "dual_cute_silu"),
    )
    compile_command.set_defaults(function=command_compile)

    select = subparsers.add_parser("select-workload", help="extract one workload")
    select.add_argument("problem_id", choices=("29", "179"))
    select.add_argument("--index", type=int, required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(function=command_select_workload)

    summarize = subparsers.add_parser("summarize", help="summarize three harness trials")
    summarize.add_argument("problem_id", choices=("29", "179"))
    summarize.add_argument("--output", required=True)
    summarize.add_argument("traces", nargs=3)
    summarize.set_defaults(function=command_summarize)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        arguments.function(arguments)
    except (
        RepoError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
