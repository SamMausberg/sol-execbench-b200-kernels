"""Check changed values through the exact packaged cuBLASLt core/configurations."""

import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy

import torch
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.data.workload import ToleranceSpec


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[3]
    spec = importlib.util.spec_from_file_location("cublaslt_loader", directory.parent / "cublaslt_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    bridge = loader.load_bridge()
    assert bridge.source_sha256 == hashlib.sha256((directory / "cublaslt_bridge.h").read_bytes()).hexdigest()
    selected = {r["m"]: r for r in json.loads((directory / "selected-algorithms.json").read_text())["records"]}
    reference = runpy.run_path(str(root / ".work/problems/10/reference.py"))["run"]
    tolerance = ToleranceSpec(max_atol=1e-5, max_rtol=0.05, required_matched_ratio=0.99)
    records = []
    with Path(os.environ.get("SOL_GPU_LOCK", str(root / ".work/gpu.lock"))).open("a+") as lock:
        print("Waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for seed in (41, 733, 8109):
            torch.manual_seed(seed)
            for batch, sequence in ((1, 128), (2, 128), (4, 541), (1, 1571), (1, 8192)):
                rows = batch * sequence
                choice = selected[rows]
                hidden = torch.randn(batch, sequence, 5120, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(1024, 5120, device="cuda", dtype=torch.bfloat16)
                previous, previous_copy = None, None
                for revision in (0, 1):
                    if revision:
                        hidden.normal_(mean=0.02, std=0.7)
                        weight.normal_(mean=-0.01, std=0.8)
                    expected = reference(hidden, weight)
                    projected = torch.full((rows, 1024), float("nan"), device="cuda", dtype=torch.bfloat16)
                    workspace = torch.empty(choice["workspace_bytes"], device="cuda", dtype=torch.uint8)
                    bridge.matmul_config(hidden.view(rows, 5120), weight, projected,
                                         workspace, choice["configuration"])
                    output = projected.view(batch, sequence, 8, 128).transpose(1, 2)
                    stats, exceeds = compute_error_stats(output, expected, tolerance)
                    assert not exceeds, (seed, batch, sequence, revision, stats)
                    assert type(output) is torch.Tensor and output.dtype == expected.dtype
                    assert output.shape == expected.shape
                    assert output.untyped_storage().data_ptr() not in (
                        hidden.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr())
                    if previous is not None:
                        assert torch.equal(previous, previous_copy)
                    delta = (output.float() - expected.float()).abs()
                    matched = (delta <= tolerance.max_atol + tolerance.max_rtol * expected.float().abs()).float().mean().item()
                    records.append({"seed": seed, "batch": batch, "sequence": sequence,
                                    "input_revision": revision, "passed": True,
                                    "matched_ratio": matched, "errors": stats.model_dump(),
                                    "output_strides": list(output.stride())})
                    previous, previous_copy = output, output.clone()
                print(json.dumps(records[-2:]), flush=True)
    target = root / ".work/validation/10"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"cublaslt-v2-values-{stamp}.json"
    path.write_text(json.dumps({"passed": True, "tolerance": tolerance.model_dump(),
                               "bridge_source_sha256": bridge.source_sha256,
                               "cublaslt_version": bridge.cublaslt_version,
                               "cases": records}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
