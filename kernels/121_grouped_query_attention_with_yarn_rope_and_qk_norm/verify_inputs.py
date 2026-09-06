"""Check attention semantics with changed input values and reused tensor storage."""

import datetime
import fcntl
import json
from pathlib import Path
import runpy

import torch
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.data.workload import ToleranceSpec

from kernel import run


def main():
    root = Path(__file__).resolve().parents[2]
    reference = runpy.run_path(str(root / ".work/problems/121/reference.py"))
    results = []
    tolerance = ToleranceSpec(max_atol=0.01, max_rtol=0.05, required_matched_ratio=0.99)
    with (root / ".work/gpu.lock").open("a+") as lock:
        print("Waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.manual_seed(713)
        values = reference["get_inputs"]({"batch_size": 2, "seq_len": 131}, torch.device("cuda"))

        def check(name):
            expected = reference["run"](**values)
            output = torch.full_like(expected, float("nan"))
            run(*values.values(), output)
            torch.cuda.synchronize()
            stats, exceeds = compute_error_stats(output, expected, tolerance)
            delta = (output.float() - expected.float()).abs()
            ratio = (delta <= tolerance.max_atol + tolerance.max_rtol * expected.float().abs()).float().mean().item()
            record = {"case": name, "passed": not exceeds, "matched_ratio": ratio,
                      "errors": stats.model_dump()}
            results.append(record)
            print(json.dumps(record), flush=True)
            if exceeds:
                raise AssertionError(f"{name} failed numerical validation")

        check("original values")
        values["q_norm_weight"].copy_(1 + 0.2 * torch.randn_like(values["q_norm_weight"]))
        values["k_norm_weight"].copy_(1 + 0.2 * torch.randn_like(values["k_norm_weight"]))
        check("nonuniform normalization weights")

        values["attention_mask"] = 0.2 * torch.randn((2, 1, 131, 131), dtype=torch.bfloat16, device="cuda")
        check("arbitrary finite additive mask")

        values["position_ids"] = torch.randint(-512, 7000, (2, 131), dtype=torch.int64, device="cuda")
        values["inv_freq"].mul_(1.07)
        values["attention_factor"] = 0.93
        values["scaling"] = 0.07
        values["rms_norm_eps"] = 0.001
        check("changed positions frequencies and scalar inputs")

        values["hidden_states"].normal_()
        values["q_proj_weight"].normal_(std=0.02)
        values["attention_mask"].normal_(std=0.3)
        check("fresh values in reused storage")

    directory = root / ".work/validation/121"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"inputs-{stamp}.json"
    path.write_text(json.dumps({"tolerance": tolerance.model_dump(), "cases": results}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
