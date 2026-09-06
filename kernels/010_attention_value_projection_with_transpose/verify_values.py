"""Verify changed inputs, output ownership and the concrete tensor contract."""

import datetime
import fcntl
import json
from pathlib import Path
import runpy

import torch

from kernel import run


def main():
    root = Path(__file__).resolve().parents[2]
    reference = runpy.run_path(str(root / ".work/problems/10/reference.py"))["run"]
    records = []
    with (root / ".work/gpu.lock").open("a+") as lock:
        print("Waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for seed in (41, 733, 8109):
            torch.manual_seed(seed)
            for batch, sequence in ((1, 128), (2, 128), (4, 541), (1, 1571), (1, 8192)):
                hidden = torch.randn(batch, sequence, 5120, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(1024, 5120, device="cuda", dtype=torch.bfloat16)
                previous_output = None
                previous_copy = None
                for revision in (0, 1):
                    if revision:
                        hidden.normal_(mean=0.02, std=0.7)
                        weight.normal_(mean=-0.01, std=0.8)
                    expected = reference(hidden, weight)
                    output = run(hidden, weight)
                    assert type(output) is torch.Tensor
                    assert output.shape == (batch, 8, sequence, 128)
                    assert output.dtype == torch.bfloat16
                    assert torch.equal(output, expected)
                    assert output.untyped_storage().data_ptr() != hidden.untyped_storage().data_ptr()
                    assert output.untyped_storage().data_ptr() != weight.untyped_storage().data_ptr()
                    if previous_output is not None:
                        assert torch.equal(previous_output, previous_copy)
                    records.append({"seed": seed, "batch": batch, "sequence": sequence,
                                    "input_revision": revision, "exact_match": True,
                                    "output_strides": list(output.stride()),
                                    "output_type": type(output).__name__})
                    previous_output, previous_copy = output, output.clone()
                print(json.dumps(records[-2:]), flush=True)
    path = root / ".work/validation/10"
    path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path / f"values-{stamp}.json"
    target.write_text(json.dumps({"passed": True, "cases": records}, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
