#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tune BF16 QK tiles with official input shifting and CUPTI timing."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor
from sol_execbench.core.bench.timing import time_runnable

from kernel import _qk_scores
from tma import _qk_tma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["1x128", "2x449", "2x4096"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep", type=int, default=15)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--tma-only", action="store_true")
    parser.add_argument("--bk", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--programs", nargs="+", type=int, default=[148, 296, 592])
    args = parser.parse_args()
    shapes = [tuple(map(int, shape.split("x"))) for shape in args.shapes]
    if any(len(shape) != 2 or min(shape) <= 0 for shape in shapes):
        parser.error("shapes must be positive BxS pairs")
    directory = Path(__file__).parent
    results = {
        "clock_mode": "unlocked",
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("kernel.py", "tma.py", "tune.py")
        },
        "records": [],
    }
    lock = Path(os.environ.get(
        "SOL_GPU_LOCK", "/workspace/sol-execbench-b200-kernels/.work/gpu.lock",
    ))
    lock.parent.mkdir(parents=True, exist_ok=True)
    plain = [
        (32, 64, 128, 4), (64, 64, 128, 4),
        (64, 128, 128, 4), (128, 64, 128, 4),
        (128, 128, 128, 4), (128, 128, 128, 8),
        (128, 256, 128, 8), (64, 256, 128, 8),
        (64, 128, 64, 4), (128, 128, 64, 8),
    ]
    tma = [
        (64, 128, args.bk, 4), (128, 128, args.bk, 4),
        (128, 256, args.bk, 8),
    ]
    configs = [("plain", *tile, 0, False) for tile in plain]
    configs += [("plain", *tile, 0, True) for tile in plain[2:8]]
    configs += [
        ("tma", *tile, programs, True)
        for tile in tma for programs in args.programs
    ]
    if args.tma_only:
        configs = [config for config in configs if config[0] == "tma"]
    print(f"Waiting for GPU lock: {lock}", flush=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        print("GPU lock acquired", flush=True)
        torch.manual_seed(31457)
        for batch, seq in shapes:
            q = torch.randn((batch, 32, seq, 128), dtype=torch.bfloat16, device="cuda")
            k = torch.randn_like(q)
            out = torch.empty((batch, 32, seq, seq), dtype=torch.bfloat16, device="cuda")
            reference = (q.float() @ k.float().transpose(-1, -2) * (128 ** -0.5)).to(torch.bfloat16)
            for kind, bm, bn, bk, warps, programs, stream in configs:
                grid = (triton.cdiv(seq, bm), triton.cdiv(seq, bn), batch * 32)
                programs = min(programs, grid[0] * grid[1] * grid[2])

                def launch(a, b, o):
                    if kind == "plain":
                        _qk_scores[grid](
                            a, b, o, seq, bm, bn, bk, stream,
                            num_warps=warps, num_stages=args.stages,
                        )
                    else:
                        aq = TensorDescriptor(a, [batch * 32 * seq, 128], [128, 1], [bm, bk])
                        bk_desc = TensorDescriptor(b, [batch * 32 * seq, 128], [128, 1], [bn, bk])
                        _qk_tma[(programs,)](
                            aq, bk_desc, o, seq, batch * 32, bm, bn, bk,
                            programs, stream,
                            num_warps=warps, num_stages=args.stages,
                        )

                record = {
                    "batch": batch, "seq": seq, "kind": kind,
                    "bm": bm, "bn": bn, "bk": bk,
                    "warps": warps, "programs": programs, "stream": stream,
                    "stages": args.stages,
                }
                try:
                    out.fill_(float("nan"))
                    launch(q, k, out)
                    torch.testing.assert_close(out, reference, atol=1e-5, rtol=0.05)
                    record["latency_ms"] = time_runnable(
                        launch, [q, k], [out], "cuda", warmup=3, rep=args.rep,
                    )
                    record["passed"] = True
                except Exception as error:
                    record["passed"] = False
                    record["error"] = f"{type(error).__name__}: {error}"
                record["time"] = time.time()
                results["records"].append(record)
                print(json.dumps(record), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
            del q, k, out, reference


if __name__ == "__main__":
    main()
