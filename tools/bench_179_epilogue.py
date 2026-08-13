#!/usr/bin/env python3
"""Microbenchmark the SOL-ExecBench 179 SwiGLU epilogue on a local GPU."""

from __future__ import annotations

import argparse
import statistics

import torch
import triton
import triton.language as tl


N = 18944
M_VALUES = (384, 512, 768, 1152, 1280, 1792, 2048, 2304, 2816, 3072,
            3584, 3840, 5120, 5632, 6144, 8704)


@triton.jit
def swiglu(
    gate_ptr,
    up_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    block: tl.constexpr,
    gate_is_activated: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    if gate_is_activated:
        activated = gate.to(tl.bfloat16)
    else:
        activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
    tl.store(output_ptr + offsets, activated.to(tl.float32) * up, mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    maximum = max(M_VALUES) * N
    gate = torch.randn(maximum, device=device, dtype=torch.bfloat16)
    up = torch.randn(maximum, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(gate)

    configurations = tuple(
        (block, warps) for block in (256, 512, 1024, 2048) for warps in (4, 8)
    )
    for gate_is_activated in (False, True):
        mode = "multiply" if gate_is_activated else "swiglu"
        print(f"mode={mode}")
        summaries = []
        for block, warps in configurations:
            timings = []
            for m in M_VALUES:
                elements = m * N
                grid = (triton.cdiv(elements, block),)

                def invoke() -> None:
                    swiglu[grid](
                        gate,
                        up,
                        output,
                        n_elements=elements,
                        block=block,
                        gate_is_activated=gate_is_activated,
                        num_warps=warps,
                    )

                samples = [
                    triton.testing.do_bench(
                        invoke,
                        warmup=arguments.warmup,
                        rep=arguments.rep,
                    )
                    for _ in range(arguments.trials)
                ]
                timings.append(statistics.median(samples))
            geometric_mean = statistics.geometric_mean(timings)
            summaries.append((geometric_mean, block, warps, timings))
            print(
                f"  block={block:4d} warps={warps} "
                f"geo_ms={geometric_mean:.6f} "
                f"small_ms={timings[0]:.6f} large_ms={timings[-1]:.6f}"
            )
        best = min(summaries)
        print(f"best={best[1]}x{best[2]} geo_ms={best[0]:.6f}")


if __name__ == "__main__":
    main()
