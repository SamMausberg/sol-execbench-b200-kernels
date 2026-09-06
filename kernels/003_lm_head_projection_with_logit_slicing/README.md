# LM-head projection

Problem #3 projects BF16 hidden states `[B, S, 2048]` through the actual
weight matrix `[102400, 2048]`. The pinned reference computes every sequence
position. All 16 official workloads set `logits_to_keep` equal to `seq_len`;
the implementation returns the complete `[B, S, 102400]` result.

The frozen candidate uses explicit cuBLASLt configurations with FP32
accumulation and final BF16 output. Alternative CuTe implementations were
compared during development. The CuTe candidate supports writing the reversed GEMM orientation
directly into a column-major descriptor of the required output storage.
Shape, layout, alignment, and algorithm metadata determine dispatch. Inputs,
outputs, and workspaces are supplied afresh on every invocation.

`cute_gemm.py` is adapted from CUTLASS's
[`dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/4370102f9dacab813282e1d67722fceb0b90a019/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py).
Its BSD license is retained. The command-line demonstration and its testing
helpers are omitted, and the tile raster direction is configurable.
`cublaslt_bridge.h` reuses this repository's #10 bridge with its Apache license.

The frozen package passed all 16 official workloads in one full trial and
all 21 additional value, layout, and ownership checks. Its raw score was
`0.6705828072`, with geometric mean latency `350.251394 us`. The process
monitor observed no unrelated CUDA context during the trial. Clocks remained
unlocked, and no repeated performance qualification was performed.

The saved leaderboard snapshot leads at `0.536161`. The scoring bounds are
close to the baseline: approximately a 6% uniform latency increase would
erase this candidate's apparent advantage. It remains experimental and is
not recommended for submission under the clock uncertainty. No hosted
submission has been made.

`validation.json` records the source and report hashes. The package SHA-256 is
`7696f3e4e898fc78b21e0fc88b327b3815279974a0beeacafff0ce4d1f9a4cb3`.
The official run is
`.work/runs/3/20260906T212617.238544Z-lm-head-final-7696f3e4e898`.
Only `binding.cpp` and `cublaslt_bridge.h` belong to the official source set.
The persistent [archive index](../../results/2026-09-06/3/archive-index.json)
links that run, the changed-input audit, and the bounded tuning reports.

Run direct GPU work under the shared lock, including extension initialization:

```bash
source tools/native_env.sh
flock /workspace/sol-execbench-b200-kernels/.work/gpu.lock \
  python kernels/003_lm_head_projection_with_logit_slicing/bench_local.py \
  .work/problems/3 --indices 3,6,4,2 --heuristics 8 --iterations 20 \
  --output kernels/003_lm_head_projection_with_logit_slicing/tune_initial.json
```

The local input generator calls the definition's `get_inputs` factory, matching
the official weight scaling. `verify_values.py` covers fresh random values,
zero inputs, zero weights, basis inputs, cancellation, noncontiguous tensors,
shifted addresses, and input ownership. Development loaders are excluded from
the official source sets; cuBLASLt compilation uses the evaluator's first phase.
