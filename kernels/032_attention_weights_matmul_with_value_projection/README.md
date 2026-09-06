# Attention/value products with direct output layout

This candidate is parked. It passes all 16 official workloads in a clean
monitored trial, but its local score of 0.727321 is below the public leader
at 0.749744. All 48 changed-input and ownership checks pass.

This kernel computes BF16 attention weights `[B,40,S,S]` times BF16 values
`[B,40,S,128]`, accumulates in FP32, and returns a fresh contiguous tensor
`[B,S,5120]`. The reference materializes a head transpose after its batched
matrix product. Each selected kernel instead stores directly into the final
layout using `(batch*S + row)*5120 + head*128 + column`.

`solution.json` selects persistent CuTe GEMM for aligned sequence lengths and
masked Triton for odd lengths. Other input strides or unaligned base pointers
use the same matrix product, transpose, and reshape as the reference. Dispatch
uses tensor metadata only. All results are computed from the current inputs;
only compiled functions are cached.

The CuTe tensors group heads and examples into a nested batch mode. Output
shape `(S,128,(40,B))` has strides `(5120,1,(128,S*5120))`, so every head writes
a disjoint part of the output. The scheduler counts the logical batch elements,
while the TMA descriptor preserves both strides. `cute_av.py` derives from the
CUTLASS v4.4.1 persistent GEMM example and retains its complete BSD license.
`cute-source.json` records the upstream source and exact adaptation.

The bounded comparisons include legacy cuBLAS and cuBLASLt batched products,
Triton geometries, both CuTe matrix orientations, and padding the reduction
axis to enable TMA for odd lengths. The selected direct kernels beat the padded
variants. Native implementations under `native` and `native_lt` are retained
as experiments. NVIDIA documents interleaved matrix outputs through leading
dimensions and batch strides in its
[batched GEMM article](https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/).

`tune.py` and `tune_cute.py` use the pinned evaluator's CUPTI timer with shifted
arguments and cold L2. Their standalone reports record sampled CUDA process
identities using basenames. The selected entry point is frozen in `selected.py`;
its full official trial and changed-input audit are recorded in `validation.json`.
The package SHA256 is
`ae6a44a926e27ecf82ad127ad8ffb73229884185cbee8ead5ef03aa41d30e6e7`.
`verify.py` checks changed attention/value data, basis and cancellation inputs,
scaling, per-head values, unaligned storage, column strides, transposed attention,
and fresh output ownership across calls with changing layouts.
The persistent [archive index](../../results/2026-09-06/32/archive-index.json)
retains all three evaluator runs, the qualified timing audit, and diagnostics.

Runpod clocks are unlocked. Local scores require separate hosted evaluation
before they establish a leaderboard rank. This campaign has made no hosted
submission for this problem.
