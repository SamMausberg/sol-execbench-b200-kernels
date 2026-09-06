# BF16 QKV projection with GQA views

The current candidate is `v2.solution.json`: five full correctness trials,
38 additional layout/value calls, and two qualified timing trials pass.
One more qualified repeat and stronger evidence about the clock difference
remain needed before a confident submission recommendation.

The frozen v1 manifest is `solution.json`. It computes Q, K, and V in one
persistent Blackwell kernel, then returns ordinary tensor views with the
reference's batch/head/sequence/dimension layouts. The output contract specifies
shape and dtype without requiring contiguous storage; the reference itself
returns these transpose views.

All three weight matrices remain separate. The grouped scheduler chooses the
appropriate weight and output descriptor for each output-column tile. Products
accumulate in FP32 and round once to BF16. Every call allocates new output
storage and reads its current inputs. The cache contains compiled functions
keyed by row count, device, and tile configuration; it retains no tensor values,
outputs, workspaces, or input-dependent decisions.

Rows up to 256 use 64-by-64 tiles, rows up to 512 use 128-by-128 tiles, and
larger cases use 256-by-256 tiles spanning two cooperating CTAs. The implementation
adapts the repository's problem 050 grouped scheduler, derived from CUTLASS
4.4.1's persistent GEMM example. The original BSD license is retained.
This variant removes all bias loads and epilogue arithmetic. The inherited C
arguments provide layout metadata only and are not read by the kernel.

All 16 official workloads passed the selected package. After tile selection, 30 variations
across three shapes produced 90 outputs bitwise equal to the pinned reference.
The audit checks zero inputs, basis vectors, changed weights, signed values,
small and large amplitudes, matching output strides, input preservation, and
preservation of previously returned outputs. Its report is
`.work/tuning/55-variation-selected.json`, with SHA256
`640287c0e4889b11473b9801e8fcd5e6c06f030132ff4d2445de0d40f1ac5a9b`.
The selected JSON package has SHA256
`d5b468f29fd9ed221f2f4ff4483b9d6d28b92b5e63d60f8ce4e05ebea093c6e7`.
Its full correctness run is
`.work/runs/55/20260906T194527.947495Z-qkv-cute-queued-20260906T194525497646Z-a1-d5b468f29fd9`.
The process monitor rejected that trial's timing because another CUDA context
was present throughout the measurement window. Its raw score of 0.791501 is
excluded from any qualified comparison. The audit is
`.work/qualification/55/20260906T194525497646Z-qkv-final-queued/qualification.json`.
A further bounded campaign completed three full passing trials, all rejected
for foreign CUDA contexts. Its report is
`.work/qualification/55/20260906T195228550858Z-qkv-cute-clean/qualification.json`.
That campaign produced no qualified timing trial for V1.

The frozen v1 evidence covers the official contiguous, aligned input layouts.
`v2.solution.json` adds a guard for those TMA assumptions and uses ordinary
linear projections for other input strides or pointer alignment. The fallback
also returns fresh concrete output views. `verify_layouts.py` alternates layouts
and values at the same shapes and checks that earlier outputs remain unchanged.
All 38 calls and 114 outputs passed bitwise against the reference, including
per-input column strides, padded rows, transposes, and unaligned storage offsets.
Dense calls before and after these changes exercise the original compiled cache.
The report is `.work/tuning/55-layout-v2.json`. The v2 package has SHA256
`6e1a242602da79feaa02ca7173f0aafae7c9cc8c964d3fd4918d0f80f3074d75`.
V2 also passes all 16 official workloads in
`.work/runs/55/20260906T200624.854758Z-qkv-v2-initial-20260906T200622947830Z-a1-6e1a242602da`.
Its timing is excluded for observed foreign CUDA contexts; the monitor report is
`.work/qualification/55/20260906T200622947830Z-qkv-v2-initial/qualification.json`.
V2 is the candidate for qualification because it also supports the additional
input layouts.

A later quiet window completed four more V2 trials, all passing correctness.
Two qualify under the unchanged process-window audit, at scores 0.796167 and
0.796062. The combined median-latency score is 0.796114 versus refreshed leader
0.684349. The largest workload timing spread between those trials is 0.59%.
Two other trials contain foreign CUDA contexts and are excluded. One further
qualified repeat remains needed for the three-trial protocol.

The two qualified trials give a 25.47% uniform latency buffer. This is a
sensitivity calculation, not a prediction of the clock difference. A hypothetical
uniform 31% slowdown would produce 0.664693, below the leader; no confident
submission recommendation is made yet. The exact reports are linked from
`validation.json` and the persistent
[partial qualification](../../results/2026-09-06/55/qualified-v2-partial.json).

`triton.experimental.solution.json` retains the earlier plain/TMA implementation.
`library.py` is the exact three-matmul comparison. `tune.py` compares bounded
Triton and CuTe configurations with the official shifted-input/output CUPTI
timer. Tuning reports distinguish observed competing CUDA processes; later
CuTe sweeps had contention and are diagnostic.

After sourcing `tools/native_env.sh`, run the value and view audit with
`python kernels/055_attention_qkv_projection_with_gqa_reshape/verify.py`.
Full evaluation uses the existing `tools/qualify_gpu.py` helper and shared GPU
lock. Runpod clocks remain unlocked, so local results do not establish a hosted
rank. No hosted submission has been made by this campaign.
