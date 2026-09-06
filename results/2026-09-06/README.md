# Local B200 validation, September 6, 2026

These results use NVIDIA's unmodified, pinned v1.1 evaluator on the Runpod B200.
Every candidate in the table passed all its official workloads in three complete trials.
The host denies SM clock locking, so scores are local estimates and do not establish
a hosted leaderboard rank. NVIDIA's hosted evaluations fix the SM clock at 1500 MHz.

| Problem | Workloads | Local SOL score | Published leader snapshot | Submission recommendation |
| --- | ---: | ---: | ---: | --- |
| 010 value projection and transpose | 16 | 0.635502 | 0.534899 | Local lead; improve clock margin |
| 025 video GELU | 16 | 0.630734 | 0.652275 | Continue optimizing |
| 031 attention QK scores | 16 | 0.635278 | 0.594534 | Local lead; improve clock margin |
| 033 post-norm residual | 16 | 0.782529 | 0.795472 | Retain for collection coverage |
| 038 Q/K RMSNorm, Triton alternative | 16 | 0.596559 | 0.615074 | Retain as alternative |
| 049 grouped QK scores | 16 | 0.645105 | 0.579975 | Local lead; improve clock margin |
| 053 Gaussian sparse activation | 12 | 0.638735 | 0.657158 | Continue optimizing |
| 084 SiLU backward | 16 | 0.527926 | 0.560079 | Continue optimizing |
| 085 GEGLU | 16 | 0.656663 | 0.680444 | Continue optimizing |
| 088 rotary embedding | 16 | 0.678962 | 0.688375 | Continue optimizing |

Scores use the arithmetic mean of workload scores, computed from each workload's
median latency across trials and its stored scoring baseline and SOL bound.
Problems 010 and 088 combine their initial complete trial and two complete repeat trials.
Each kernel's `validation.json` records its exact package hash and result provenance.
Run directories here preserve the embedded source package, evaluator traces, summary,
contract hash, scoring snapshot, and environment for review after the pod is stopped.
Generated input definitions and workload files stay in `.work`, as required by the
repository checks; `campaign.py fetch` reconstructs them from the pinned dataset.
No hosted submissions were made.

Problem 121 is archived separately as an experimental checkpoint: all 16 workloads
passed in one complete trial, with local score 0.419214 versus leader 0.573023.
It also passes five changed-input checks, but does not justify more full trials
until its performance improves.

Additional experimental checkpoints preserve full correctness evidence without
claiming accepted ranking timings:

- 030 projection with residual: 16 workloads and 32 input variations pass.
  The unqualified observed score is 0.498701, below leader snapshot 0.546380.
- 092 grouped causal attention: 16 workloads and six extra cases pass with
  explicit BF16 logits and normalized probability rounding. Foreign GPU jobs
  overlapped the full trial, so its timing is excluded from ranking comparisons.
- 218 GEMM: the exact library comparison passes 25 workloads with a documented
  optional-metadata adapter. Its raw score of 0.687683 exceeds the leader snapshot
  0.535630 under different clocks; this does not demonstrate a kernel improvement.

To reproduce, run `bash tools/bootstrap_native.sh`, source `tools/native_env.sh`,
then use `tools/campaign.py bench ID --solution PATH --trials 3`.
GPU evaluations acquire `/workspace/sol-execbench-b200-kernels/.work/gpu.lock`.
Other GPU jobs on the same pod should acquire that same lock.
The [qualification helper](../../tools/qualify_gpu.md) monitors CUDA process
ownership, rejects observed overlap and retries only contaminated timing trials.

The configured cuBLASLt variant for problem 010 passes nine complete correctness
trials and 30 changed-input checks. One monitored trial qualifies at 0.663578;
five other monitored attempts contain foreign CUDA contexts and are excluded.
The original 010 table entry retains its three-trial estimate until the new
variant obtains two more uncontended repeats. Its exact source, all trials, and
the process qualification report are preserved in the 010 result directories.

The leaderboard values are snapshots of the official problem boards:
[010](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/10/B200),
[025](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/25/B200),
[031](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/31/B200),
[033](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/33/B200),
[038](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/38/B200),
[049](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/49/B200),
[053](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/53/B200),
[084](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/84/B200),
[085](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/85/B200),
[088](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/88/B200),
[121](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/121/B200).

The MoE backward (119) and GIT encoder (173) investigations are archived with
partial evaluator runs and diagnostic hashes. Neither is a submission candidate.
The encoder work also established a reusable numerical distinction for problem
36: explicit FAST_TF32 fails 32× weight scaling, while BF16x9-emulated FP32
passes those checks. These library measurements remain component experiments.

Further archived correctness checkpoints:

- 050 grouped QKV with bias passes three full trials and ten exact edge cases.
  All timings contain observed foreign CUDA contexts; no timing trial qualifies.
- 055 grouped QKV with output views passes four full trials for V1 and one for
  V2. V1 passes 30 additional value/view calls; V2 passes 38 calls covering strides,
  pointer alignment, values, and output ownership. All 204 audited outputs are
  exact. The five full timing trials contain foreign CUDA contexts. V2's raw
  score of 0.796083 versus leader 0.684349 needs clean repeat measurements.
- 036 Flux processing passes 16 official workloads and 16 additional input
  checks using explicit BF16x9-emulated FP32. Its unqualified score of 0.352235
  is below leader 0.644426. A six-product expansion passes 192 additional
  numerical checks, but its full conversion path is slower than BF16x9.

Each problem's `archive-index.json` connects exact run artifacts to diagnostic
reports. Copies of older GPU process reports reduce executable paths to basenames
and record both original and archived hashes. Process identities, timestamps, and
qualification decisions remain intact. These checkpoints add no confirmed lead
and use no hosted submission quota.

The next checkpoint preserves three more complete candidates:

- 004 attention projection backward: selected cuBLASLt version passes all 16
  workloads and 40 value, layout, and ownership checks. Explicit alignment
  handling fixes a failure found in the original ATen version's extra audit.
  Its raw 0.799523 score versus leader 0.753779 has only a 9.44% uniform latency
  buffer and contains 14 foreign CUDA context observations.
- 006 Hyena depthwise convolution: all 16 workloads and 30 extra calls pass
  bitwise. Its raw 0.772851 score remains below leader 0.785398. The full timing
  run contains foreign contexts and is excluded from qualification.
- 043 MLA projections: all 16 workloads, 36 value/scalar/layout/ownership calls,
  and three additional unaligned calls pass. Its unqualified 0.769533 score
  versus leader 0.719510 has only a 6.87% uniform latency buffer. Bounded native
  and CuTe alternatives do not establish a useful further improvement.

All three are parked experiments with zero hosted submissions. Their archive
indices retain exact packages, full and partial trials, input audits, and
diagnostic source hashes.

Problem 055 V2 now has two qualified timing trials at 0.796167 and 0.796062.
Their combined median-latency score is 0.796114 versus refreshed leader 0.684349;
the largest workload spread is 0.59%. Two further attempts were contaminated
and are excluded. All nine full V1/V2 trials pass correctness. V2 still needs
one more qualified repeat and a stronger clock margin before a confident
submission recommendation; its uniform latency buffer is 25.47%.

Problem 048's dual-accumulator CuTe kernel passes all 16 official workloads and
11 changed-input/layout checks. It computes paired gate/up projections in TMEM,
then preserves the reference's BF16 rounding around GELU and multiplication.
Its unqualified 0.770943 score versus leader 0.719587 has only a 3.76% uniform
latency buffer. The failed DSMEM prototype and successful replacement are both
archived, with the selected candidate identified by its exact package hash.
