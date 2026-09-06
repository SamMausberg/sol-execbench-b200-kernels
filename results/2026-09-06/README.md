# Local B200 validation, September 6, 2026

These results use NVIDIA's unmodified, pinned v1.1 evaluator on the Runpod B200.
Every listed candidate passed all its official workloads in three complete trials.
The host denies SM clock locking, so scores are local estimates and do not establish
a hosted leaderboard rank. NVIDIA's hosted evaluations fix the SM clock at 1500 MHz.

| Problem | Workloads | Local SOL score | Published leader snapshot | Submission recommendation |
| --- | ---: | ---: | ---: | --- |
| 025 video GELU | 16 | 0.630734 | 0.652275 | Continue optimizing |
| 038 Q/K RMSNorm, Triton alternative | 16 | 0.596559 | 0.615074 | Retain as alternative |
| 049 grouped QK scores | 16 | 0.645105 | 0.579975 | Local lead; improve clock margin |
| 053 Gaussian sparse activation | 12 | 0.638735 | 0.657158 | Continue optimizing |
| 084 SiLU backward | 16 | 0.527926 | 0.560079 | Continue optimizing |
| 085 GEGLU | 16 | 0.656663 | 0.680444 | Continue optimizing |
| 088 rotary embedding | 16 | 0.678962 | 0.688375 | Continue optimizing |

Scores use the arithmetic mean of workload scores, computed from each workload's
median latency across trials and its stored scoring baseline and SOL bound.
Problem 088 combines its initial complete trial and the two complete repeat trials.
Each kernel's `validation.json` records its exact package hash and result provenance.
Run directories here preserve the embedded source package, evaluator traces, summary,
contract hash, scoring snapshot, and environment for review after the pod is stopped.
Generated input definitions and workload files stay in `.work`, as required by the
repository checks; `campaign.py fetch` reconstructs them from the pinned dataset.
No hosted submissions were made.

To reproduce, run `bash tools/bootstrap_native.sh`, source `tools/native_env.sh`,
then use `tools/campaign.py bench ID --solution PATH --trials 3`.
GPU evaluations acquire `/workspace/sol-execbench-b200-kernels/.work/gpu.lock`.
Other GPU jobs on the same pod should acquire that same lock.

The leaderboard values are snapshots of the official problem boards:
[025](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/25/B200),
[038](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/38/B200),
[049](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/49/B200),
[053](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/53/B200),
[084](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/84/B200),
[085](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/85/B200),
[088](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/88/B200).
