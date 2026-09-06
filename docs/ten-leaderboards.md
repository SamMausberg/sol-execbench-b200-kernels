# Goal: #1 on ten B200 problem leaderboards

Develop ten NVIDIA SOL-ExecBench candidates with strong chances of taking the top
individual v1.1 B200 positions. Validate all official workloads locally, compare
against fresh public rankings, and regularly push exact source packages and results.
Samuel Mausberg handles hosted submissions; this campaign has used no hosted quota.

Current progress: **three candidates above their public leaders in local evaluation;
zero hosted #1 results confirmed for this campaign.** The Runpod host cannot match
the official 1500 MHz SM clock setting, so local leads need additional margin.

| Problem | Local score | Public leader | Local evidence | Next action |
| --- | ---: | ---: | --- | --- |
| 049 grouped QK scores | 0.645105 | 0.579975 | 16 workloads x 3 trials; 30 input variations | Improve the 19.80% uniform latency buffer |
| 031 attention QK scores | 0.635278 | 0.594534 | 16 workloads x 3 trials; 30 input variations | Improve the 8.11% uniform latency buffer |
| 010 value projection and transpose | 0.635502 | 0.534899 | 16 workloads x 3 trials; 30 exact input variations | Tune GEMM algorithms; current latency buffer is 17.54% |
| 033 post-norm residual | 0.782529 | 0.795472 | 16 workloads x 3 trials; 4 exact edge cases | Below leader; retain for collection coverage |

The remaining positions will be selected from measured opportunities. Work is
underway on 050 grouped QKV projection, 119 MoE backward, and 173 vision attention.
Problems 030 and 092 now pass all official workloads, but their experimental
implementations do not establish leads. Problem 218's exact library comparison
shows a large apparent score advantage under different clocks without a kernel improvement.
Problem 121 passes all 16 workloads in one trial, but its 0.419214 local
score is below the 0.573023 leader; it remains an experimental checkpoint.
Other completed activation and normalization candidates are recorded in the
[local results](../results/2026-09-06/README.md); none currently exceeds its leader.

A local score lead is separate from an official rank. Submission recommendations
must identify the exact tested JSON, refreshed leader, timing variance, numerical
checks, and remaining hardware differences. Do not mark this goal achieved from
local scores alone or submit automatically.

Uniform latency buffers measure how much every workload could slow down before
the local score ties the snapshot leader. They are sensitivity calculations and
do not predict the effect of changing the GPU clock. Problem 031's variation audit
passes the pinned 99% matching requirement in all 30 cases; 27 also satisfy every
element's tolerance. The remaining three have one or two near-zero deviations.

The newer 010 cuBLASLt candidate passes nine full correctness trials and 30 input
variations. A monitor qualified one timing trial at 0.663578 and rejected five
other attempts for foreign CUDA contexts. It needs two more uncontended repeats
before replacing the table entry; its uniform latency buffer is 22.67%.
This campaign continues to use zero hosted submission quota.
