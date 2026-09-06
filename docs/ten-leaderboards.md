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

The remaining positions will be selected from measured opportunities. Problem
055 passes nine full correctness trials across two frozen versions. The current
version also passes 38 layout and value checks with 114 exact outputs. Two
timing trials qualify at a combined score of 0.796114 versus leader 0.684349,
with a 25.47% uniform latency buffer. One further qualified repeat and stronger
clock-margin evidence remain needed. Other timing trials contain foreign
CUDA contexts and are excluded. Problem 050 passes three full trials
and ten exact edge cases, also without a qualified timing trial.

Problem 036 passes all official cases and 16 extra checks with explicit FP32
emulation. Its six-product expansion passes 192 checks, but conversion overhead
prevents a complete-path speedup. It is parked below the leader. Current work
targets 003 vocabulary projection, 032 attention output layout, 035 modulation
projection. Problem 048 passes all official workloads and 11 extra checks;
its unqualified 0.770943 score has only a 3.76% uniform latency buffer and is
parked. A bounded CuTe experiment for
036 will generate FP32 fragments inside GEMM shared memory to reduce conversion
traffic. Problems 119 and 173 remain partial investigations; neither establishes a lead.

Problems 004, 006, and 043 now pass full official correctness and additional
input checks. Problem 006 remains below its leader. Problems 004 and 043 have
unqualified raw leads with only 9.44% and 6.87% uniform latency buffers, respectively;
they remain parked and do not add to the three local leads in the table.
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
