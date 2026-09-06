# Goal: #1 on ten B200 problem leaderboards

Develop ten NVIDIA SOL-ExecBench candidates with strong chances of taking the top
individual v1.1 B200 positions. Validate all official workloads locally, compare
against fresh public rankings, and regularly push exact source packages and results.
Samuel Mausberg handles hosted submissions; this campaign has used no hosted quota.

Current progress: **two candidates above their public leaders in local evaluation;
zero hosted #1 results confirmed for this campaign.** The Runpod host cannot match
the official 1500 MHz SM clock setting, so local leads need additional margin.

| Problem | Local score | Public leader | Local evidence | Next action |
| --- | ---: | ---: | --- | --- |
| 049 grouped QK scores | 0.645105 | 0.579975 | 16 workloads x 3 trials; 30 input variations | Improve the 19.80% uniform latency buffer |
| 031 attention QK scores | 0.635278 | 0.594534 | 16 workloads x 3 trials | Tune longer unaligned cases; current latency buffer is 8.11% |
| 010 value projection and transpose | Pending | 0.534899 | Contract reviewed; implementation underway | Validate output layout freedom and measure full workloads |
| 033 post-norm residual | Pending | 0.792710 | Representative reduction variants pass | Freeze dispatch and run all workloads |

The remaining positions will be selected from measured opportunities. Problem 121
has correct experimental fused attention but has not yet demonstrated a lead.
Other completed activation and normalization candidates are recorded in the
[local results](../results/2026-09-06/README.md); none currently exceeds its leader.

A local score lead is separate from an official rank. Submission recommendations
must identify the exact tested JSON, refreshed leader, timing variance, numerical
checks, and remaining hardware differences. Do not mark this goal achieved from
local scores alone or submit automatically.
