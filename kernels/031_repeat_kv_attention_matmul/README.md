# BF16 attention score GEMM

The selected solution fuses FP32 scaling into the BF16 output. Aligned sequences
use the library GEMM epilogue. Short odd sequences use a custom tensor-core tile;
longer odd sequences use persistent TMA loads and masked output stores. Dispatch
depends on tensor dimensions. Inputs are read without mutation, and every result
is written into the supplied output.

All 16 pinned workloads passed in three official evaluator trials on 2026-09-06.
The local mean SOL score using median workload latencies was 0.635278; the trial
scores were 0.634976, 0.635692, and 0.632798. A refreshed public snapshot reported
the hosted leader at 0.594534. The local score is 6.85% higher and would fall to
that score after an 8.11% uniform latency increase.

Clocks were unlocked; trial snapshots reported 1965 MHz SM and 3996 MHz memory.
The official SM preset is 1500 MHz, so this is a candidate for hosted evaluation,
not an established leaderboard result.

| Batch, sequence | Library median, us | Selected median, us |
| --- | ---: | ---: |
| 1, 131 | 11.4245 | 4.768 |
| 2, 449 | 55.936 | 15.552 |
| 1, 1321 | 196.8785 | 47.9995 |
| 2, 1571 | 519.133 | 127.552 |

The exact selected package SHA-256 is
`ce09baae28bcf17c282a6c60d2339c492deb0761e63b12bccfc905a87cbd054b`.
Raw traces, sources, environment, clock snapshots, and score calculations are in
`.work/runs/31/20260906T175740.379421Z-qk-bf16-hybrid-final-ce09baae28bc`.

The separate variation audit passed all 30 cases under the pinned evaluator's
tolerance, with unchanged inputs and poisoned destinations. Twenty-seven cases
also passed every element's tolerance. Scaling inputs by 16 caused one or two
near-zero mismatches in each of three cases; the lowest matched ratio was
99.9996185%, versus the contract's required 99%. Full counts and errors are in
`.work/tuning/31-hybrid-variation-checks.json`.

```bash
source tools/native_env.sh
python tools/campaign.py bench 31 \
  --solution kernels/031_repeat_kv_attention_matmul/solution.json \
  --trials 3 --label qk-bf16-hybrid-final
python kernels/031_repeat_kv_attention_matmul/verify.py \
  --output .work/tuning/31-hybrid-variation-checks.json
```

`tune.py` uses official input shifting and CUPTI timing under the shared GPU lock.
The initial 75 candidates, 18 one-stage candidates, and 24 BK64 candidates are
recorded in `.work/tuning/31-*.json`. The latter two experiments were slower than
the selected BK128, two-stage schedule. `library.solution.json` retains the pure
library comparison. Experimental tuners and audit code are excluded from the
submission manifest.
