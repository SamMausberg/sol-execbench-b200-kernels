# Grouped QK attention scores

The selected package shares a key matrix across its query heads and applies scaling
inside the GEMM epilogue. Small outputs use a custom Triton tile. Larger unaligned
outputs use persistent tensor-map loads with masked stores. Aligned outputs use a
library batched GEMM with `alpha=scaling` and `beta=0`.

All 16 official workloads passed three complete local v1.1 trials. The local SOL
score is **0.645105**, versus the refreshed public leader's **0.579975**. This is an
11.23% local score lead. The maximum workload latency spread was 2.57%.
Thirty additional checks passed with changed scalar values, noncontiguous inputs,
and new input values written into reused storage.

The Runpod host denies the official 1500 MHz SM clock lock. A uniform 19.80%
increase in every measured latency would erase the local lead; this calculation
is a sensitivity analysis, not a prediction of hosted performance. No hosted #1
rank is claimed and no hosted submission has been made.

`validation.json` records the exact package hash and evidence. The same package is
archived under `results/2026-09-06/49` and generated locally as
`dist/049_grouped_qk_b200_v2.json`. Development alternatives remain available for
comparison; `solution.json` selects `dispatch.py` and its explicit dependencies.
