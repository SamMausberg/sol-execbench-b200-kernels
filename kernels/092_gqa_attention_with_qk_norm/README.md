# Grouped causal attention with QK normalization on B200

The candidate fuses per-head FP32 RMS normalization and BF16 rotary position
embedding preparation. It keeps eight KV heads and maps the 96 query heads
directly onto them, without materializing repeated keys or values. Q/K/V and
output projections use PyTorch's matrix multiplication libraries.

Attention rounds the QK product and scaled logits to BF16 before softmax.
Short sequences that fit one key tile normalize directly in that tile. Longer
sequences use two passes: the first computes each row's global maximum and
normalizer; the second repeats QK, rounds normalized probabilities to BF16,
and multiplies by V. This preserves the reference's BF16 intermediate stages.

Library Flash and cuDNN attention were evaluated first. Both passed the
official random workloads and fresh seeds, but the cuDNN candidate failed a
valid input with Q normalization weights scaled by four: only 96.69% of
elements met the official tolerance, below the required 99%. The failing
input report is preserved in `validation_edges_cudnn.json`, with its source
in `experimental_cudnn.py`. That file is excluded from `solution.json`.
The custom attention recovered 100% of elements within tolerance on that
case. A faster online variant reached 99.56%; the candidate keeps globally
normalized BF16 probabilities.

From the repository root:

```sh
source tools/native_env.sh
flock "$SOL_GPU_LOCK" python kernels/092_gqa_attention_with_qk_norm/bench_local.py \
  .work/problems/92 --indices 12 \
  --variants no_rotation,zero_q_norm,zero_values,small_input_epsilon,strong_q_norm
python kernels/092_gqa_attention_with_qk_norm/bench_monitored.py \
  --output kernels/092_gqa_attention_with_qk_norm/gpu_processes_final.json \
  --solution kernels/092_gqa_attention_with_qk_norm/solution.json --trials 3
```

The campaign takes the shared GPU lock itself. Its wrapper records compute
process PIDs during the run so overlapping external GPU work can be detected.
All performance measurements use unlocked B200 clocks and need hosted
evaluation to establish an official rank.

The frozen candidate passed all 16 official workloads in one full trial on
2026-09-06, plus five additional edge cases and the stronger normalization
case at sequence length 2048. The full-run timing is **invalid for ranking**:
the monitor recorded foreign GPU processes overlapping the evaluator.
`validation.json` records the provenance and this limitation. Earlier stage
tuning already showed a substantial performance deficit, so the candidate
is retained as experimental and is not recommended for submission.

The immutable official run is
`.work/runs/92/20260906T183855.993912Z-rounded-exact-full-11905b1d6f2d/`.
