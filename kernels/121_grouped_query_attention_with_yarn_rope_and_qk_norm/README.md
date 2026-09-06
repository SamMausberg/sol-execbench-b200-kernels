# Grouped attention with YARN rotation and Q/K normalization

The candidate passes all 16 official workloads and five checks that change the actual input values. Its local SOL score is **0.419214**, below the public leader snapshot of **0.573023**. It is an experimental checkpoint, and is not recommended for submission.

The implementation uses four PyTorch BF16 projections and two Triton kernels. One Triton kernel combines full-width Q/K RMSNorm, coefficient generation, and YARN rotation. The attention kernel uses the original eight KV heads directly and performs online softmax without materializing repeated KV tensors or the complete attention matrix.

Normalization reads both supplied weight vectors. Rotation reads actual positions, inverse frequencies, and scalar parameters. Attention reads the supplied additive mask and skips a tile only after confirming that every mask entry in that tile is negative infinity. It supports finite additive masks and changes to values in reused tensor storage. Normalization, coefficients, rotation products, attention dot products, scaling, and mask addition retain the reference's BF16 rounding boundaries. Online softmax uses FP32 state and BF16 tensor-core operands.

## Validation and limitations

The final package SHA256 is `480c8ebbc28b08d88edafa89a622706c57b7742d011d42d472ae4b92e99900a0`. The full run is `.work/runs/121/20260906T180023.950832Z-grouped-fused-final-480c8ebbc28b`. Exact contract, source, environment, and trace hashes are in `validation.json`.

One full trial was sufficient to establish that this candidate is below the leader. The geometric mean latency was 0.434712 ms. The largest sequence case, batch 4 and sequence 2131, took 4.809082 ms and remains the main performance problem. These results use the unmodified v1.1 evaluator at revision `a9fa0804c793d438e70850c33fe34426e66d53dd`. The Runpod host denies clock locking, so all measurements record unlocked clocks and do not establish an official ranking.

The changed-input checks cover nonuniform normalization weights, arbitrary finite masks, changed positions/frequencies/scalars, and fresh values written into reused input storage. All pass a 99% matching requirement with absolute tolerance 0.01 and relative tolerance 5%; the lowest observed matching ratio was 99.9989569%.

The exported workloads contain `required_match_ratio: 0.98`. The pinned parser's actual field is `required_matched_ratio`; it therefore uses its default 0.99 for those workloads. No contract or evaluator was modified.

## Experiments

`variants/kernel_cudnn.py` preserves the initial cuDNN prototype. Its long-case attention kernel took about 3.19 ms. The standard Triton attention sweep reached about 1.14 ms for attention alone with 64 query rows, 128 key rows, four warps, and one stage. Those component timings exclude projections and normalization and are not full-problem scores.

Making heads contiguous did not materially improve the attention kernel. Descriptor loads reached about 1.06 ms for attention alone, but automatic warp specialization failed inside the pinned Triton compiler with `ttng.tmem_alloc operation destroyed but still has uses`. `variants/attention_experiments.py` and `tune_attention.py` preserve the tested variants and reproducible tuning entry point. `profile_timeline.py` is a diagnostic profiler; its instrumented total spans are not benchmark scores.

```bash
source tools/native_env.sh
python tools/campaign.py bench 121 \
  --solution kernels/121_grouped_query_attention_with_yarn_rope_and_qk_norm/solution.json \
  --trials 1
python kernels/121_grouped_query_attention_with_yarn_rope_and_qk_norm/verify_inputs.py
```
