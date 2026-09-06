# RMS normalization with residual on B200

`run(sublayer_output, residual, weight, eps, output)` fuses FP32 RMS
normalization and weight multiplication, rounds that result to BF16, then adds
the residual and writes BF16. The intermediate rounding is necessary: adding
the residual directly to the FP32 normalized value produces different results
near cancellation.

The kernel loads the residual before reducing the squared input, which overlaps
memory access with the reduction. It uses 16 warps for at most 512 rows and four
warps for larger tensors. Input, residual, weight and output strides are
respected. The launch choices are recorded in `tune_initial.json` and
`tune_small.json`, measured with the official cold L2 CUPTI timer.

From the repository root:

```sh
source tools/native_env.sh
flock "$SOL_GPU_LOCK" python kernels/033_post_norm_residual/bench_local.py \
  .work/problems/33 --edges-only
python tools/campaign.py bench 33 \
  --solution kernels/033_post_norm_residual/solution.json --trials 3
```

`campaign.py` handles the shared GPU lock internally. Validation on 2026-09-06
passed all 16 official workloads in each of three trials, plus four additional
checks for BF16 cancellation, zero input, zero weight, and strided tensors with
a different epsilon. `validation_edges.json` records the additional checks.

The local SOL estimate was **0.78252910** with unlocked clocks. It falls below
the refreshed leader at 0.795472 and does not justify a standalone hosted
submission. The implementation is available for collection coverage.

The immutable run is
`.work/runs/33/20260906T180317.763778Z-fused-v1-final-901b981150c3/`.
The validated `kernel.py` SHA256 is
`717272822b5fe15ba572b42d3a9e0384f0e6f32c2f87d0cbd68d9a04c4e72872`.
