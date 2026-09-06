# Gaussian sparse activation on B200

`run(inputs, target_sparsity, output)` implements the official destination
passing interface. It computes each row's FP32 mean and population standard
deviation, subtracts the Gaussian threshold, applies ReLU, and writes BF16.
Zero sparsity copies the input unchanged, including negative values.

The scalar quantile uses the reference rational approximation, with explicit
FP32 rounding after every operation. It depends on the current Python scalar
argument and does not read or cache tensor values. The GPU kernel reads each
input element once and writes each output element once.

The launch uses four warps for widths up to 4096 and eight for larger widths.
For short tensors with width 12288, paired first and second moments share one
reduction. Rows whose mean would make moment subtraction poorly conditioned
recompute the variance around the mean. Other shapes use centered variance
directly. Floating point contraction is disabled.

Run the development checks from the repository root:

```sh
source tools/native_env.sh
flock "$SOL_GPU_LOCK" python kernels/053_gaussian_topk_sparse_activation/bench_local.py \
  .work/problems/53 --edges --indices 0
```

The edge checks exercise 36 combinations covering zero sparsity, both quantile
tails and branch boundaries, constant and shifted inputs, strided tensors,
width 511, width 1, and the centered fallback for width 12288. The tuning harness
also accepts `--tune` for warp counts and `--experiments` for moment and cache
policies; it uses the official input generator, tolerances, and cold L2 CUPTI
timer. `tune_initial.json` and `tune_moments_cache.json` preserve those sweeps.

Run all 12 official workloads with the unmodified evaluator:

```sh
python tools/campaign.py bench 53 \
  --solution kernels/053_gaussian_topk_sparse_activation/solution.json \
  --trials 3
```

`campaign.py` acquires the shared GPU lock internally. Timing on this Runpod
uses unlocked clocks because the host rejects clock locking. Derived local
SOL estimates do not establish an official leaderboard position.

Validation on 2026-09-06 passed all 36 edge cases and all 12 official workloads
in each of three trials. The local mean SOL estimate was **0.63873524**;
individual trial means were 0.63779439, 0.63950082, and 0.63916503. This does not
justify a standalone hosted submission against the recorded leader at
0.657158. The implementation remains useful for collection coverage.

The immutable run is
`.work/runs/53/20260906T174720.558504Z-fused-v1-final-92d54b609389/`.
Its `summary.json` links the three raw traces, pinned contract, submission, and
environment records. The validated `kernel.py` SHA256 is
`8bda8ef89c8421a9ca9725d90314bd6a9db266eb8a8937a4a493d46c38ecc1e3`.
