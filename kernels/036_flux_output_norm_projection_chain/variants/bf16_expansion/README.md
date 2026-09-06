# BF16 expansion experiment for problem 36

This experiment is parked. Both the three-product and four-product implementations fail problem 36's original tolerance when both projection weights are multiplied by 32. They are unsuitable replacements for the precise BF16x9 path.

The CUDA kernel converts each actual FP32 operand to `hi = BF16_rn(x)` and `lo = BF16_rn(x - float(hi))`. The native C++ bridge accumulates `lo*hi`, `hi*lo`, and `hi*hi` into FP32. The four-product variant also includes `lo*lo`. The final GEMM applies the actual FP32 bias. All work uses the caller's CUDA stream. The only cached state consists of CPU shape descriptors and cuBLASLt algorithm metadata.

The correctness experiment uses actual pinned problem 36 inputs for workloads 0, 1, 3, and 15, including batch sizes 1, 2, and 32 and output row counts 128, 586, 1024, and 8192. For each expansion, it replaces the modulation GEMM, output GEMM, and then both GEMMs. The original PyTorch normalization and modulation arithmetic remains in the numerical test so the GEMM error can be isolated.

| Input weights | Three products | Four products |
| --- | ---: | ---: |
| Original generated values | 12/12 checks pass | 12/12 checks pass |
| Both matrices scaled 32 times | 0/12 checks pass | 0/12 checks pass |
| Lowest matched fraction at 32 times scale | 79.3302% | 85.7767% |
| Largest absolute error at 32 times scale | 0.039795 | 0.035400 |

Each workload requires 99% of values within its original absolute tolerance of 0.0031 to 0.0036 plus relative tolerance 0.00001. Adding the low-by-low product improves accuracy, but cannot recover the information discarded by rounding the residuals to BF16. The same 32 times weight check previously passed with the frozen `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` bridge.

The following exploratory latencies are microseconds for output projection with N=64 and K=3072. Each cell uses the faster of at most two tested heuristic configurations. The BF16x9 comparison uses the same explicit compute mode as the frozen bridge and applies bias within the native cuBLASLt epilogue. The products column includes bias but excludes conversion; the complete column includes both operand conversions, every product, and bias.

| M | BF16x9 + bias | Three products | Three complete | Four products | Four complete |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 15.296 | 31.792 | 35.984 | 63.295 | 78.655 |
| 1024 | 34.000 | 24.688 | 38.879 | 37.344 | 53.952 |
| 8192 | 105.327 | 40.336 | 128.559 | 50.624 | 138.400 |

Conversion alone took 4.048, 13.824, and 89.727 microseconds respectively. Fusing the normalization output into fragment generation could remove a material cost on the largest shape, but the two-fragment representation still fails numerical validation. The optional modulation timing sweep was skipped after this failure. No complete solution was submitted or scored.

Timings use the pinned evaluator's shifted tensor addresses, cold L2 cache, and CUPTI measurement with two warmup and eight measured iterations. The GPU ran with unlocked clocks. PID 260355 was visible at the initial snapshot and had exited by the final snapshot; the latter showed only this experiment's PID 276522. These snapshots do not establish that the measurements were free of transient contention. They are unsuitable for leaderboard qualification.

Reproduce the bounded experiment from the repository root:

```bash
source tools/native_env.sh
python kernels/036_flux_output_norm_projection_chain/variants/bf16_expansion/experiment.py \
  --workloads 0,1,3,15 --shapes output --max-candidates 2 --rep 8
```

The script acquires the shared lock before importing Torch and loading CUDA code. An outer `flock` process retains that lock until the CUDA child exits. The extension is built from an immutable source snapshot. [The report](reports/expansion-output-20260906T194804Z.json) records exact algorithms, source hashes, all 48 correctness checks, and all timings. [Validation metadata](validation.json) records the archive hashes and rejection reason.
