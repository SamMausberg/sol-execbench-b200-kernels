# Six-product BF16 expansion for problem 36

The three-fragment expansion passes the sampled numerical checks when the dominant product uses split-K=4. Its complete conversion path is slower than BF16x9 on all three measured output sizes, so this remains a reusable experiment rather than a submission candidate.

Each actual FP32 operand is split into three successive BF16 residuals: `x0 = BF16_rn(x)`, `x1 = BF16_rn(x - float(x0))`, and `x2 = BF16_rn(x - float(x0) - float(x1))`. The implementation evaluates six terms, ordered from the smallest residual level to the dominant product:

```
A0*B2 + A1*B1 + A2*B0 + A0*B1 + A1*B0 + A0*B0
```

The sequential schedule performs six GEMMs into FP32. The batched schedule stores input planes as `[A0,A1,A2]` and weight planes as `[B2,B1,B0]`. One strided batched GEMM computes the three smallest products, another computes the next two, and a CUDA kernel sums their five FP32 result matrices. The final GEMM adds `A0*B0` with beta=1 and the actual FP32 bias. Its split-K implementation can itself launch a reduction kernel; the four native calls therefore do not imply exactly four GPU kernels.

The initial schedule without explicit splitting passed 23 of 24 end-to-end checks, failing the combined chain at batch 32 and sequence length 256 with weights scaled 32 times. Only 98.5046% of values matched the unchanged tolerance, below the required 99%. Split-K=4 on the final, dominant GEMM raised that case to 99.9931%. No other split setting was needed or measured.

Validation then covered all 16 pinned workloads, original weights and both matrices scaled 32 times, sequential and batched schedules, and replacement of the modulation GEMM, output GEMM, and both GEMMs. All 192 checks passed. The minimum matched fraction was 99.9931%, and the largest absolute error was 0.014160. Every check met the original requirement that at least 99% of values satisfy the combined absolute and relative tolerance. The test keeps the reference normalization and modulation operations unchanged to isolate the GEMM contribution. These are standalone numerical checks, not official harness trials or a full problem score.

The following exploratory latencies are microseconds for output projection with N=64 and K=3072. Every column includes bias. Complete paths also convert both operands on each invocation.

| M | BF16x9 | Six sequential products | Sequential complete | Batched products | Batched complete |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 15.312 | 133.567 | 158.431 | 31.616 | 47.455 |
| 1024 | 34.016 | 92.944 | 111.984 | 44.320 | 62.863 |
| 8192 | 105.728 | 98.016 | 184.960 | 85.327 | 174.655 |

Conversion alone took 4.368, 14.720, and 94.384 microseconds. Only the largest product-only case has useful headroom, about 20 microseconds. Fusing normalization directly into the three BF16 planes would be necessary to remove the separate conversion cost; its extra plane writes and integration costs remain unmeasured. No broader timing sweep was run.

Timing used at most two cuBLASLt heuristics per component, two warmup iterations, eight measured iterations, shifted input/output addresses, cold L2, and the pinned CUPTI method. The numerical checks use heuristic 0 for the smaller products and explicit split-K=4 for the dominant product. Timing also examines heuristic 1 for the smaller products; its component errors are recorded, but it has not received all 192 end-to-end checks. The table shows the faster measured choice and is not a qualified candidate comparison.

The full validation/timing process began with no external CUDA context and ended with only its own PID 322966. Only before/after snapshots were recorded, so transient contention cannot be excluded. GPU clocks were unlocked and differ from hosted evaluation. No hosted submissions or ranking claims were made.

Both implementations use only the caller's CUDA stream. Every conversion, product, reduction, and bias operation reads the actual current input values. Cached state contains CPU shape descriptors and algorithm metadata. Workspaces, fragment planes, and partial results are supplied afresh or overwritten on every measured call; no input-derived GPU state is retained.

Reproduce the complete bounded check from the repository root:

```bash
source tools/native_env.sh
python kernels/036_flux_output_norm_projection_chain/variants/bf16_expansion6/experiment.py \
  --workloads 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --shapes output --bias-split-k 4 --max-candidates 2 --rep 8
```

An outer `flock` process acquires the shared GPU lock before Torch import and extension loading, then retains it until the CUDA child exits. Source snapshots are immutable and addressed by their hash. [The full report](reports/expansion6-output-20260906T201205Z.json) records algorithms, timings, and all 192 checks. [Validation metadata](validation.json) includes hashes for the initial failure, focused correction, and full check. The original failed source is preserved under [history/initial](history/initial/bridge.cpp); the earlier two-fragment experiment remains untouched in the sibling directory.
