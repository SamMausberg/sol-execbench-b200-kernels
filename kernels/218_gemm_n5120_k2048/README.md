# FP16 GEMM investigation

This investigation established an exact `torch.mm(A, B.T, out=C)` comparison and
tested small-M GEMV, split-K tensor-core, and cuBLASLt heuristic candidates.
The evidence did not establish a strong candidate for the hosted title.

The library comparison passed all 25 pinned workloads. Its raw local SOL score
was 0.687683, while the refreshed hosted leader was 0.535630. Four large-M local
measurements were below the official SOL estimates, giving uncapped workload
scores above one. This occurred with the exact library operation and unlocked
SM clocks, so the apparent leaderboard margin is not evidence of an optimization.

The 84 small-M candidates all met the pinned correctness tolerance. At M=1,
the custom GEMV took 6.560 us versus its paired library comparison at 7.936 us.
At M=6, 34, and 172, the tested GEMV or split-K alternatives were slower than the
library. A separate 48-candidate cuBLASLt sweep also passed correctness. It found
about 11% at M=34 and 4% at M=172; other sampled improvements were at most 2%.

The official library run is
`.work/runs/218/20260906T181956.072932Z-fp16-gemm-library-metadata-adapted-77dc66487fff`.
It preserves the original definition and contract. The dataset's empty `hf_id`
requires the documented `null` metadata adaptation to load in the pinned
evaluator; both definition hashes and the exact change are archived in the run.

Tuning artifacts are `.work/tuning/218-small-m.json` and
`.work/tuning/cublaslt/20260906T182058Z-float16-n5120-k2048.json`.
The hosted snapshot is `.work/tuning/218-leaderboard-20260906.json`.
`experimental.solution.json` packages the initial custom experiments; it has
not completed full official validation. `library.solution.json` is the validated
comparison, and no custom candidate has been selected for submission.

```bash
source tools/native_env.sh
python tools/campaign.py bench 218 \
  --solution kernels/218_gemm_n5120_k2048/library.solution.json \
  --label fp16-gemm-library-metadata-adapted
```
