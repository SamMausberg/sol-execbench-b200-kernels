# Triton Q/K RMSNorm variant

The packaged kernel processes eight contiguous 128-element rows per CTA, with
separate grid planes for Q and K. Squares, normalization, and multiplication by
the weights remain FP32. FP32 fusion is disabled to preserve the reference's
separate arithmetic operations. Each output is written into the supplied buffer.

On 2026-09-06, all 16 pinned workloads passed in each of three official evaluator
trials. The local mean SOL score from median workload latencies was 0.596559;
individual trial scores were 0.596352, 0.598160, and 0.596184. The largest workload
took a median 119.056 us. The maximum workload spread across trials was 4.02%.

These are local measurements with unlocked clocks. After each trial the GPU
reported 1965 MHz SM and 3996 MHz memory; the official SM preset is 1500 MHz.
The measurements do not establish an official rank or an advantage over the
existing CUDA implementation, whose two-case local comparison was similar.

The exact validated package SHA-256 is
`ddc923a62d73f468338e7a9af5b1f6825b931f537dcc1a086d0c96d0c23d615c`.
Raw traces, exact sources, package, clock snapshots, and score calculations are
under `.work/runs/38/20260906T174415.031174Z-triton-v2-final-ddc923a62d73`.

To reproduce from the repository root:

```bash
source tools/native_env.sh
python tools/campaign.py bench 38 \
  --solution kernels/038_flux_multi_head_rmsnorm_qk/variants/triton_v2/solution.json \
  --trials 3 --label triton-v2-final
```

`tune.py` measures candidates with the evaluator's input shifting and CUPTI
timer while holding the shared GPU lock. The 108-configuration first sweep and
38-configuration focused sweep are saved in `.work/tuning/38-v2.json` and
`.work/tuning/38-v2-focused.json`. Packed multiplication, split-row reductions,
dual-Q/K CTAs, and persistent scheduling gave no repeatable improvement over the
eight-row grid. Their experimental sources are retained for reproduction and
are excluded from `solution.json`.
