# Problem 48: fused gate/up projection and GELU

The CuTe experiment passes all 16 official workloads and 11 changed-input checks. Its unlocked local score is 0.770943 versus the recorded leader's 0.719587, but only a 3.76% uniform latency increase erases that lead. This margin does not justify a hosted submission because local clocks exceed the hosted setting. The candidate is frozen after bounded tuning.

Despite the problem title, the actual reference uses GELU-tanh. Both BF16 matrix products are rounded before activation. The gate is converted to FP32 for GELU, rounded to BF16 again, and multiplied by the BF16 up projection. The implementation retains these rounding steps.

[dual.solution.json](dual.solution.json) selects [dual_kernel.py](dual_kernel.py) and [cute_dual.py](cute_dual.py). Each CTA reuses its input tile for the gate and up products. Both FP32 accumulations stay in tensor memory until the epilogue, which applies the rounded activation and product and stores only the final output. Small inputs use 128x64 tiles; larger inputs use 2-CTA 256x128 tiles. The wrapper uses the actual caller stream, reads every input on each invocation, and caches only compiled code and shape metadata. Strided inputs take an explicit PyTorch fallback.

One complete official harness trial passed 16/16 workloads with packet SHA256 `6837735b20ab162e42112cb789ce084cfee01e27930ecf00e4941e1b058a5f06`. Representative latencies were about 67 us at M128, 181 us at M1024, and 1.49-1.53 ms at M8192. Eleven separate checks covered M131, M1024, and M8192 with both weight matrices scaled by 1, 1/32, and 32, plus strided inputs and zero inputs. All passed the original workload tolerances.

The full run used the shared GPU lock. An attached CPU monitor recorded 133 samples with no foreign CUDA context or monitoring error, but it started after the campaign acquired its lock. This is not a qualification-wrapper run or evidence of hosted performance. The initial geometry sweep did observe a foreign CUDA process in 42 of 179 samples; those measurements remain exploratory. GPU clocks were unlocked throughout, and no hosted submission was made.

The bounded search tested six tile and cluster configurations at M128, M1024, and M8192. The initial 256-column configurations failed compilation because paired accumulators exceeded the 512-column tensor-memory allocation limit. Capping these configurations to one accumulator stage made them correct but slower. The final candidate keeps the faster double-buffered configurations. Raw tuning, changed-input checks, monitoring, and the full-run summary are preserved in [reports](reports), with hashes in [validation.json](validation.json).

Earlier prototypes are retained for inspection. [solution.json](solution.json) selects the paired Triton prototype, which passed two workloads but took 123 us at M128 and 10.38 ms at M8192. [cute.solution.json](cute.solution.json) selects a failed DSMEM ring prototype; its first official launch raised an illegal instruction. Neither is the frozen candidate. The pre-correction CuTe source for the initial tuning report is retained under [history/dual_initial](history/dual_initial).

Reproduce the frozen candidate from the repository root:

```bash
source tools/native_env.sh
python tools/campaign.py bench 48 \
  --solution kernels/048_fused_gate_up_projection_with_swiglu/dual.solution.json \
  --trials 1 --label dual-full
python kernels/048_fused_gate_up_projection_with_swiglu/validate_dual.py
```

The campaign acquires the shared lock internally. The standalone validation script acquires it before CUDA imports and retains it until its CUDA process exits.
