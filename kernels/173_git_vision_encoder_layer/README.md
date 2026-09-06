# GIT vision encoder exploration

This FP32 encoder uses seven kernels: layer normalization, a combined QKV projection, online attention, output projection with the residual, a second layer normalization, a linear projection with Quick GELU, and the final linear projection with the residual. Every invocation reads the actual input tensors, biases, normalization parameters, and epsilon. The output uses destination passing.

The current schedules are uncompetitive. The canonical TF32x3 implementation and several precision variants pass the two selected correctness workloads, `batch=2, sequence=197` and `batch=64, sequence=197`. The full problem has 16 workloads; full validation has not been completed. `validation.json` records the exact packages, source, environment, traces, and partial results. None of these observations establishes a full problem score or an official rank.

## Matrix precision findings

| Product implementation | Selected correctness results | Finding |
| --- | --- | --- |
| Built-in TF32x3 | 2/2 pass | The pinned compiler inserts TMEM transfers and waits between decomposed products. |
| Built-in BF16x3 | 2/2 pass | Its modest speed improvement does not resolve the synchronization cost. |
| Explicit BF16 high/low accumulation chain | 2/2 pass | The compiler still inserts intermediate TMEM transfers. |
| BF16 chain with `arch="sm80"` | 2/2 pass | Uses register accumulation through `mma.sync` in a valid SM100 binary, but remains slow. |
| Ordinary TF32 | 0/2 pass | Maximum absolute errors are 0.009523 and 0.010644. |
| TF32 with explicit operand rounding | 2/2 pass | Maximum absolute errors fall to 0.002465 and 0.003050. |

The rounded TF32 helper in `variants/kernel_tf32_rna.py` applies `cvt.rna.tf32.f32` to each operand before the dot product. The generated ordinary TF32 code omitted this conversion. Explicit rounding removes enough error to pass these two cases while retaining one tensor-core product. That result is specific to the validated workloads and does not establish accuracy for every input magnitude.

Increasing the TF32x3 reduction tile from 32 to 256 worsened performance. A component profile shows that the GEMMs dominate GPU execution. The diagnostic CUPTI collector also adds substantial launch gaps, so its total span is not used as a benchmark latency. Other CUDA sessions overlapped some experiments; these timings remain exploratory.

## Reusable library experiments

`variants/fast_tf32_bridge.cpp` extends the problem 10 bridge with FP32 input and output tensors and the explicit per-operation `CUBLAS_COMPUTE_32F_FAST_TF32` mode. `variants/emulated_bf16x9_bridge.cpp` instead selects `CUBLAS_COMPUTE_32F_EMULATED_16BFX9`. Each experiment uses its own immutable source snapshot and compiled module. The bridges leave framework precision settings unchanged and cache only device handles and shape, dtype, workspace, and algorithm metadata.

`variants/check_fast_tf32.py` checks these modes against actual problem 36 inputs, with an optional weight scaling stress check, and records shifted-pointer component timings. These are reusable precision and scheduling experiments, separate from this encoder's partial kernel validation.

FAST_TF32 passes both sampled original problem 36 workloads, but fails most cases when both weight matrices are multiplied by 32. The emulation mode passes all original and scaled checks, with a smaller speed improvement. `variants/library-validation.json` records exact sources and reports. The broad input check supports using emulation when the mathematical contract requires FP32 accuracy across changing value magnitudes.
