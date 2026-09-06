# Attention projection backward

The selected candidate is `variants/cublaslt/solution.json`. It passes all 16
official workloads and 40 additional checks, including changed amplitudes,
zeros, basis inputs, strides, padding, unaligned pointers, and output ownership.
It is parked pending a stronger performance margin. No submission is recommended.

Two cuBLASLt GEMMs read the actual BF16 operands and accumulate in FP32 to
produce BF16 input and weight gradients. Returning the fresh input-gradient
buffer as a concrete attention-layout view removes the reference's transpose
copy. The wrapper allocates both outputs and workspace before either GEMM.
Descriptions and algorithms are cached by device and shape; tensor values and
workspaces are never cached. The current CUDA stream is used throughout.

The original ATen version passes all official workloads but fails one unaligned
input case in the additional audit. The selected wrapper creates aligned dense
copies when required and explicitly selects FP32 accumulation. Its 40-case
audit includes the previously failing case. A bounded search checks 96 component
configurations. Only the 512-row shape shows a clear enough improvement to
change the default algorithm pair; the remaining shapes retain index zero.

The selected package is
`ba1d1e73cc9148adb99a08cb47771decd1670b3aef7982318d2b6ad3c3e0f1e1`.
Its raw SOL score is 0.799523 versus leader snapshot 0.753779. The process monitor
records 14 foreign CUDA context observations during that full trial, so this
timing is excluded from ranking qualification. Its unqualified uniform latency
buffer is 9.44%, which does not establish enough margin for the host's different
clock setting. No hosted submission has been made.

`validation.json` links the exact package, numerical audits, and
[persistent archive](../../results/2026-09-06/4/archive-index.json).
Earlier ATen and default-cuBLASLt implementations remain as experiments;
`variants/cublaslt/history/default` preserves the source used by the component
search. The failed initial audit is retained with its original verification script.

After sourcing `tools/native_env.sh`, run the selected additional audit with:

```bash
flock .work/gpu.lock python kernels/004_attention_output_projection_with_reshape_backward/verify.py --variant cublaslt --output .work/validation/4-cublaslt-selected-inputs.json
```
