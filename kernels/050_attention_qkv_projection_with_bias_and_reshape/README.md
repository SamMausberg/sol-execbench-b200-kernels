# QKV projection, bias, and reshape on B200

`solution.json` launches the CuTe kernel in `cute_kernel.py`. One persistent grid computes the query, key, and value projections using their separate weight and output descriptors. Each FP32 matrix product rounds to BF16 before the bias addition, matching the reference's separate matmul and addition. Bias values load directly into the epilogue registers; output stores use TMA.

The grid covers the logical 1536 output columns: 1024 for query, 256 for key, and 256 for value. This layout supplies scheduler coordinates. Each memory operation uses the corresponding tensor's actual descriptor and local column coordinate. The launch receives fresh tensor pointers on every call; its cache contains compiled code keyed by tensor metadata.

| Flattened input rows | MMA tile | CTA cluster |
| --- | --- | --- |
| Up to 1024 | 64 × 64 | 1 |
| 1025–2047 | 128 × 128 | 1 |
| 2048 or more | 128 × 256 | 1 |

The Triton fallback handles noncontiguous inputs, weights, biases, and outputs. The CuTe implementation derives from CUTLASS's BSD-3-Clause persistent GEMM example through this repository's problem 30 kernel; its NVIDIA copyright and license are preserved in `cute_qkv.py`.

The additional checks in `validation_edges.json` cover fresh random inputs, zero inputs, zero weights, noncontiguous tensors, and cancellation that requires rounding before the bias addition. They exercise the smallest and largest official shapes. Official correctness, timing qualification, source hashes, and the benchmark artifact path are recorded in `validation.json`.

All 16 official workloads passed in three full harness trials, and all ten additional cases passed. The observed scores were 0.70181809, 0.70807530, and 0.70269478. All three runs contained foreign CUDA contexts, so no timing trial is accepted. The first qualification session expired before an attempt; the renewed session completed two contaminated attempts and then expired waiting for an idle GPU before its third attempt. No ranking improvement is claimed.

Reproduce local qualification with:

```bash
source tools/native_env.sh
python tools/qualify_gpu.py 50 \
  --solution kernels/050_attention_qkv_projection_with_bias_and_reshape/solution.json \
  --trials 3 --max-attempts 3 --deadline 900
```

The `tune_*.json` files record the bounded experiments. Several include foreign CUDA contexts and their timings are excluded from performance claims. The separate grouped cuBLAS and direct output experiments are excluded from the submitted source set. No hosted submission has been made.
