# Configured cuBLASLt value projection

This candidate computes the complete BF16 projection with cuBLASLt and FP32 accumulation. Nine explicit algorithm configurations depend only on the number of input rows, `batch_size * seq_len`. The output is an ordinary tensor view with the required `[batch, 8, sequence, 128]` axes. The shape, dtype, value, and return convention follow the [output contract audit](../../README.md#output-contract).

`binding.cpp` allocates fresh output and scratch storage for every invocation and passes the actual input pointers to the GEMM. The bridge caches only CUDA handles and shape, dtype, and algorithm metadata. It uses the caller's current CUDA stream. The 128-row shape uses three partitions of the reduction dimension and 1,572,880 bytes of scratch. Other selected shapes need no scratch. The exact nine configuration fields and selection measurements are in `selected-algorithms.json`; cuBLASLt checks each explicit configuration before using it.

The submission is compiled through the official C++ binding, before evaluation starts. `cublaslt_bridge.h` is a frozen copy of the reusable [bridge](../cublaslt_bridge.cpp). Local experiments use an immutable copy of that source selected by its SHA-256 hash, so later edits cannot alter an already compiled experiment.

## Validation and timing qualification

All 16 workloads passed nine complete official harness trials. Thirty additional cases change both inputs, reuse their storage, and check that earlier outputs keep their computed values. They use three random seeds and five shapes, including the partitioned reduction. At least 99.99695% of elements matched under the problem's absolute and relative tolerances. The extra checks call the exact packaged arithmetic core; the official trials exercise the complete C++ binding.

The one qualified timing trial scores **0.663578**, compared with 0.635502 for the first implementation and 0.534899 for the current B200/v1.1 leader. A bounded six-attempt run sampled CUDA process identities every 0.5 seconds. It qualified one trial and rejected five with foreign CUDA contexts during evaluation. The three earlier trials also lack clean timing qualification. Further uncontended repeats are required; no overlap shorter than the monitor's sample interval can be excluded. All observations and exact package, source, environment, trace, and monitor hashes are recorded in `validation.json`.

Runpod denies clock locking. The qualified local lead would tolerate a 22.67% uniform latency increase. A hypothetical fully compute-bound slowdown of `1965 / 1500` reduces the estimated score to 0.500409. That is a sensitivity calculation, not a measured correction. Hosted submission is not recommended until further timing repeats and official clock behavior establish sufficient margin.

```bash
source tools/native_env.sh
python tools/campaign.py bench 10 \
  --solution kernels/010_attention_value_projection_with_transpose/variants/cublaslt_v2/solution.json \
  --trials 3
```

The campaign acquires the shared GPU lock itself. Other benchmark sessions must use the same lock for the resulting timing qualification to be useful. The [CPU monitor and retry helper](../../../../tools/qualify_gpu.md) automates observation and preserves each qualified trial.
