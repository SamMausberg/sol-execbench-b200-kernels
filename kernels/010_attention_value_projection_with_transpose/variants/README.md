# cuBLASLt experiments

`cublaslt_bridge.cpp` implements contiguous row-major `A[M,K] @ B[N,K].T` for FP16 or BF16 inputs, with FP32 accumulation and an output of the input dtype. It accepts fresh input, output, and byte workspace tensors on each call. Its caches contain only device handles and shape, dtype, workspace, and algorithm metadata.

`cublaslt_loader.py` hashes the C++ source, saves an immutable copy under the persistent Torch extension cache, and compiles a uniquely named module. It exposes the source hash, source path, and cuBLASLt version alongside these entry points:

```python
bridge.algorithms(a, b, c, workspace_limit_bytes)
bridge.matmul(a, b, c, workspace, algorithm_index)
bridge.matmul_config(a, b, c, workspace, configuration)
```

The explicit configuration order is `id, tile, split_k, reduction, swizzle, custom, stages, inner_shape, cluster_shape`. Use all nine fields to preserve the Blackwell cluster choice. cuBLASLt validates the algorithm and required workspace before execution. The bridge uses the current Torch CUDA stream and provides the default GEMM epilogue only.

`tune_cublaslt.py` compares the heuristic candidates with `torch.mm(out=...)`. It defaults to the pinned evaluator's shifted input and output pointers and cold L2 timing. `--atol` and `--rtol` select the problem's tolerance. The standalone script acquires `SOL_GPU_LOCK`; it should not be wrapped in another acquisition of the same lock.

`select_cublaslt.py` verifies the explicit configurations chosen for problem 10. `cublaslt_v2` packages the resulting candidate for the official C++ evaluator. Python compilation through the loader is an experiment helper; it is not called by the submitted solution.
