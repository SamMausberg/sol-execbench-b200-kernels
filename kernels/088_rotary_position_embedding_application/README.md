# Paired RoPE for query and key

This kernel rotates FP32 query and key pairs in one Triton launch. It shares cosine/sine loads where flattened Q/K positions align, writes each result once, and preserves separate multiplication/addition rounding. It supports the official head counts and sequence sizes without assuming cosine/sine values.

All 16 official workloads passed in three full runs of the pinned v1.1 evaluator. The combined local SOL estimate is **0.678962**; individual trial scores were 0.678655, 0.678903, 0.678897. Maximum workload latency spread was 1.32%. Clocks were unlocked, and the public leader snapshot was 0.688375, so this candidate is retained for coverage without a submission recommendation. Exact source, contract, environment, and trace hashes are in [validation.json](validation.json).

`solution.json` uses destination passing style. The entry point receives query, key, cosine, sine, query output, and key output in that order.

For additional geometry work, `tune.py` uses the official CUPTI helper with cold L2 and holds the shared GPU lock.
