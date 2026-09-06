# Attention value projection with transpose

`kernel.py` computes the complete BF16 value projection with `torch.nn.functional.linear`, then returns a standard `torch.Tensor` view with axes `[batch, 8, sequence, 128]`. This removes the reference's output copy. It uses the actual hidden states and projection weight on every invocation.

The output owns its computed values through the projection tensor's storage. It has strides `(sequence * 1024, 128, 1024, 1)` and is usually noncontiguous. No computation is deferred and no input storage is modified.

## Output contract

The [problem definition](https://research.nvidia.com/benchmarks/sol-execbench/kernels/10) specifies output shape, dtype, and numerical values, with no output stride or contiguity requirement. The [definition schema](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/docs/definition.md) describes the reference as the mathematical specification and lists tensor shape, dtype, and description fields.

The current [submission guide](https://research.nvidia.com/benchmarks/sol-execbench/blog/submission-guide) explicitly permits `destination_passing_style: false` and requires concrete tensors. Pinned `driver/templates/eval_driver.py` implements the return convention and validates exact `torch.Tensor` type, shape, dtype, and numerical values. It does not require matching strides. The manifest declares this return convention explicitly.

There is a documentation inconsistency: examples in the pinned `docs/solution.md` label the return convention as disallowed, while the current website guide, schema, and evaluator support it. This candidate follows the current website guide and executable contract. If an application requires contiguous attention tensors, a downstream copy would still be required; that is not part of this problem's declared output contract.

## Evaluation

All 16 official workloads passed three full trials. The local score using the median latency per workload is **0.635502**, compared with the fresh B200/v1.1 leader score **0.534899** for RIAC as well. Individual trial scores were 0.635850, 0.635080, and 0.635595. The geometric mean latency is 20.813 microseconds, and the largest workload latency spread is 1.733%.

Thirty additional cases use three random seeds, five shapes, and two input revisions. They match the reference exactly and verify that returned tensors retain their own computed values after subsequent invocations. Exact source, package, environment, and result hashes are in `validation.json`.

This is a local leader candidate. A uniform latency increase of 17.54% would erase the lead. A hypothetical fully compute-bound slowdown of `1965 / 1500` would reduce the estimated score to 0.478646. This is a sensitivity calculation, not a measured clock correction. Official clock behavior and ranking remain unverified, so a hosted submission is not yet recommended.

Use the repository's pinned evaluator:

```bash
source tools/native_env.sh
python tools/campaign.py bench 10 \
  --solution kernels/010_attention_value_projection_with_transpose/solution.json \
  --trials 3
```

Local results use unlocked Runpod clocks because the host denies clock locking. They do not establish an official leaderboard position.
