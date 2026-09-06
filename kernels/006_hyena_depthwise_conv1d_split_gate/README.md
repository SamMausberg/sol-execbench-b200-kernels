# Hyena depthwise convolution and split gate

Problem #6 applies three causal depthwise filters to an FP32 tensor of shape
`[B, 768, S]`, splits its channels into `x0`, `x1`, and `v`, and returns
`v * x0`, `x0`, and `x1`. The implementation computes all three output tensors
in one launch, removing the full convolution intermediate and separate gate
launch.

Each filter has three taps. The accumulator starts with the actual channel
bias and applies valid taps in increasing kernel order using FP32 fused
multiply-adds. Padding taps are skipped. The final gate uses an FP32 multiply.
This preserves the operation order in PyTorch 2.9's
[native depthwise CUDA convolution](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/DepthwiseConv2d.cu).
The contiguous FP32 workload selects this implementation through
[PyTorch's convolution dispatch](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/Convolution.cpp).

The frozen CUDA implementation loads neighboring samples in groups of four
for aligned contiguous tensors, uses a scalar specialization for the odd
sequence length, and provides a general strided fallback. Its dispatch depends
on tensor shape, stride, and alignment. It reads all input values on every
invocation and honors noncontiguous input and output strides.
`cuda_local.py` is a local build helper; official CUDA packages compile through
the evaluator's normal first phase and do not include this helper.

The official contract has 16 workloads and uses maximum absolute and relative
error thresholds of `1e-5`. Development tuning reports record bitwise equality
in addition to the official error metrics. `bench_local.py` provides random,
zero-input, boundary-impulse, cancellation, and strided checks.

The frozen package passed all 16 official workloads in one full harness trial,
plus 30 additional input cases with bitwise equality. The trial's raw score
was `0.7728506720` and geometric mean latency was `6.553751 us`. Ten monitor
samples recorded unrelated CUDA contexts while the campaign held its lock;
the trial is excluded from performance qualification. The B200 clocks were
also unlocked. The saved leaderboard snapshot leads at `0.785398` and
`6.261 us`, so this kernel remains experimental and is not recommended for
submission. No hosted submission has been made.

`validation.json` records source and report hashes. The package SHA-256 is
`b8def70a5bf12632bceea7dc9b33b434b56174e76804273037d5b33a5ee6a03d`.
The official run is
`.work/runs/6/20260906T205100.594513Z-hyena-final-b8def70a5bf1`.
The retained `experiments_*.cpp` and `experiments_*.cu` files preserve the
development candidates and are excluded from the official source set.

All direct GPU commands must acquire the shared lock, including the local
extension build:

```bash
source tools/native_env.sh
flock /workspace/sol-execbench-b200-kernels/.work/gpu.lock \
  python kernels/006_hyena_depthwise_conv1d_split_gate/bench_local.py \
  .work/problems/6 --indices 13,9,4 --cuda \
  --variants random,zero_input,boundary_impulses,cancellation,strided \
  --output kernels/006_hyena_depthwise_conv1d_split_gate/variation.json
```

`tools/campaign.py` and `tools/qualify_gpu.py` acquire the lock themselves and
must be invoked without an outer `flock`.
