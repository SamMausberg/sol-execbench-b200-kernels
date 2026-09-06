# MoE expert backward investigation

This is an experimental implementation, with no demonstrated leaderboard lead.
The pinned workload has 256 experts, hidden size 4096, intermediate size 2048,
and eight selected experts per token. Its FP32 weights total 24 GiB, and the
three dense parameter-gradient outputs require another 24 GiB.

The reference explicitly computes backward derivatives; it does not use
autograd. `kernel.py` rebuilds compact expert routing on the GPU, groups the
expert GEMMs, fuses activation derivatives, and writes each dense gradient
directly. It accumulates hidden gradients in ascending expert order. It also
uses `(G @ Wdown) · intermediate` for routing-weight gradients, and multiplies
route weights after that GEMM. These arithmetic reassociations require numerical
validation. Every invocation reads its current inputs and rebuilds its routing
and intermediate tensors.

`tma.py` additionally pads each expert's row count to a multiple of 32, gathers
input rows once, and uses tensor-map loads. Its current weight-gradient tile
must retain a reduction width of 32, matching that padding.

The first complete 2048-token workload passed the pinned evaluator. Its exact
package SHA256 was
`d7f1816a1bc91d0cf534156a9241f59bf98e32c758cc39a08676fe9565dc0bd0`.
The run is retained at
`.work/runs/119/20260906T184506.803662Z-moe-grouped-initial-d7f1816a1bc9`.
It took 99.33 ms, far above the stored 13.59 ms scoring baseline. Foreign CUDA
contexts were observed, so its timing is provisional. The evaluator worker
peaked near 150 GiB. Later source refactoring and compiler experiments have
separate hashes and are not that exact validated package.

Cold-cache CUPTI component profiles identified the three parameter-gradient
GEMMs as the largest costs: about 22 ms each initially. Larger tiles and TF32
backward products reduced these to about 6 ms each. BF16 backward products
reduced them to about 3.8 ms each. Forward projections and input gradients still
prevented a competitive end-to-end result. Tensor-map loads alone did not help.
Profiles are diagnostic: they use fixed addresses, unlocked clocks, and record
competing process identities. The timing span includes gaps between kernels.

Direct TF32 forward products failed routing-weight gradients on all three
reduced-expert checks with the real hidden dimensions. A three-product BF16
high/low decomposition passed those checks. Single BF16 backward products also
passed the pinned 99% matching requirement, with up to 15 unmatched elements
among 134 million. This does not establish an all-elements tolerance guarantee.
The dataset's misspelled `required_match_ratio` field is preserved; the evaluator
uses its default `required_matched_ratio=0.99`. No tolerance was changed.

The supported per-launch compiler option `arch="sm80"` selects register-based
`mma.sync` lowering while the generated binary remains `sm_100a` for this B200.
It eliminated TCGEN05 accumulator transfers and reduced the tested forward
projections from about 7.4 ms to 5.1 ms. Its full-hidden-dimension precision audit
passed all 15 output checks, including empty experts. Total component time
remained about 36 ms, so further broad tile sweeps were not justified.

Reproduce diagnostics after sourcing `tools/native_env.sh`:

```sh
python kernels/119_moe_expert_parallel_execution_backward/verify_small.py
python kernels/119_moe_expert_parallel_execution_backward/verify_precision.py \
  --module kernel.py --label mma-bf16 \
  --config '{"FORWARD_PRECISION":"bf16x3","BACKWARD_PRECISION":"bf16","PROJECT_TILE":[64,128,32,2],"WEIGHT_TILE":[64,128,32,1],"INPUT_TILE":[32,128,32,2],"GEMM_ARCH":"sm80"}'
```

`component_profile.py` accepts the same module/config options. Reports are in
`.work/tuning/119-{profile,precision}-*.json`; the initial reduced-shape audit is
`.work/tuning/119-small-verification.json`. Both experimental manifests package
only implementation sources. No full 16-workload validation or hosted
submission has been completed for this problem.
