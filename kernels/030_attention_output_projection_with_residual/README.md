# Projection with residual on B200

The selected experimental implementation uses a persistent CuTe matrix product
and adds the residual in its epilogue. It explicitly rounds the accumulated
projection to BF16 before adding the residual, preserving the reference's
intermediate rounding. Every call reads the supplied activations, weights and
residual. Only compiled kernels and shape metadata are cached.

All 16 official workloads passed in one full evaluator trial. The 32 additional
checks also passed every element's tolerance, including fresh values in reused
storage, exact cancellation, scaled inputs, zero input/weights, and strided
inputs and outputs. The exact embedded package SHA256 is
`8c957f454449047e06f8c3d06bff7493e1eb2107c29b113ea91e562eeab73324`.
The authoritative hash and full provenance are in `validation.json`.

The observed full-run SOL score was 0.498701, below the leader snapshot of
0.546380. Its timing is unqualified: clocks were unlocked, the pre-run snapshot
contained another CUDA context, and no continuous process monitor was active.
Part of the extended tuning sweep also overlapped another session. This is a
correctness checkpoint and is not recommended for a hosted submission.

The immutable run and input checks are archived under
`results/2026-09-06/30/20260906T183657.633913Z-cute-fused-full-initial-8c957f454449/`.
The launch uses 64x64 tiles for small row counts, 128x128 for medium counts, and
256x256 tiles with two CTAs for larger counts. The input and output descriptors
handle row tails; noncontiguous tensors use the exact library fallback.

`cute_residual.py` adapts NVIDIA's BSD-3-Clause
[CUTLASS 4.4.1 persistent GEMM example](https://github.com/NVIDIA/cutlass/blob/v4.4.1/examples/python/CuTeDSL/blackwell/dense_gemm_alpha_beta_persistent.py).
Its copyright and license are retained. The numerical change is the BF16 cast
before residual addition; its Blackwell scheduling comes from the upstream example.

The other files preserve bounded experiments. `triton.solution.json` selects the
initial Triton kernel, which passed three smoke workloads but was slow. Automatic
Triton warp specialization failed in the pinned compiler, while ordinary TMA
schedules compiled. `lt/` compares library algorithms with the residual addition
launched immediately from C++ to reduce Python dispatch gaps. Neither alternative
demonstrated a sufficiently large improvement to justify a hosted submission.

```bash
source tools/native_env.sh
python tools/campaign.py bench 30 \
  --solution kernels/030_attention_output_projection_with_residual/solution.json
python kernels/030_attention_output_projection_with_residual/verify_inputs.py \
  --output .work/validation/30/cute-inputs.json
```

Both entry points acquire the shared GPU lock. Use `tools/qualify_gpu.py` for
performance qualification when other sessions share the GPU.
