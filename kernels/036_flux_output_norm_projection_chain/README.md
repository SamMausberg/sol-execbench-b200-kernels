# Flux output normalization and projection

The current complete candidate is `cuda/solution.json`. It passes all 16 official
workloads and 16 additional input checks, but its observed runtime is below the
public leader's performance. It is an experimental checkpoint, with no submission
recommendation. See `validation.json` and the persistent
[result index](../../results/2026-09-06/36/archive-index.json).

One native C++ entry launches FP32 SiLU, a modulation projection with bias,
centered layer normalization with modulation, and an output projection with
bias. The projections explicitly select cuBLASLt's BF16x9 emulation of FP32.
Each invocation allocates fresh intermediate storage and uses the current CUDA
stream. The cache holds operation descriptions and algorithms, and each call
sets its actual bias pointers. Framework precision settings are not changed.
CUDA compilation disables FMA contraction to preserve the pointwise rounding.

The additional audit covers two shapes with random inputs, weights scaled by 32,
larger conditioning inputs, zeros, changed epsilon, near-constant rows, and
strided tensors. All checks preserve the inputs. This audit covers the recorded
layouts; it does not establish every possible pointer alignment.

The exact selected package is
`2a4e8533c652f4c6eab1275d871fd293499a5c4ada4ad0e28205a02be7e7d593`.
Its full trial has raw SOL score 0.352235 versus the refreshed leader's 0.644426.
Foreign CUDA contexts were present, so the timing is excluded from qualified
ranking comparisons. The Runpod host also cannot match the hosted 1500 MHz
clock setting. No hosted submission has been made.

Earlier root-level manifests preserve partial Triton and library experiments.
The `variants` directory contains bounded BF16 expansion studies. Three- and
four-product approximations fail scaled-weight checks. The six-product version
with four-way splitting of its dominant reduction passes 192 checks across all
official shapes, including scaled weights. Its full conversion path remains
slower than BF16x9. Both successful and failed numerical evidence are retained.

After sourcing `tools/native_env.sh`, evaluate the current candidate with:

```bash
python tools/campaign.py bench 36 --solution kernels/036_flux_output_norm_projection_chain/cuda/solution.json --trials 1
```

The evaluator takes the shared GPU lock. Standalone audit scripts should run
under `flock .work/gpu.lock` before importing CUDA libraries.
