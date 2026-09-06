# FP32 modulation projection investigation

This experimental Triton kernel fuses the 3072-to-18432 projection and bias,
then returns six fresh tensor views. It reads the actual input values and
strides. It is parked because the measured larger cases are substantially
slower than the scoring baseline.

The initial package passes three official workloads (batch sizes 5, 131, and
919). Its exact source and manifest are preserved in `history/initial/`.
The current source exposes the warp count for a bounded register-pressure
comparison; it has not completed all 16 official workloads.

All 16 additional comparisons pass the pinned correctness checker: four
configurations at batch sizes 131 and 919, with original inputs and with both
projection operands multiplied by 32. These checks cover the projection
numerics, not every possible layout or output-ownership condition.

The initial large tile uses 255 registers and spills 744 bytes per thread.
Smaller K tiles eliminate the measured local load/store instructions, but the
best timings remain about 512 microseconds at batch 131 and 2424 microseconds
at batch 919. Their scoring baselines are about 94 and 170 microseconds.
Eliminating spills alone does not make this instruction schedule competitive.

`arch="sm80"` selects register-based MMA lowering; the actual compiled binary
still targets SM100a. The archived compiler metadata and resource reports
record both settings. The standalone tuner observed no foreign CUDA context
in 15 samples and no query errors. These exploratory timings are not qualified
full-problem ranking measurements, and the host clocks remain unlocked.

Evidence and exact hashes are in `validation.json` and
[`results/2026-09-06/35/archive-index.json`](../../results/2026-09-06/35/archive-index.json).

To repeat the bounded comparison from the repository root:

```bash
source tools/native_env.sh
flock .work/gpu.lock python kernels/035_flux_ada_layer_norm_zero_modulation_extraction/tune.py --output .work/tuning/35-register-pressure-repeat.json
```

No hosted submission is recommended for this experiment.
