# Video latent GELU

`solution.json` selects `gelu_fast.py`, which fuses the reference's FP32 tanh GELU
expression into one pass. Launch sizes were chosen with cold-L2 CUPTI measurements.
The original libdevice implementation remains in `kernel.py` for comparison.

The selected package passed all 16 official workloads in three full local trials.
Its local SOL estimate is 0.630734, below the 0.652275 public leader snapshot.
The GPU's SM clocks are unlocked. See `validation.json` and the archived results
under `results/2026-09-06/25` for hashes, latencies, and environment details.

The native tanh instruction is approximate; this source targets the benchmark's
specified numerical tolerance. NVIDIA documents the instruction in the
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh).
