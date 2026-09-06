# SiLU activation backward

One Triton pass computes the FP32 derivative from `grad_output`, `x`, and the
supplied `sigmoid_x` tensor. It preserves the reference arithmetic order.

All 16 official workloads passed in three full local trials. The local SOL estimate
is 0.527926, below the 0.560079 public leader snapshot. SM clocks are unlocked.
See `validation.json` and `results/2026-09-06/84` for the exact package and evidence.
