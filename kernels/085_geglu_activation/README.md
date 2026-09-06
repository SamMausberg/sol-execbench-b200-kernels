# GEGLU activation

The selected kernel fuses split-channel loads, tanh GELU, and multiplication by
the other channel half. Smaller inputs use row tiles and an exponential identity
for tanh; larger inputs use a flat launch and libdevice tanh. The launch choices
were measured with the evaluator's cold-L2 CUPTI timing helper.

All 16 official workloads passed three complete local evaluator trials. The local
SOL estimate is 0.656663, below the 0.680444 public leader snapshot. SM clocks are
unlocked. See `validation.json` and `results/2026-09-06/85` for exact source hashes,
packages, and results. The native tanh alternative in the development kernel is
not selected by `run`.
