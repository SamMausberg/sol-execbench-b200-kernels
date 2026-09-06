# MLA projection and split experiments

This candidate is parked after full correctness validation. Its raw local score
is 0.769533 against a 0.719510 public leader, but the uniform latency buffer is
only 6.87%. That margin is insufficient for a persuasive title candidate under
the different Runpod clocks and unresolved external GPU contention.

This problem projects BF16 hidden states through query and KV paths. The query
path includes RMSNorm and a second projection. The pinned reference copies four
split outputs; the output contract specifies their shapes and dtypes. These
experiments return ordinary views into fresh projected tensors, avoiding those
copies while computing every output from the current inputs.

`library.experimental.solution.json` uses three ordinary matrix products and a
Triton RMSNorm. The normalization preserves the reference's BF16 cast before the
learned weight multiplication and its final BF16 rounding. It passes all 16
official workloads in one full trial. The exact package
has SHA256 `29f431be5c14814f120b3733ff7c02347f10f1ca62143720ba460b71dbcc40a2`;
the run is `.work/runs/43/20260906T202003.011088Z-mla-library-full-initial-29f431be5c14`.
`validation.json` records the exact package, raw score, and audit hashes.

The 36 changed-input/scalar/layout checks pass the pinned tolerance for all 144
outputs; 130 outputs are bitwise equal to the reference. The other outputs have
at most 0.000883% of elements outside the per-element tolerance, below the
pinned 1% allowance. Different FP32 reduction ordering can change BF16 rounding.
Inputs and retained prior outputs remain unchanged. Mutating the first query
view leaves the other logical output regions unchanged. The report is
`.work/tuning/43-variation-library.json`, SHA256
`15e498360a6f10eabb989aac461ad9da81c3b10ed1d0fef7745ed9e8c7992ef1`.
Three additional calls place every input at an unaligned storage offset; all
12 outputs are bitwise equal and the ownership checks pass. Their separate
report is `.work/tuning/43-variation-library-unaligned.json`.

`cute.experimental.solution.json` groups the independent initial query and KV
products into one persistent Blackwell kernel. Separate TMA descriptors address
the real 1536-column query and 576-column KV weights and outputs. A padded virtual
column grid controls scheduling; actual descriptors handle the tail columns.
The implementation derives from problem 055's grouped CUTLASS 4.4.1 variant and
retains NVIDIA's BSD license. The inherited C arguments provide layout metadata
and are never loaded by the epilogue. The compiled-function cache stores only
code specialized for fixed shapes, layouts, and tile configurations. A layout
and alignment guard routes other inputs through the ordinary matrix products.

`tune.py` compares a bounded set of initial-projection tile configurations with
the pinned shifted-argument CUPTI timer. It records competing CUDA contexts using
process basenames and treats overlapped timings as diagnostic. Grouped CuTe is
faster for the initial small projection pair in isolation, but the full chain
has a large launch gap. Native ATen and direct cuBLASLt chains under
`variants/native` pass representative cases with similar overall latency to the
Python library chain. `tune_lt.py` found modest algorithm gains, insufficient to
remove the dominant final-GEMM cost.

The final-GEMM variants also pass their two representative workloads. Preparing
fresh execution arguments through the pinned CuTe public methods before the
first math kernel reduces launch overhead, but the packed variant remains slow
at 192.192 microseconds for 131 rows and 534.237 microseconds for 8192 rows.
Both descriptor and argument adapters remain owned by the current invocation;
the compiled-function cache retains no values or output buffers. These variants
are retained as experiments, and no further sweeps are planned for this target.

The fresh public v1.1 leader snapshot is `.work/tuning/43-leaderboard-20260906.json`:
RIAC as well at 0.719510. Runpod clocks remain unlocked and no hosted submission
has been made by this campaign.
