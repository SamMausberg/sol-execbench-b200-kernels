# FP32 projection with shared-memory BF16 fragments

This bounded component experiment passes the original problem 36 tolerances with original and 32-times weights. Its best complete path takes 111.04 us at M8192, N64, K3072, compared with 105.36 us for the same-run BF16x9 library path. The experiment is frozen because it still trails the library. It is not a complete problem 36 solution or a submission candidate.

The kernel reads actual FP32 input values and forms three successive BF16 residual fragments directly in shared memory. Six tensor-core products reconstruct the useful triangular terms:

```
A0*B2 + A1*B1 + A2*B0 + A0*B1 + A1*B0 + A0*B0
```

The grid splits K into four chunks of 768 elements. Each CTA converts only its current input tile; global fragment buffers are unnecessary. Two shared-memory stages connect cooperative conversion threads to the TCGEN MMA instructions through `PipelineAsyncUmma`. The accumulations remain in tensor memory until the epilogue. A second CuTe kernel reduces the partial results and adds the actual FP32 bias. Both kernels launch from one compiled entry on the caller's stream.

Three implementations were measured, preserving each prior source and its report:

| Variant | Accumulation | Shared stages | Complete latency | Same-run BF16x9 |
| --- | --- | ---: | ---: | ---: |
| [kernel.py](kernel.py) | Six independent products | 2 | 124.560 us | 105.344 us |
| [kernel_two.py](kernel_two.py) | Five lower terms together; dominant term separate | 2 | 111.040 us | 105.360 us |
| [kernel_two_stage1.py](kernel_two_stage1.py) | Five lower terms together; dominant term separate | 1 | 114.143 us | 106.176 us |

The two-accumulator schedule reduces partial storage from 50,331,648 to 16,777,216 bytes and the allocated tensor-memory columns from 512 to 128. It accumulates the lower terms from smaller to larger within each K instruction block and retains the dominant product separately. Changing this grouping required fresh numerical validation. The one-stage follow-up reduced operand shared memory from 147,456 to 73,728 bytes to test a possible occupancy improvement; it was slower, ending the bounded investigation.

Every variant passed all six end-to-end checks: the three M8192 workloads with B/S of 32/256, 2/4096, and 16/512, each at weight scales 1 and 32. The reference supplies the exact normalized and modulated FP32 activation, and only its output projection is replaced. Across 18 checks, the minimum matched fraction was 99.999809% and the largest absolute error was 0.011474609375. These satisfy the original 99% requirement with each workload's original absolute tolerance and relative tolerance of 1e-5. Both weight matrices are scaled for the stress cases. No precision flags were changed.

The first variant's separate profile measured roughly 113 us for conversion/products, 8 us for reduction, and 1.2-2.5 us between kernels. Its compiled resource report shows 209 registers with zero stack or local memory. The PTX contains six TMEM loads after accumulation, no TMEM stores, and 24 static MMA instructions for the six products over a 64-element K tile. This identifies GPU work as the limiting cost in this implementation. Reviewable PTX and the resource dump are in [reports/compiler](reports/compiler); the generated binary is retained locally with its checksum in that directory's manifest.

Timings use the pinned shifted-address CUPTI method with cold L2, two warmup iterations, and ten measured iterations. The BF16x9 comparison evaluates two recorded library heuristics and records both. The table reports the faster one. The three processes recorded 19, 14, and 14 CPU monitor samples respectively, with no observed foreign CUDA context or monitoring error. Sampling cannot exclude activity between observations, and these are exploratory component measurements with unlocked clocks. No official harness score, hosted submission, or leaderboard claim is associated with this experiment.

The earlier global-fragment implementation measured about 175 us for the complete large path. Shared-memory conversion removes much of that cost, but further integration and scheduling work would be needed to beat the library. This Python/CuTe component also cannot be directly inserted into the existing pure CUDA_CPP package under the evaluator's mixed-language restriction.

Reproduce the best variant from the repository root:

```bash
source tools/native_env.sh
python kernels/036_flux_output_norm_projection_chain/variants/bf16_expansion6_cute/experiment_two.py
```

The other two experiments use `experiment.py` and `experiment_two_stage1.py`. Each script acquires the shared GPU lock before CUDA imports, retains it until its CUDA child exits, and has a 240-second child timeout. Every invocation overwrites its partial workspace and reads current input values. Cached state contains only compiled code. [validation.json](validation.json) records source and report hashes; exact source snapshots are retained under [history](history).
