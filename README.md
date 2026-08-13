# SOL-ExecBench B200 Kernels

CUDA and CUTLASS solutions for NVIDIA SOL-ExecBench, with architecture-specific
paths for B200 and reproducible submission artifacts.

## Kernels

| Leaderboard problem | Definition | Source |
| --- | --- | --- |
| 29 | `029_mamba_conv1d_with_gating` | `kernels/029_mamba_conv1d_with_gating/` |
| 38 | `038_flux_multi_head_rmsnorm_qk` | `kernels/038_flux_multi_head_rmsnorm_qk/` |
| 179 | `003_fp8_mlp_gate_up_projection` | `kernels/179_fp8_mlp_gate_up_projection/` |

Problem 179 intentionally uses definition prefix `003`; the leaderboard ID and
the collection-local definition index are different identifiers.

## Submission files

Upload exactly one self-contained JSON to each NVIDIA problem:

| Problem | File |
| --- | --- |
| 29 | `dist/029_mamba_conv1d_with_gating-b200.json` |
| 38 | `dist/038_flux_multi_head_rmsnorm_qk-b200.json` |
| 179 | `dist/003_fp8_mlp_gate_up_projection-b200.json` |

Each JSON embeds its source files. The adjacent `.sha256` file identifies the
exact candidate used for a leaderboard run.

## Reproduce and validate

Requirements: Git, Docker, NVIDIA Container Toolkit, driver 580 or newer, and
an NVIDIA GPU. The evaluator and benchmark datasets are revision- and
checksum-pinned.

```sh
git submodule update --init --recursive
make setup
make lint
make extra-lint
make extra-verify-package
```

Local correctness checks:

```sh
make test
make extra-test KERNEL_ID=29 TARGET=local
make extra-test KERNEL_ID=179 TARGET=local
```

B200 validation before submission:

```sh
make test TARGET=b200
make extra-test KERNEL_ID=29 TARGET=b200
make extra-test KERNEL_ID=179 TARGET=b200
```

Local RTX timings are validation data, not B200 SOL scores. Native SM100
correctness and leaderboard performance must be established on B200.

Apache-2.0. See `LICENSE`.
