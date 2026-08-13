# SOL-ExecBench B200 kernels

CUDA C++ and CuTe DSL solutions for:

- `029_mamba_conv1d_with_gating`
- `038_flux_multi_head_rmsnorm_qk`
- `003_fp8_mlp_gate_up_projection` (SOL problem 179)

Each source submission lives under its matching `kernels/` subdirectory.

## Requirements

- Git and Make
- Docker with NVIDIA Container Toolkit
- NVIDIA driver 580 or newer
- NVIDIA GPU

## Setup

```sh
git submodule update --init --recursive
make setup
```

`make setup` builds the pinned evaluator image and downloads problem #38. The
benchmark data stays under `.work/` and is not committed.

## Local

```sh
make info
make lint
make test
make bench
make profile WORKLOAD=0
```

Local timings are not official B200 SOL Scores.

## B200

```sh
make test TARGET=b200
make bench TARGET=b200
make status
```

Problems 29 and 179 use the `extra-*` targets with `KERNEL_ID=29` or
`KERNEL_ID=179`, for example:

```sh
make extra-test KERNEL_ID=179
make extra-compile KERNEL_ID=179 TARGET=b200
make extra-package
```

## Package

```sh
make package
make verify-package
```

Run `make verify-package` on B200. Kernel #38 is generated under `dist/`;
problems 29 and 179 are generated under `.work/extra-kernels/submissions/`.
Upload the matching JSON to NVIDIA. Submission and publication are manual.

Apache-2.0. See `LICENSE`.
