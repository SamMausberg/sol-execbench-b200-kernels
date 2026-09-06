# SOL-ExecBench B200 kernels

CUDA C++, CuTe DSL, and Triton solutions for SOL-ExecBench.
Sources, manifests, and validation reports live in [kernels/](kernels/).

## Native Runpod setup

The checkout, environment, toolchain, datasets, and caches live under `/workspace`.
Run `bash tools/bootstrap_native.sh`, then `source tools/native_env.sh` in each shell.
See [native setup](docs/native-runpod.md), [strategy](docs/strategy-2026-09-06.md),
and [local validation results](results/2026-09-06/README.md).
The active [ten-leaderboard goal](docs/ten-leaderboards.md) tracks local leads and
confirmed hosted results separately.

```sh
python tools/campaign.py fetch 25 53 84 85 88
python tools/campaign.py bench 84 --solution kernels/084_silu_activation_backward/solution.json
python tools/campaign.py package 84 --solution kernels/084_silu_activation_backward/solution.json
```

GPU work uses a shared file lock. Local scores are estimates; this Runpod host
denies clock locking, and official evaluations fix B200 SM clocks at 1500 MHz.

## Docker setup

Requires Git, Make, Docker with NVIDIA Container Toolkit, and driver 580 or newer.

```sh
git submodule update --init --recursive
make setup
```

`make setup` builds the pinned evaluator image and downloads problem #38 into `.work/`.

```sh
make info
make lint
make test
make bench
make profile WORKLOAD=0
```

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
