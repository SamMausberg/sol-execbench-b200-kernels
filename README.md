# SOL-ExecBench #38

CUDA C++ solution for `038_flux_multi_head_rmsnorm_qk`.

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

## Package

```sh
make package
make verify-package
```

Run `make verify-package` on B200. Upload the generated JSON from `dist/` to
kernel #38. Submission and publication are manual.

Apache-2.0. See `LICENSE`.
