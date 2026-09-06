#!/usr/bin/env bash
# Source this file before running the native evaluator on Runpod.

export SOL_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SOL_WORKSPACE="${SOL_WORKSPACE:-/workspace}"
export SOL_VENV="${SOL_VENV:-$SOL_WORKSPACE/venvs/sol-execbench}"
export UV_PROJECT_ENVIRONMENT="$SOL_VENV"
export VIRTUAL_ENV="$SOL_VENV"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SOL_WORKSPACE/cache/uv}"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SOL_WORKSPACE/toolchains/python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SOL_WORKSPACE/cache}"
export HF_HOME="${HF_HOME:-$SOL_WORKSPACE/cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SOL_WORKSPACE/cache/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$SOL_WORKSPACE/cache/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$SOL_WORKSPACE/cache/cuda}"
export CUTE_DSL_CACHE_DIR="${CUTE_DSL_CACHE_DIR:-$SOL_WORKSPACE/cache/cute_dsl}"
export CCACHE_DIR="${CCACHE_DIR:-$SOL_WORKSPACE/cache/ccache}"
export TMPDIR="${TMPDIR:-$SOL_WORKSPACE/tmp/sol-execbench}"
export FLASHINFER_TRACE_DIR="${FLASHINFER_TRACE_DIR:-$SOL_REPO_ROOT/.work/flashinfer-trace}"
export CUTLASS_DIR="${CUTLASS_DIR:-$SOL_WORKSPACE/upstream/cutlass}"
export CUDA_HOME="${SOL_CUDA_HOME:-$SOL_WORKSPACE/toolchains/cuda-13.1/nvidia/cu13}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$SOL_VENV/bin:$SOL_WORKSPACE/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$SOL_REPO_ROOT/third_party/sol-execbench/src${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS="${MAX_JOBS:-8}"
export SOL_GPU_LOCK="${SOL_GPU_LOCK:-$SOL_WORKSPACE/sol-execbench-b200-kernels/.work/gpu.lock}"

for sol_site_packages in "$SOL_VENV"/lib/python*/site-packages; do
    [[ -d "$sol_site_packages" ]] || continue
    export CPLUS_INCLUDE_PATH="$CUTLASS_DIR/include:$CUTLASS_DIR/tools/util/include:$CUDA_HOME/include:$CUDA_HOME/include/cccl:$sol_site_packages/include:$sol_site_packages/nvidia/cu13/include:$sol_site_packages/nvidia/cudnn/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export LD_LIBRARY_PATH="$sol_site_packages/nvidia/cu13/lib:$sol_site_packages/nvidia/cudnn/lib:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$sol_site_packages/nvidia/cu13/lib:$sol_site_packages/nvidia/cudnn/lib:$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    break
done
unset sol_site_packages

mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$CUTE_DSL_CACHE_DIR" \
    "$CCACHE_DIR" "$TMPDIR" "$FLASHINFER_TRACE_DIR" "$SOL_REPO_ROOT/.work"
