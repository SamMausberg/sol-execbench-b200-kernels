#!/usr/bin/env bash
# Install the pinned evaluator stack and CUDA compiler on persistent Runpod storage.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/native_env.sh"

for sol_command in python3 curl git tar gcc g++ make; do
    command -v "$sol_command" >/dev/null || {
        echo "Missing $sol_command. Install host prerequisites: apt-get install -y python3 python3-dev curl git build-essential libblas-dev liblapack-dev sudo" >&2
        exit 2
    }
done
[[ -d "$SOL_WORKSPACE" ]] || { echo "$SOL_WORKSPACE does not exist" >&2; exit 2; }

sol_uv_version=0.11.26
sol_installed_uv_version=""
if [[ -x "$SOL_WORKSPACE/bin/uv" ]]; then
    read -r _ sol_installed_uv_version _ < <("$SOL_WORKSPACE/bin/uv" --version)
fi
if [[ "$sol_installed_uv_version" != "$sol_uv_version" ]]; then
    [[ "$(uname -m)" == x86_64 ]] || { echo "This bootstrap currently supports x86_64 Runpod images" >&2; exit 2; }
    mkdir -p "$SOL_WORKSPACE/bin"
    sol_download_dir="$(mktemp -d "$TMPDIR/uv.XXXXXX")"
    trap 'rm -rf -- "$sol_download_dir"' EXIT
    curl --fail --location --retry 3 \
        "https://github.com/astral-sh/uv/releases/download/$sol_uv_version/uv-x86_64-unknown-linux-gnu.tar.gz" \
        --output "$sol_download_dir/uv.tar.gz"
    tar -xzf "$sol_download_dir/uv.tar.gz" -C "$sol_download_dir"
    install -m 755 "$sol_download_dir/uv-x86_64-unknown-linux-gnu/uv" "$SOL_WORKSPACE/bin/uv"
    install -m 755 "$sol_download_dir/uv-x86_64-unknown-linux-gnu/uvx" "$SOL_WORKSPACE/bin/uvx"
    rm -rf -- "$sol_download_dir"
    trap - EXIT
fi

cd "$SOL_REPO_ROOT"
git submodule update --init --recursive third_party/sol-execbench
sol_expected_revision="$(python3 -c 'import json; print(json.load(open("benchmark.lock.json"))["evaluator"]["revision"])')"
[[ "$(git -C third_party/sol-execbench rev-parse HEAD)" == "$sol_expected_revision" ]] || {
    echo "Evaluator checkout does not match benchmark.lock.json" >&2; exit 2;
}
[[ -z "$(git -C third_party/sol-execbench status --porcelain --untracked-files=no)" ]] || {
    echo "Evaluator checkout has local changes; restore the pinned evaluator before setup" >&2; exit 2;
}

mkdir -p "$SOL_WORKSPACE/upstream"
if [[ ! -d "$CUTLASS_DIR/.git" ]]; then
    git clone --depth 1 --branch v4.4.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi
[[ "$(git -C "$CUTLASS_DIR" describe --tags --exact-match HEAD)" == v4.4.1 ]] || {
    echo "CUTLASS_DIR must contain CUTLASS v4.4.1" >&2; exit 2;
}

# uv's frozen graph contains upstream Triton. Match the Dockerfile's final overrides.
uv sync --project "$SOL_REPO_ROOT/third_party/sol-execbench" --frozen --no-editable --all-groups --python 3.12
uv pip uninstall --python "$SOL_VENV/bin/python" triton
uv pip install --python "$SOL_VENV/bin/python" --no-deps fbtriton==3.7.1
uv pip install --python "$SOL_VENV/bin/python" --no-deps --force-reinstall nvidia-cutlass-dsl-libs-cu13==4.4.2

# fbtriton 3.7.1 ships launch.h in runtime/, while its driver searches the backend.
# Supply the expected path to the identical wheel header; do not patch driver code.
sol_triton_dir="$("$SOL_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/triton")')"
[[ -f "$sol_triton_dir/runtime/launch.h" ]] || { echo "Pinned fbtriton launcher header is missing" >&2; exit 2; }
if [[ ! -e "$sol_triton_dir/backends/nvidia/launch.h" ]]; then
    ln -s ../../runtime/launch.h "$sol_triton_dir/backends/nvidia/launch.h"
fi
cmp -s "$sol_triton_dir/runtime/launch.h" "$sol_triton_dir/backends/nvidia/launch.h" || {
    echo "fbtriton backend launcher header differs from the pinned wheel header" >&2; exit 2;
}

# Wheels provide nvcc, CRT, CCCL, NVVM and runtime headers without Docker or a host CUDA change.
# Pin NVVM explicitly: allowing its latest release mixes CUDA compiler generations.
uv pip install --python "$SOL_VENV/bin/python" --target "$SOL_WORKSPACE/toolchains/cuda-13.1" \
    nvidia-cuda-nvcc==13.1.115 nvidia-cuda-crt==13.1.115 nvidia-cuda-cccl==13.1.115 \
    nvidia-nvvm==13.1.115 nvidia-cuda-runtime==13.1.80
source "$SOL_REPO_ROOT/tools/native_env.sh"
# CUDA wheels omit the unversioned runtime soname used by torch's -lcudart link.
if [[ ! -e "$CUDA_HOME/lib/libcudart.so" ]]; then
    ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
fi
if [[ ! -e "$CUDA_HOME/lib64" ]]; then
    ln -s lib "$CUDA_HOME/lib64"
fi
"$CUDACXX" --version
python "$SOL_REPO_ROOT/tools/campaign.py" info --output "$SOL_REPO_ROOT/.work/native-environment.json"
echo "Native environment ready. Source $SOL_REPO_ROOT/tools/native_env.sh in each new shell."
