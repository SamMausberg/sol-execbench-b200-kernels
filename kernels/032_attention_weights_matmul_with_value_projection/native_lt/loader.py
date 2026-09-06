# SPDX-License-Identifier: Apache-2.0
"""Immutable source snapshots for a local cuBLASLt batched-GEMM experiment."""
import hashlib
import os
from pathlib import Path
import sysconfig
from torch.utils.cpp_extension import load


def load_bridge():
    directory = Path(__file__).resolve().parent
    sources = {name: (directory / name).read_bytes() for name in ("binding.cpp", "pointers.cu")}
    digest = hashlib.sha256(b"".join(name.encode() + data for name, data in sources.items())).hexdigest()
    cache = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / "sol_avlt_sources" / digest
    cache.mkdir(parents=True, exist_ok=True)
    for name, data in sources.items():
        target = cache / name
        if not target.exists():
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
    cuda_package = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
    module = load(name=f"sol_avlt_{digest[:16]}", sources=[str(cache / name) for name in sources],
                  extra_include_paths=[str(cuda_package / "include")],
                  extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=["-O3", "-std=c++17"],
                  extra_ldflags=[f"-L{cuda_package / 'lib'}", "-l:libcublasLt.so.13", "-l:libcudart.so.13"],
                  verbose=True)
    module.source_sha256 = digest
    return module
