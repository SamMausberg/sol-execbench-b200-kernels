# SPDX-License-Identifier: Apache-2.0
"""Build immutable CUDA/C++ three-fragment expansion experiments in the persistent cache."""
import hashlib
import os
from pathlib import Path
import sysconfig

from torch.utils.cpp_extension import load


def load_bridge(verbose=False):
    directory = Path(__file__).resolve().parent
    sources = {name: (directory / name).read_bytes() for name in ("bridge.cpp", "kernels.cu")}
    digest = hashlib.sha256(b"".join(name.encode() + b"\0" + data for name, data in sources.items())).hexdigest()
    snapshot_dir = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / "sol_expansion_sources" / digest
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, data in sources.items():
        snapshot = snapshot_dir / name
        if not snapshot.exists():
            temporary = snapshot.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(data)
            temporary.replace(snapshot)
    cuda_package = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
    module = load(
        name=f"sol_expansion_{digest[:16]}",
        sources=[str(snapshot_dir / name) for name in sources],
        extra_include_paths=[str(cuda_package / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--fmad=false", "-gencode=arch=compute_100a,code=sm_100a"],
        extra_ldflags=[f"-L{cuda_package / 'lib'}", "-l:libcublasLt.so.13"],
        with_cuda=True,
        verbose=verbose,
    )
    module.source_sha256 = digest
    module.source_path = str(snapshot_dir)
    module.file_sha256 = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    return module


if __name__ == "__main__":
    print(load_bridge(verbose=True))
