# SPDX-License-Identifier: Apache-2.0
"""Build the projection/residual experiment in the persistent extension cache."""

import hashlib
from pathlib import Path
import sysconfig
from torch.utils.cpp_extension import load


def load_bridge():
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256(b"".join((directory / n).read_bytes()
                                   for n in ("binding.cpp", "cublaslt_bridge.h"))).hexdigest()
    cuda_package = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
    return load(
        name=f"sol_projection_residual_{digest[:16]}",
        sources=[str(directory / "binding.cpp")],
        extra_include_paths=[str(cuda_package / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[f"-L{cuda_package / 'lib'}", "-l:libcublasLt.so.13", "-l:libcudart.so.13"],
        with_cuda=False,
        verbose=False,
    )
