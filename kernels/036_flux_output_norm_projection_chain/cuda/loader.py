# SPDX-License-Identifier: Apache-2.0
"""Load a content-addressed native module for local tuning and verification."""
import hashlib
import os
from pathlib import Path
import sysconfig
from torch.utils.cpp_extension import load


def load_native():
    directory = Path(__file__).resolve().parent
    names = ('binding.cpp', 'kernels.cu', 'kernels.h')
    data = {name: (directory / name).read_bytes() for name in names}
    digest = hashlib.sha256(b''.join(name.encode() + b'\0' + data[name] for name in names)).hexdigest()
    target = Path(os.environ['TORCH_EXTENSIONS_DIR']) / 'sol_flux_sources' / digest
    target.mkdir(parents=True, exist_ok=True)
    for name, content in data.items():
        path = target / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
            temporary.write_bytes(content)
            temporary.replace(path)
    cuda_package = Path(sysconfig.get_paths()['purelib']) / 'nvidia/cu13'
    module = load(name='sol_flux_' + digest[:16], sources=[str(target / name) for name in names if name.endswith(('.cpp', '.cu'))],
                  extra_include_paths=[str(target), str(cuda_package / 'include')],
                  extra_cflags=['-O3', '-std=c++17'],
                  extra_cuda_cflags=['-O3', '--fmad=false', '-std=c++17'],
                  extra_ldflags=[f'-L{cuda_package / "lib"}', '-l:libcublasLt.so.13', '-l:libcudart.so.13'],
                  with_cuda=True)
    module.source_sha256 = digest
    return module
