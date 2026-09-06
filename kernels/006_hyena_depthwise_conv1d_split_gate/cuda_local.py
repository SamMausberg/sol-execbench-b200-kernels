"""Local development loader; the official CUDA package compiles in Phase1."""
import hashlib
from pathlib import Path
from torch.utils.cpp_extension import load

_modules = {}


def module(experiments=False):
    if experiments not in _modules:
        base = Path(__file__).parent
        prefix = 'experiments_' if experiments else ''
        files = [base / (prefix + 'main.cpp'), base / (prefix + 'kernel.cu')]
        digest = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
        _modules[experiments] = load(name=f'sol_hyena_{digest[:16]}', sources=[str(p) for p in files],
                       extra_cflags=['-O3', '-std=c++17'],
                       extra_cuda_cflags=['-O3', '--ftz=false'], verbose=False)
    return _modules[experiments]


def run(*args):
    module().run(*args)


def configured_run(*args):
    module().configured_run(*args)
