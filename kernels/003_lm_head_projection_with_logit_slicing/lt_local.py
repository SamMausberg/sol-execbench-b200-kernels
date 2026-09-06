"""Development loader. Run under the shared GPU lock, including initialization."""
import hashlib
from pathlib import Path
from torch.utils.cpp_extension import load

_module = None


def module():
    global _module
    if _module is None:
        base = Path(__file__).parent
        digest = hashlib.sha256((base/'binding.cpp').read_bytes() +
                                (base/'cublaslt_bridge.h').read_bytes()).hexdigest()[:16]
        _module = load(name=f'sol_lm_head_{digest}', sources=[str(base/'binding.cpp')],
                       extra_cflags=['-O3', '-std=c++17', '-I/usr/local/cuda/include'],
                       extra_ldflags=['-L/usr/local/cuda/lib64', '-l:libcublasLt.so.13',
                                      '-l:libcudart.so.13'], verbose=False)
    return _module
