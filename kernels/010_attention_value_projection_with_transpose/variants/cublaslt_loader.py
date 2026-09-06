"""Build the generic cuBLASLt tuning bridge in the persistent torch cache."""

from pathlib import Path
import hashlib
import os
import sysconfig

from torch.utils.cpp_extension import load


def load_bridge(verbose=False):
    directory = Path(__file__).resolve().parent
    cuda_package = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
    source_bytes = (directory / "cublaslt_bridge.cpp").read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    snapshot_dir = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / "sol_cublaslt_sources" / digest
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / "cublaslt_bridge.cpp"
    if not snapshot.exists():
        temporary = snapshot.with_suffix(f".cpp.{os.getpid()}.tmp")
        temporary.write_bytes(source_bytes)
        temporary.replace(snapshot)
    module = load(
        name=f"sol_cublaslt_{digest[:16]}",
        sources=[str(snapshot)],
        extra_include_paths=[str(cuda_package / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[f"-L{cuda_package / 'lib'}", "-l:libcublasLt.so.13", "-l:libcudart.so.13"],
        with_cuda=False,
        verbose=verbose,
    )
    module.source_sha256 = digest
    module.source_path = str(snapshot)
    return module


if __name__ == "__main__":
    print(load_bridge(verbose=True))
