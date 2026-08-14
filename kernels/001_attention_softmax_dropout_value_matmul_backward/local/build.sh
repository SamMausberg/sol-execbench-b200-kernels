#!/bin/bash
# usage: build.sh <variant> <arch: 120a|100a> [-c]
set -e
V=${1:-flash_v1}
ARCH=${2:-120a}
TP=$(python3 -c "import torch, os; p=os.path.dirname(torch.__file__); print(p)")
INC="-I$TP/include -I$TP/include/torch/csrc/api/include -I/home/sam/cutlass-v441/include -I/home/sam/cutlass-v441/tools/util/include"
PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
OUT=variants/$V/build_$ARCH
mkdir -p $OUT
FLAGS="-O3 -std=c++17 --use_fast_math --expt-relaxed-constexpr -gencode=arch=compute_$ARCH,code=sm_$ARCH -DTORCH_EXTENSION_NAME=benchmark_kernel -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 -Xcompiler -fPIC"
if [ "$3" == "-c" ]; then
  time nvcc $FLAGS $INC -I$PY_INC -c variants/$V/kernel.cu -o $OUT/kernel.o
else
  time nvcc $FLAGS $INC -I$PY_INC -c variants/$V/kernel.cu -o $OUT/kernel.o
  g++ -shared -o $OUT/benchmark_kernel.so $OUT/kernel.o -L$TP/lib -ltorch -ltorch_cpu -ltorch_python -lc10 -ltorch_cuda -lc10_cuda -L/usr/local/cuda/lib64 -lcudart -L/usr/lib/wsl/lib -lcuda
fi
echo OK
