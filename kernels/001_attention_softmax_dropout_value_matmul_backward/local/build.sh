#!/bin/bash
# usage: build.sh <variant> <arch: 120a|100a> [-c]
set -e
V=${1:-flash_v1}
ARCH=${2:-120a}
TP=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
INC="-I$TP/include -I$TP/include/torch/csrc/api/include -I/home/sam/cutlass-v441/include -I/home/sam/cutlass-v441/tools/util/include"
PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
OUT=variants/$V/build_$ARCH
SRCDIR=variants/$V
mkdir -p $OUT
FLAGS="-O3 -std=c++17 --use_fast_math --expt-relaxed-constexpr -gencode=arch=compute_$ARCH,code=sm_$ARCH -DTORCH_EXTENSION_NAME=benchmark_kernel -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 -Xcompiler -fPIC -I$SRCDIR"
OBJS=""
PIDS=""
for f in $SRCDIR/*.cu; do
  base=$(basename $f .cu)
  nvcc $FLAGS $INC -I$PY_INC -c $f -o $OUT/$base.o &
  PIDS="$PIDS $!"
  OBJS="$OBJS $OUT/$base.o"
done
for p in $PIDS; do wait $p; done
if [ "$3" != "-c" ]; then
  g++ -shared -o $OUT/benchmark_kernel.so $OBJS -L$TP/lib -ltorch -ltorch_cpu -ltorch_python -lc10 -ltorch_cuda -lc10_cuda -L/usr/local/cuda/lib64 -lcudart -L/usr/lib/wsl/lib -lcuda
fi
echo OK
