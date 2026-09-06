// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void split_pair(const float* a, const float* b,
                           __nv_bfloat16* ah, __nv_bfloat16* al,
                           __nv_bfloat16* bh, __nv_bfloat16* bl,
                           int64_t acount, int64_t bcount) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= acount + bcount) return;
    bool left = i < acount;
    int64_t offset = left ? i : i - acount;
    float value = left ? a[offset] : b[offset];
    __nv_bfloat16 high = __float2bfloat16_rn(value);
    __nv_bfloat16 low = __float2bfloat16_rn(value - __bfloat162float(high));
    if (left) { ah[offset] = high; al[offset] = low; }
    else { bh[offset] = high; bl[offset] = low; }
}

void launch_split_pair(const float* a, const float* b, void* ah, void* al, void* bh, void* bl,
                       int64_t acount, int64_t bcount, cudaStream_t stream) {
    split_pair<<<(acount+bcount+255)/256,256,0,stream>>>(a,b,
        static_cast<__nv_bfloat16*>(ah),static_cast<__nv_bfloat16*>(al),
        static_cast<__nv_bfloat16*>(bh),static_cast<__nv_bfloat16*>(bl),acount,bcount);
}
