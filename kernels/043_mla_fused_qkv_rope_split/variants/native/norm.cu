// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void normalize_rows(const __nv_bfloat16* latent,
                               const __nv_bfloat16* weight,
                               __nv_bfloat16* output, float epsilon) {
    constexpr int width = 1536;
    constexpr int threads = 256;
    __shared__ float warp_sums[8];
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float values[6];
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        values[i] = __bfloat162float(latent[row * width + threadIdx.x + i * threads]);
        sum = __fadd_rn(sum, __fmul_rn(values[i], values[i]));
    }
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta));
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int delta = 16; delta; delta >>= 1)
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta));
        if (lane == 0) warp_sums[0] = sum;
    }
    __syncthreads();
    const float inverse = rsqrtf(warp_sums[0] / float(width) + epsilon);
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int column = threadIdx.x + i * threads;
        const auto normalized = __float2bfloat16_rn(__fmul_rn(values[i], inverse));
        const float result = __fmul_rn(__bfloat162float(normalized), __bfloat162float(weight[column]));
        output[row * width + column] = __float2bfloat16_rn(result);
    }
}

void mla_normalize(const void* latent, const void* weight, void* output,
                   int rows, float epsilon, cudaStream_t stream) {
    normalize_rows<<<rows, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(latent),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output), epsilon);
}
