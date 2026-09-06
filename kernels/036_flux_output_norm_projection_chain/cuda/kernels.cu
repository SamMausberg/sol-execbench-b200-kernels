// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#include <cmath>
namespace flux {
__device__ float warp_sum(float value) {
    for (int delta = 16; delta; delta >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, delta);
    return value;
}

__device__ float block_sum(float value, float* shared) {
    value = warp_sum(value);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        float partial = lane < blockDim.x / 32 ? shared[lane] : 0.0f;
        partial = warp_sum(partial);
        if (lane == 0) shared[0] = partial;
    }
    __syncthreads();
    return shared[0];
}

__global__ void silu_kernel(const float* input, float* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float value = input[i];
        output[i] = value * (1.0f / (1.0f + __expf(-value)));
    }
}

__global__ void norm_mod_kernel(const float* input, const float* modulation,
                               float* output, int sequence, float eps) {
    constexpr int channels = 3072, items = channels / 256;
    int row = blockIdx.x, thread = threadIdx.x;
    float values[items], sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        values[j] = input[row * channels + thread + j * 256];
        sum += values[j];
    }
    __shared__ float partial[8];
    float mean = block_sum(sum, partial) / channels;
    float square_sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        values[j] -= mean;
        square_sum += values[j] * values[j];
    }
    // Every thread has consumed the previous reduction before scratch is reused.
    __syncthreads();
    float variance = block_sum(square_sum, partial) / channels;
    float denominator = sqrtf(variance + eps);
    int base = (row / sequence) * (2 * channels);
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        int channel = thread + j * 256;
        float shift = modulation[base + channel];
        float scale = modulation[base + channels + channel];
        float normalized = values[j] / denominator;
        output[row * channels + channel] = normalized * (1.0f + scale) + shift;
    }
}

void launch_silu(const float* input, float* output, int count, cudaStream_t stream) {
    silu_kernel<<<(count+255)/256,256,0,stream>>>(input,output,count);
}
void launch_norm(const float* input, const float* modulation, float* output,
                 int rows, int sequence, float eps, cudaStream_t stream) {
    norm_mod_kernel<<<rows,256,0,stream>>>(input,modulation,output,sequence,eps);
}
}
