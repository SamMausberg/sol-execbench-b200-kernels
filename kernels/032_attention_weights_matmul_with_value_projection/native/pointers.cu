// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cstdint>

__global__ void pointer_arrays(const uint16_t* weights, const uint16_t* values,
                               uint16_t* output, uintptr_t* pointers,
                               int combined_heads, int sequence) {
    const int combined_head = blockIdx.x * blockDim.x + threadIdx.x;
    if (combined_head >= combined_heads) return;
    const int batch = combined_head / 40, head = combined_head % 40;
    pointers[combined_head] = reinterpret_cast<uintptr_t>(values + int64_t(combined_head) * sequence * 128);
    pointers[combined_heads + combined_head] = reinterpret_cast<uintptr_t>(weights + int64_t(combined_head) * sequence * sequence);
    pointers[2 * combined_heads + combined_head] = reinterpret_cast<uintptr_t>(output + int64_t(batch) * sequence * 5120 + head * 128);
}

void make_av_pointer_arrays(const void* weights, const void* values, void* output,
                            void* pointers, int batch, int sequence, cudaStream_t stream) {
    pointer_arrays<<<(batch * 40 + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint16_t*>(weights), static_cast<const uint16_t*>(values),
        static_cast<uint16_t*>(output), static_cast<uintptr_t*>(pointers), batch * 40, sequence);
}
