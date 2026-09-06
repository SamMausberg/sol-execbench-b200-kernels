// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
namespace flux {
void launch_silu(const float* input, float* output, int count, cudaStream_t stream);
void launch_norm(const float* input, const float* modulation, float* output,
                 int rows, int sequence, float eps, cudaStream_t stream);
}
