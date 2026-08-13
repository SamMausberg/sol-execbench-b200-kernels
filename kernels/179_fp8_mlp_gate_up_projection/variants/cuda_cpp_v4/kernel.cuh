// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>

void launch_fp8_mlp_gate_up(
    const at::Tensor& x,
    const at::Tensor& scale_x,
    const at::Tensor& gate_weight,
    const at::Tensor& scale_gate,
    const at::Tensor& up_weight,
    const at::Tensor& scale_up,
    at::Tensor output,
    cudaStream_t stream);
