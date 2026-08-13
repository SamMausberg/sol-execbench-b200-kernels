// SPDX-FileCopyrightText: 2026 sol-038-flux-rmsnorm contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>

// The evaluator compiles one architecture per artifact.  This lets the
// binding select the native Blackwell projection without teaching the shared
// repository tooling about per-architecture preprocessor flags.
bool has_sm100_cutlass_projection();

void launch_sm100_cutlass_projection(
    const at::Tensor& hidden_states,
    const at::Tensor& in_proj_weight,
    const at::Tensor& in_proj_bias,
    at::Tensor& projected,
    cudaStream_t stream);

void launch_mamba_conv1d_tail(
    const at::Tensor& projected,
    const at::Tensor& attention_mask,
    const at::Tensor& conv1d_weight,
    const at::Tensor& conv1d_bias,
    at::Tensor& output_hidden_states,
    cudaStream_t stream);
