// SPDX-FileCopyrightText: 2026 sol-038-flux-rmsnorm contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

void launch_flux_rmsnorm_qk(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_k,
    float eps,
    torch::Tensor query_norm,
    torch::Tensor key_norm,
    cudaStream_t stream);
