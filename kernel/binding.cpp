// SPDX-FileCopyrightText: 2026 sol-038-flux-rmsnorm contributors
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

#include "kernel.cuh"

namespace {

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have float32 dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 31u) == 0,
              name, " must be 32-byte aligned");
}

}  // namespace

void run(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_k,
    double eps,
    torch::Tensor query_norm,
    torch::Tensor key_norm) {
  c10::cuda::CUDAGuard guard(query.device());

  check_tensor(query, "query");
  check_tensor(key, "key");
  check_tensor(weight_q, "weight_q");
  check_tensor(weight_k, "weight_k");
  check_tensor(query_norm, "query_norm");
  check_tensor(key_norm, "key_norm");

  TORCH_CHECK(query.dim() == 4, "query must be four-dimensional");
  TORCH_CHECK(query.sizes() == key.sizes(), "query and key shapes must match");
  TORCH_CHECK(query.size(2) == 48 && query.size(3) == 128,
              "query and key must have 48 heads of width 128");
  TORCH_CHECK(weight_q.sizes() == torch::IntArrayRef({48, 128}),
              "weight_q must have shape [48, 128]");
  TORCH_CHECK(weight_k.sizes() == torch::IntArrayRef({48, 128}),
              "weight_k must have shape [48, 128]");
  TORCH_CHECK(query_norm.sizes() == query.sizes(),
              "query_norm shape must match query");
  TORCH_CHECK(key_norm.sizes() == key.sizes(),
              "key_norm shape must match key");
  TORCH_CHECK(query.get_device() == key.get_device() &&
                  query.get_device() == weight_q.get_device() &&
                  query.get_device() == weight_k.get_device() &&
                  query.get_device() == query_norm.get_device() &&
                  query.get_device() == key_norm.get_device(),
              "all tensors must use the same CUDA device");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
  launch_flux_rmsnorm_qk(query, key, weight_q, weight_k,
                         static_cast<float>(eps), query_norm, key_norm, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &run, "Flux query/key per-head RMSNorm");
}
