// SPDX-FileCopyrightText: 2026 sol-038-flux-rmsnorm contributors
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ops/linear.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <tuple>

#include "kernel.cuh"

namespace {

constexpr int64_t kHidden = 8192;
constexpr int64_t kIntermediate = 16384;
constexpr int64_t kProjected = 32768;

void check_bf16_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have bfloat16 dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& attention_mask,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& in_proj_bias,
    const torch::Tensor& conv1d_weight,
    const torch::Tensor& conv1d_bias) {
  c10::cuda::CUDAGuard guard(hidden_states.device());

  check_bf16_cuda_contiguous(hidden_states, "hidden_states");
  check_bf16_cuda_contiguous(attention_mask, "attention_mask");
  check_bf16_cuda_contiguous(in_proj_weight, "in_proj_weight");
  check_bf16_cuda_contiguous(in_proj_bias, "in_proj_bias");
  check_bf16_cuda_contiguous(conv1d_weight, "conv1d_weight");
  check_bf16_cuda_contiguous(conv1d_bias, "conv1d_bias");

  TORCH_CHECK(hidden_states.dim() == 3 && hidden_states.size(2) == kHidden,
              "hidden_states must have shape [B, L, 8192]");
  const int64_t batch = hidden_states.size(0);
  const int64_t sequence = hidden_states.size(1);
  TORCH_CHECK(attention_mask.sizes() ==
                  torch::IntArrayRef({batch, sequence}),
              "attention_mask must have shape [B, L]");
  TORCH_CHECK(in_proj_weight.sizes() ==
                  torch::IntArrayRef({kProjected, kHidden}),
              "in_proj_weight must have shape [32768, 8192]");
  TORCH_CHECK(in_proj_bias.sizes() == torch::IntArrayRef({kProjected}),
              "in_proj_bias must have shape [32768]");
  TORCH_CHECK(conv1d_weight.sizes() ==
                  torch::IntArrayRef({kIntermediate, 1, 4}),
              "conv1d_weight must have shape [16384, 1, 4]");
  TORCH_CHECK(conv1d_bias.sizes() == torch::IntArrayRef({kIntermediate}),
              "conv1d_bias must have shape [16384]");

  const int device = hidden_states.get_device();
  TORCH_CHECK(attention_mask.get_device() == device &&
                  in_proj_weight.get_device() == device &&
                  in_proj_bias.get_device() == device &&
                  conv1d_weight.get_device() == device &&
                  conv1d_bias.get_device() == device,
              "all tensors must be on the same CUDA device");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

  torch::Tensor projected;
  if (has_sm100_cutlass_projection()) {
    // B200: keep the entire [M,8192] x [8192,32768] projection in one native
    // Blackwell TMA/UMMA launch, including the per-column BF16 bias epilogue.
    projected = torch::empty(
        {batch, sequence, kProjected}, hidden_states.options());
    launch_sm100_cutlass_projection(
        hidden_states, in_proj_weight, in_proj_bias, projected, stream);
  } else {
    // Local SM120 validation deliberately retains the already-proven ATen
    // path.  CUTLASS SM100 kernels cannot execute on the GeForce target.
    projected = at::linear(hidden_states, in_proj_weight, in_proj_bias);
  }

  // The reference returns this transpose/chunk as a view.  Keeping the same
  // view avoids a 2 * B * L * 16384-byte read/write transpose and also retains
  // projected's storage until the caller releases gate.
  torch::Tensor gate =
      projected.slice(/*dim=*/2, kIntermediate, kProjected)
          .transpose(/*dim0=*/1, /*dim1=*/2);

  torch::Tensor output_hidden_states = torch::empty(
      {batch, kIntermediate, sequence}, hidden_states.options());
  launch_mamba_conv1d_tail(
      projected,
      attention_mask,
      conv1d_weight,
      conv1d_bias,
      output_hidden_states,
      stream);

  return std::make_tuple(output_hidden_states, gate);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &run,
             "B200 fused Mamba projection/causal-conv1d/SiLU");
}
