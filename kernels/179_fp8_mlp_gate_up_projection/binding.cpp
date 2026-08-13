// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "kernel.cuh"

namespace {

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void run(
    const at::Tensor& x,
    const at::Tensor& scale_x,
    const at::Tensor& gate_proj_weight,
    const at::Tensor& scale_gate,
    const at::Tensor& up_proj_weight,
    const at::Tensor& scale_up,
    at::Tensor output) {
  c10::cuda::CUDAGuard guard(x.device());

  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(scale_x, "scale_x");
  check_cuda_contiguous(gate_proj_weight, "gate_proj_weight");
  check_cuda_contiguous(scale_gate, "scale_gate");
  check_cuda_contiguous(up_proj_weight, "up_proj_weight");
  check_cuda_contiguous(scale_up, "scale_up");
  check_cuda_contiguous(output, "output");

  TORCH_CHECK(x.dim() == 2 && x.size(1) == 3584,
              "x must have shape [M, 3584]");
  const auto m = x.size(0);
  TORCH_CHECK(m > 0 && (m % 128) == 0, "M must be a positive multiple of 128");
  TORCH_CHECK(scale_x.sizes() == at::IntArrayRef({m, 28}),
              "scale_x must have shape [M, 28]");
  TORCH_CHECK(gate_proj_weight.sizes() == at::IntArrayRef({18944, 3584}),
              "gate_proj_weight must have shape [18944, 3584]");
  TORCH_CHECK(up_proj_weight.sizes() == gate_proj_weight.sizes(),
              "up_proj_weight must match gate_proj_weight");
  TORCH_CHECK(scale_gate.sizes() == at::IntArrayRef({148, 28}),
              "scale_gate must have shape [148, 28]");
  TORCH_CHECK(scale_up.sizes() == scale_gate.sizes(),
              "scale_up must match scale_gate");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({m, 18944}),
              "output must have shape [M, 18944]");

  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "x must be float8_e4m3fn");
  TORCH_CHECK(gate_proj_weight.scalar_type() == x.scalar_type() &&
                  up_proj_weight.scalar_type() == x.scalar_type(),
              "weights must be float8_e4m3fn");
  TORCH_CHECK(scale_x.scalar_type() == at::kFloat &&
                  scale_gate.scalar_type() == at::kFloat &&
                  scale_up.scalar_type() == at::kFloat,
              "scales must be float32");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16,
              "output must be bfloat16");

  const int device = x.get_device();
  TORCH_CHECK(scale_x.get_device() == device &&
                  gate_proj_weight.get_device() == device &&
                  scale_gate.get_device() == device &&
                  up_proj_weight.get_device() == device &&
                  scale_up.get_device() == device &&
                  output.get_device() == device,
              "all tensors must be on the same CUDA device");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  launch_fp8_mlp_gate_up(x, scale_x, gate_proj_weight, scale_gate,
                         up_proj_weight, scale_up, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &run, "Blockwise FP8 gate/up projection (DPS)");
}
