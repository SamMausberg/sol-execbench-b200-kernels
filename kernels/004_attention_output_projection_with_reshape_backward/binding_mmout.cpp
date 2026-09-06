// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

// BF16 operands are exactly representable in FP32. Both GEMMs accumulate in
// FP32 and return the contract's BF16 gradients directly.
std::tuple<torch::Tensor, torch::Tensor> run(
    torch::Tensor grad_output, torch::Tensor reshaped, torch::Tensor weight) {
    c10::cuda::CUDAGuard guard(grad_output.device());
    for (const auto& tensor : {grad_output, reshaped, weight}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == grad_output.device() &&
                    tensor.scalar_type() == at::kBFloat16,
                    "Expected BF16 tensors on the same CUDA device");
    }
    TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(2) == 2048 &&
                reshaped.sizes() == grad_output.sizes() && weight.dim() == 2 &&
                weight.size(0) == 2048 && weight.size(1) == 2048,
                "Unsupported gradient dimensions");
    const auto batch = grad_output.size(0), sequence = grad_output.size(1);
    auto gradient = grad_output.reshape({batch * sequence, 2048});
    auto saved = reshaped.reshape({batch * sequence, 2048});
    auto grad_weight = torch::empty({2048, 2048}, gradient.options());
    auto grad_input = torch::empty({batch * sequence, 2048}, gradient.options());
    auto transposed_gradient = gradient.transpose(0, 1);
    at::mm_out(grad_weight, transposed_gradient, saved);
    at::mm_out(grad_input, gradient, weight);
    return {grad_input.view({batch, sequence, 32, 64}).transpose(1, 2), grad_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run, pybind11::call_guard<pybind11::gil_scoped_release>());
}
