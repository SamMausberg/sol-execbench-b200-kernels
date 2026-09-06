// SPDX-License-Identifier: Apache-2.0
// Keep both launches in C++ while preserving the BF16 GEMM output rounding.
#define SOL_LT_NO_BINDING
#include "cublaslt_bridge.h"

void project(const torch::Tensor& a, const torch::Tensor& weight,
             const torch::Tensor& residual, torch::Tensor output,
             torch::Tensor temporary, torch::Tensor workspace, int algorithm_index) {
    sol_lt::matmul(a, weight, temporary, workspace, algorithm_index);
    at::add_out(output, temporary, residual);
}

void configured(const torch::Tensor& a, const torch::Tensor& weight,
                const torch::Tensor& residual, torch::Tensor output,
                torch::Tensor temporary, torch::Tensor workspace,
                const std::vector<int>& configuration) {
    sol_lt::matmul_config(a, weight, temporary, workspace, configuration);
    at::add_out(output, temporary, residual);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("algorithms", &sol_lt::algorithms);
    module.def("project", &project);
    module.def("configured", &configured);
}
