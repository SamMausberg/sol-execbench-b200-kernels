// SPDX-License-Identifier: Apache-2.0
#define SOL_LT_NO_BINDING
#include "cublaslt_bridge.h"

// Explicit algorithm metadata for the public tensor dimensions. No input values
// or tensor addresses are cached; each invocation supplies fresh pointers.
const std::map<int64_t, std::vector<int>> selected_algorithms{
    {128, {66,20,1,0,0,1,35,0,3}},
    {256, {66,201,1,0,0,2,35,0,5}},
    {293, {66,29,1,0,0,1,35,0,3}},
    {691, {66,23,1,0,0,1,35,0,3}},
    {1024, {66,23,1,0,0,1,35,0,3}},
    {2048, {66,23,1,0,0,1,35,0,3}},
    {3011, {66,23,1,0,0,1,35,0,3}},
    {3412, {66,23,1,0,0,1,35,0,3}},
    {3988, {66,23,1,0,0,1,35,0,3}},
    {4096, {66,23,1,0,0,1,35,0,3}},
    {8192, {66,23,1,0,0,1,35,0,3}},
};

torch::Tensor run(const torch::Tensor& hidden_states, const torch::Tensor& weight) {
    const int64_t batch = hidden_states.size(0), sequence = hidden_states.size(1);
    const int64_t rows = batch * sequence, columns = weight.size(0);
    if (!hidden_states.is_contiguous() || !weight.is_contiguous() ||
        reinterpret_cast<uintptr_t>(hidden_states.data_ptr()) % 256 != 0 ||
        reinterpret_cast<uintptr_t>(weight.data_ptr()) % 256 != 0)
        return at::matmul(hidden_states, weight.t());
    auto hidden = hidden_states.view({rows, hidden_states.size(2)});
    auto output = torch::empty({rows, columns}, hidden_states.options());
    const auto selected = selected_algorithms.find(rows);
    if (selected == selected_algorithms.end()) {
        auto workspace = torch::empty({32 * 1024 * 1024}, hidden_states.options().dtype(at::kByte));
        sol_lt::matmul(hidden, weight, output, workspace, 0);
    } else {
        auto workspace = torch::empty({0}, hidden_states.options().dtype(at::kByte));
        sol_lt::matmul_config(hidden, weight, output, workspace, selected->second);
    }
    return output.view({batch, sequence, columns});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
    module.def("algorithms", &sol_lt::algorithms);
    module.def("matmul", &sol_lt::matmul);
    module.def("matmul_config", &sol_lt::matmul_config);
    module.attr("cublaslt_version") = cublasLtGetVersion();
}
