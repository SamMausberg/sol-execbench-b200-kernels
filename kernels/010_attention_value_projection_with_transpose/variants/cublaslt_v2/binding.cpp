// SPDX-License-Identifier: Apache-2.0
// Shape-selected cuBLASLt value projection with an ordinary transposed output view.
#define SOL_LT_NO_BINDING
#include "cublaslt_bridge.h"

struct Selection {
    int64_t workspace_bytes;
    std::vector<int> configuration;
};

const std::map<int64_t, Selection> selected_algorithms{
    {128, {1572880, {66, 15, 3, 2, 0, 3, 35, 0, 6}}},
    {256, {0, {66, 12, 1, 0, 0, 2, 35, 0, 6}}},
    {512, {0, {66, 15, 1, 0, 0, 3, 35, 0, 6}}},
    {1024, {0, {66, 17, 1, 0, 0, 3, 35, 0, 6}}},
    {1571, {0, {66, 328, 1, 0, 0, 1, 35, 0, 3}}},
    {2048, {0, {66, 20, 1, 0, 0, 3, 35, 0, 6}}},
    {2164, {0, {66, 20, 1, 0, 0, 1, 35, 0, 3}}},
    {4096, {0, {66, 23, 1, 0, 0, 3, 35, 0, 6}}},
    {8192, {0, {66, 513, 1, 0, 0, 3, 35, 0, 6}}},
};

torch::Tensor run(const torch::Tensor& hidden_states, const torch::Tensor& v_proj_weight) {
    TORCH_CHECK(hidden_states.dim() == 3 && hidden_states.size(2) == 5120,
                "hidden states must have shape [batch, sequence, 5120]");
    TORCH_CHECK(v_proj_weight.dim() == 2 && v_proj_weight.size(0) == 1024 && v_proj_weight.size(1) == 5120,
                "value weight must have shape [1024, 5120]");
    const int64_t batch = hidden_states.size(0), sequence = hidden_states.size(1);
    const int64_t rows = batch * sequence;
    auto hidden = hidden_states.reshape({rows, 5120});
    auto output = torch::empty({rows, 1024}, hidden_states.options());
    const auto selected = selected_algorithms.find(rows);
    if (selected == selected_algorithms.end()) {
        auto workspace = torch::empty({32 * 1024 * 1024}, hidden_states.options().dtype(at::kByte));
        sol_lt::matmul(hidden, v_proj_weight, output, workspace, 0);
    } else {
        auto workspace = torch::empty({selected->second.workspace_bytes}, hidden_states.options().dtype(at::kByte));
        sol_lt::matmul_config(hidden, v_proj_weight, output, workspace, selected->second.configuration);
    }
    return output.view({batch, sequence, 8, 128}).transpose(1, 2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
}
