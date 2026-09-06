// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace backward_lt {
void check(cublasStatus_t status, const char* operation) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation, ": cuBLAS status ", int(status));
}
constexpr size_t workspace_limit = 32 * 1024 * 1024;

struct Description {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t a{}, b{}, c{};
    cublasLtMatmulPreference_t preference{};
    std::vector<cublasLtMatmulHeuristicResult_t> candidates;
    Description(int64_t rows, bool weight_gradient, cublasLtHandle_t handle) {
        check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "operation");
        if (weight_gradient) {
            cublasOperation_t transpose = CUBLAS_OP_T;
            check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                               &transpose, sizeof(transpose)), "transpose");
        }
        check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, rows, 2048, 2048), "A layout");
        check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, weight_gradient ? rows : 2048, 2048, 2048), "B layout");
        check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, weight_gradient ? 2048 : rows, 2048, 2048), "C layout");
        cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        for (auto layout : {a, b, c})
            check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), "row order");
        check(cublasLtMatmulPreferenceCreate(&preference), "preference");
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_limit, sizeof(workspace_limit)), "workspace");
        // The wrapper supplies dense tensors with at least sixteen-byte alignment.
        const uint32_t alignment = 16;
        for (auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
            check(cublasLtMatmulPreferenceSetAttribute(preference, attribute, &alignment, sizeof(alignment)), "alignment");
        candidates.resize(24); int count = 0;
        check(cublasLtMatmulAlgoGetHeuristic(handle, operation, a, b, c, c, preference,
                                            candidates.size(), candidates.data(), &count), "heuristics");
        candidates.resize(count);
        TORCH_CHECK(count > 0, "No supported BF16 gradient algorithm");
    }
    ~Description() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};

struct State {
    cublasLtHandle_t handle{};
    std::map<std::pair<int64_t,bool>, std::unique_ptr<Description>> descriptions;
    State() { check(cublasLtCreate(&handle), "handle"); }
    ~State() { descriptions.clear(); if (handle) cublasLtDestroy(handle); }
};
thread_local std::map<int, std::unique_ptr<State>> states;

State& state_for(const torch::Tensor& input) {
    auto& state = states[input.get_device()];
    if (!state) state = std::make_unique<State>();
    return *state;
}
Description& description(State& state, int64_t rows, bool weight_gradient) {
    auto& entry = state.descriptions[{rows, weight_gradient}];
    if (!entry) entry = std::make_unique<Description>(rows, weight_gradient, state.handle);
    return *entry;
}
void multiply(State& state, Description& desc, const torch::Tensor& a, const torch::Tensor& b,
              torch::Tensor& c, torch::Tensor& scratch, cudaStream_t stream, int index) {
    TORCH_CHECK(index >= 0 && index < int(desc.candidates.size()), "Invalid algorithm index");
    const auto& candidate = desc.candidates[index];
    TORCH_CHECK(candidate.state == CUBLAS_STATUS_SUCCESS && candidate.workspaceSize <= workspace_limit,
                "Unsupported algorithm workspace");
    const float alpha = 1, beta = 0;
    check(cublasLtMatmul(state.handle, desc.operation, &alpha, a.data_ptr(), desc.a, b.data_ptr(), desc.b,
                        &beta, c.data_ptr(), desc.c, c.data_ptr(), desc.c, &candidate.algo,
                        scratch.data_ptr(), workspace_limit, stream), "gradient matmul");
}
torch::Tensor aligned_dense(torch::Tensor input) {
    if (!input.is_contiguous()) return input.contiguous();
    if (reinterpret_cast<uintptr_t>(input.data_ptr()) % 16) return input.clone();
    return input;
}

std::tuple<torch::Tensor, torch::Tensor> configured(torch::Tensor gradient, torch::Tensor saved,
                                                   torch::Tensor weight, int weight_algorithm, int input_algorithm) {
    c10::cuda::CUDAGuard guard(gradient.device());
    for (const auto& input : {gradient, saved, weight})
        TORCH_CHECK(input.is_cuda() && input.device() == gradient.device() && input.scalar_type() == at::kBFloat16,
                    "Expected BF16 tensors on one CUDA device");
    TORCH_CHECK(gradient.dim() == 3 && gradient.size(2) == 2048 && saved.sizes() == gradient.sizes()
                && weight.dim() == 2 && weight.size(0) == 2048 && weight.size(1) == 2048, "Unsupported shape");
    const int64_t batch = gradient.size(0), sequence = gradient.size(1), rows = batch * sequence;
    gradient = aligned_dense(gradient).view({rows, 2048});
    saved = aligned_dense(saved).view({rows, 2048});
    weight = aligned_dense(weight);
    auto output_weight = torch::empty({2048, 2048}, gradient.options());
    auto output_input = torch::empty({rows, 2048}, gradient.options());
    auto output_view = output_input.view({batch, sequence, 32, 64}).transpose(1, 2);
    auto scratch = torch::empty({int64_t(workspace_limit)}, gradient.options().dtype(at::kByte));
    auto& state = state_for(gradient);
    auto& dw = description(state, rows, true);
    auto& dx = description(state, rows, false);
    auto stream = c10::cuda::getCurrentCUDAStream(gradient.get_device()).stream();
    multiply(state, dw, gradient, saved, output_weight, scratch, stream, weight_algorithm);
    multiply(state, dx, gradient, weight, output_input, scratch, stream, input_algorithm);
    return {output_view, output_weight};
}

void component(torch::Tensor a, torch::Tensor b, torch::Tensor output, torch::Tensor scratch,
               bool weight_gradient, int index) {
    c10::cuda::CUDAGuard guard(a.device());
    TORCH_CHECK(a.dim() == 2 && a.size(1) == 2048 && a.is_contiguous() && b.is_contiguous()
                && output.is_contiguous() && scratch.numel() >= int64_t(workspace_limit), "Invalid component buffers");
    auto& state = state_for(a);
    auto& desc = description(state, a.size(0), weight_gradient);
    multiply(state, desc, a, b, output, scratch, c10::cuda::getCurrentCUDAStream(a.get_device()).stream(), index);
}
pybind11::dict algorithms(torch::Tensor gradient) {
    c10::cuda::CUDAGuard guard(gradient.device());
    auto& state = state_for(gradient);
    pybind11::dict result;
    for (bool weight_gradient : {false, true}) {
        auto& desc = description(state, gradient.numel() / 2048, weight_gradient);
        pybind11::list entries;
        for (size_t i = 0; i < desc.candidates.size(); ++i) {
            const auto& candidate = desc.candidates[i];
            pybind11::dict entry;
            entry["index"] = i; entry["workspace_bytes"] = candidate.workspaceSize; entry["waves"] = candidate.wavesCount;
            int id = 0; size_t written = 0;
            check(cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written), "algorithm id");
            entry["id"] = id; entries.append(entry);
        }
        result[weight_gradient ? "weight" : "input"] = entries;
    }
    return result;
}
std::tuple<torch::Tensor, torch::Tensor> run(torch::Tensor gradient, torch::Tensor saved, torch::Tensor weight) {
    return configured(gradient, saved, weight, 0, 0);
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &backward_lt::run, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("configured", &backward_lt::configured, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("component", &backward_lt::component, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("algorithms", &backward_lt::algorithms);
}
