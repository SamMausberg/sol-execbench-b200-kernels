// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

void make_av_pointer_arrays(const void*, const void*, void*, void*, int, int, cudaStream_t);

void check(cublasStatus_t s, const char* name) {
    TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, name, " failed with cuBLASLt status ", int(s));
}

struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms;
    static constexpr size_t workspace_bytes = 32 * 1024 * 1024;
    Plan(int batch, int sequence) {
        check(cublasLtCreate(&handle), "create handle");
        check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "operation");
        check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, 128, sequence, 128), "V layout");
        check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, sequence, sequence, sequence), "A layout");
        check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, 128, sequence, 5120), "C layout");
        const int32_t count = batch * 40;
        const uint32_t mode = batch == 1 ? CUBLASLT_BATCH_MODE_STRIDED : CUBLASLT_BATCH_MODE_POINTER_ARRAY;
        const int64_t strides[] = {int64_t(sequence) * 128, int64_t(sequence) * sequence, 128};
        const cublasLtMatrixLayout_t layouts[] = {a, b, c};
        for (int i = 0; i < 3; ++i) {
            check(cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                   &count, sizeof(count)), "batch count");
            check(cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_BATCH_MODE,
                                                   &mode, sizeof(mode)), "batch mode");
            if (batch == 1) check(cublasLtMatrixLayoutSetAttribute(layouts[i],
                CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i], sizeof(strides[i])), "batch stride");
        }
        cublasLtMatmulPreference_t preference;
        check(cublasLtMatmulPreferenceCreate(&preference), "preference");
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &workspace_bytes, sizeof(workspace_bytes)), "workspace");
        const uint32_t a_align = 16, b_align = sequence % 8 == 0 ? 16 : 2, c_align = 16;
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                                                   &a_align, sizeof(a_align)), "V alignment");
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                                                   &b_align, sizeof(b_align)), "A alignment");
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                                                   &c_align, sizeof(c_align)), "C alignment");
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                                                   &c_align, sizeof(c_align)), "D alignment");
        cublasLtMatmulHeuristicResult_t found[32];
        int count_found = 0;
        const auto status = cublasLtMatmulAlgoGetHeuristic(handle, operation, a, b, c, c, preference,
                                                           32, found, &count_found);
        cublasLtMatmulPreferenceDestroy(preference);
        check(status, "heuristics");
        for (int i = 0; i < count_found; ++i)
            if (found[i].state == CUBLAS_STATUS_SUCCESS) algorithms.push_back(found[i]);
    }
    ~Plan() {
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
    }
};
std::map<std::tuple<int, int, int>, std::unique_ptr<Plan>> plans;
std::mutex plans_mutex;
Plan* plan_for(int device, int batch, int sequence) {
    std::lock_guard<std::mutex> lock(plans_mutex);
    auto& result = plans[{device, batch, sequence}];
    if (!result) result = std::make_unique<Plan>(batch, sequence);
    return result.get();
}

void validate(const torch::Tensor& weights, const torch::Tensor& values) {
    TORCH_CHECK(weights.is_cuda() && values.device() == weights.device(), "expected one CUDA device");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16 && values.scalar_type() == at::kBFloat16,
                "expected BF16");
    TORCH_CHECK(weights.dim() == 4 && values.dim() == 4 && weights.size(1) == 40 &&
                weights.size(2) == weights.size(3) && values.size(0) == weights.size(0) &&
                values.size(1) == 40 && values.size(2) == weights.size(2) && values.size(3) == 128,
                "invalid shape");
}

void execute(const torch::Tensor& weights, const torch::Tensor& values,
             const torch::Tensor& output, int index) {
    validate(weights, values);
    TORCH_CHECK(weights.is_contiguous() && values.is_contiguous(), "dense inputs required");
    TORCH_CHECK(weights.data_ptr() && values.data_ptr() &&
                reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(values.data_ptr()) % 16 == 0, "aligned inputs required");
    const int batch = weights.size(0), sequence = weights.size(2);
    TORCH_CHECK(output.is_contiguous() && output.device() == weights.device() &&
                output.scalar_type() == at::kBFloat16 && output.dim() == 3 &&
                output.size(0) == batch && output.size(1) == sequence && output.size(2) == 5120,
                "invalid output");
    c10::cuda::CUDAGuard guard(weights.device());
    auto* plan = plan_for(weights.get_device(), batch, sequence);
    TORCH_CHECK(index >= 0 && index < int(plan->algorithms.size()), "algorithm index out of range");
    auto workspace = torch::empty({int64_t(Plan::workspace_bytes)}, weights.options().dtype(at::kByte));
    auto pointers = torch::empty({3, batch * 40}, weights.options().dtype(at::kLong));
    const auto stream = c10::cuda::getCurrentCUDAStream(weights.get_device()).stream();
    const void* ap = values.data_ptr();
    const void* bp = weights.data_ptr();
    void* cp = output.data_ptr();
    if (batch > 1) {
        make_av_pointer_arrays(weights.data_ptr(), values.data_ptr(), output.data_ptr(),
                               pointers.data_ptr(), batch, sequence, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        ap = pointers.data_ptr<int64_t>();
        bp = pointers.data_ptr<int64_t>() + batch * 40;
        cp = pointers.data_ptr<int64_t>() + 2 * batch * 40;
    }
    const float alpha = 1, beta = 0;
    check(cublasLtMatmul(plan->handle, plan->operation, &alpha, ap, plan->a, bp, plan->b,
                         &beta, cp, plan->c, cp, plan->c, &plan->algorithms[index].algo,
                         workspace.data_ptr(), workspace.numel(), stream), "batched AV");
}

pybind11::list algorithms(const torch::Tensor& weights, const torch::Tensor& values) {
    validate(weights, values);
    c10::cuda::CUDAGuard guard(weights.device());
    auto* plan = plan_for(weights.get_device(), weights.size(0), weights.size(2));
    pybind11::list results;
    for (int i = 0; i < int(plan->algorithms.size()); ++i) {
        pybind11::dict item;
        item["index"] = i;
        item["workspace_bytes"] = plan->algorithms[i].workspaceSize;
        const cublasLtMatmulAlgoConfigAttributes_t attrs[] = {
            CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
            CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_STAGES_ID,
            CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
        const char* names[] = {"id", "tile", "split_k", "reduction", "swizzle", "custom", "stages",
                               "inner_shape", "cluster_shape"};
        for (int j = 0; j < 9; ++j) {
            int value = -1;
            size_t written;
            if (cublasLtMatmulAlgoConfigGetAttribute(&plan->algorithms[i].algo, attrs[j],
                                                     &value, sizeof(value), &written) == CUBLAS_STATUS_SUCCESS)
                item[names[j]] = value;
        }
        results.append(item);
    }
    return results;
}

torch::Tensor run(const torch::Tensor& weights, const torch::Tensor& values) {
    at::NoGradGuard no_grad;
    validate(weights, values);
    const int batch = weights.size(0), sequence = weights.size(2);
    if (!weights.is_contiguous() || !values.is_contiguous() ||
        reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16 ||
        reinterpret_cast<uintptr_t>(values.data_ptr()) % 16)
        return at::matmul(weights, values).transpose(1, 2).reshape({batch, sequence, 5120});
    auto output = torch::empty({batch, sequence, 5120}, weights.options());
    execute(weights, values, output, 0);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
    module.def("execute", &execute);
    module.def("algorithms", &algorithms);
}
