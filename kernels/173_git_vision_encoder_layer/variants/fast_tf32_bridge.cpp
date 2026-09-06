// SPDX-License-Identifier: Apache-2.0
// Row-major A @ B.T bridge; FP32 operands explicitly use FAST_TF32 compute.
// FP16/BF16 operands retain FP32 accumulation. No framework precision flags change.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace sol_lt {

void check(cublasStatus_t status, const char* operation) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation,
                " failed with cuBLAS status ", static_cast<int>(status));
}

cublasComputeType_t compute_type(cudaDataType_t type) {
    return type == CUDA_R_32F ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

struct Description {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> candidates;
    std::map<std::vector<int>, cublasLtMatmulHeuristicResult_t> configured;

    Description(int64_t m, int64_t n, int64_t k, cudaDataType_t type,
                size_t workspace_limit, cublasLtHandle_t handle) {
        check(cublasLtMatmulDescCreate(&operation, compute_type(type), CUDA_R_32F), "create operation");
        cublasOperation_t transpose = CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                           &transpose, sizeof(transpose)), "set transpose");
        check(cublasLtMatrixLayoutCreate(&a, type, m, k, k), "create A layout");
        check(cublasLtMatrixLayoutCreate(&b, type, n, k, k), "create B layout");
        check(cublasLtMatrixLayoutCreate(&c, type, m, n, n), "create C layout");
        cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        for (auto layout : {a, b, c}) {
            check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                 &order, sizeof(order)), "set row-major layout");
        }
        check(cublasLtMatmulPreferenceCreate(&preference), "create preference");
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace_limit, sizeof(workspace_limit)), "set workspace");
        candidates.resize(32);
        int returned = 0;
        check(cublasLtMatmulAlgoGetHeuristic(handle, operation, a, b, c, c, preference,
                                           static_cast<int>(candidates.size()), candidates.data(), &returned),
              "get algorithms");
        candidates.resize(returned);
    }

    ~Description() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};

struct DeviceState {
    cublasLtHandle_t handle = nullptr;
    std::map<std::string, std::unique_ptr<Description>> descriptions;
    DeviceState() { check(cublasLtCreate(&handle), "create cuBLASLt handle"); }
    ~DeviceState() { descriptions.clear(); if (handle) cublasLtDestroy(handle); }
};

// Cache only device handles and shape/dtype/algorithm metadata. Tensor values,
// outputs and workspaces are supplied afresh by every invocation.
std::map<int, std::unique_ptr<DeviceState>> devices;
std::mutex metadata_mutex;

cudaDataType_t data_type(const torch::Tensor& tensor) {
    if (tensor.scalar_type() == at::kFloat) return CUDA_R_32F;
    if (tensor.scalar_type() == at::kHalf) return CUDA_R_16F;
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, "expected FP16, BF16, or FP32 input");
    return CUDA_R_16BF;
}

std::pair<DeviceState*, Description*> description(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& c, size_t workspace_limit) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "all matrices must be CUDA tensors");
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(), "matrix devices must match");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "matrices must have rank two");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "matrices must be row-major contiguous");
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(), "matrix dtypes must match");
    TORCH_CHECK(a.size(1) == b.size(1) && c.size(0) == a.size(0) && c.size(1) == b.size(0), "matrix shapes mismatch");
    const auto type = data_type(a);
    std::ostringstream key;
    key << a.size(0) << ':' << b.size(0) << ':' << a.size(1) << ':' << static_cast<int>(type) << ':' << workspace_limit;
    std::lock_guard<std::mutex> lock(metadata_mutex);
    auto& device = devices[a.get_device()];
    if (!device) device = std::make_unique<DeviceState>();
    auto& entry = device->descriptions[key.str()];
    if (!entry) entry = std::make_unique<Description>(a.size(0), b.size(0), a.size(1), type, workspace_limit, device->handle);
    return {device.get(), entry.get()};
}

int attribute(const cublasLtMatmulAlgo_t& algorithm, cublasLtMatmulAlgoConfigAttributes_t name) {
    if (name == CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID || name == CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID) {
        uint16_t value = 0;
        size_t written = 0;
        check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, name, &value, sizeof(value), &written), "get algorithm attribute");
        return value;
    }
    int value = 0;
    size_t written = 0;
    check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, name, &value, sizeof(value), &written), "get algorithm attribute");
    return value;
}

pybind11::list algorithms(const torch::Tensor& a, const torch::Tensor& b,
                         const torch::Tensor& c, size_t workspace_limit) {
    c10::cuda::CUDAGuard guard(a.device());
    auto [device, entry] = description(a, b, c, workspace_limit);
    pybind11::list result;
    for (size_t i = 0; i < entry->candidates.size(); ++i) {
        const auto& candidate = entry->candidates[i];
        if (candidate.state != CUBLAS_STATUS_SUCCESS) continue;
        pybind11::dict record;
        record["index"] = i;
        record["workspace_bytes"] = candidate.workspaceSize;
        record["estimated_waves"] = candidate.wavesCount;
        const std::array<std::pair<const char*, cublasLtMatmulAlgoConfigAttributes_t>, 9> fields{{
            {"id", CUBLASLT_ALGO_CONFIG_ID}, {"tile", CUBLASLT_ALGO_CONFIG_TILE_ID},
            {"split_k", CUBLASLT_ALGO_CONFIG_SPLITK_NUM}, {"reduction", CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME},
            {"swizzle", CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING}, {"custom", CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION},
            {"stages", CUBLASLT_ALGO_CONFIG_STAGES_ID},
            {"inner_shape", CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID},
            {"cluster_shape", CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID},
        }};
        for (const auto& field : fields) record[field.first] = attribute(candidate.algo, field.second);
        result.append(record);
    }
    return result;
}

void matmul(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor c,
            torch::Tensor workspace, int algorithm_index) {
    c10::cuda::CUDAGuard guard(a.device());
    TORCH_CHECK(workspace.is_cuda() && workspace.device() == a.device() && workspace.is_contiguous()
                && workspace.scalar_type() == at::kByte, "workspace must be contiguous CUDA uint8 on the matrix device");
    const size_t workspace_limit = workspace.numel();
    auto [device, entry] = description(a, b, c, workspace_limit);
    TORCH_CHECK(algorithm_index >= 0 && algorithm_index < static_cast<int>(entry->candidates.size()), "algorithm index out of range");
    const auto& candidate = entry->candidates[algorithm_index];
    TORCH_CHECK(candidate.state == CUBLAS_STATUS_SUCCESS && candidate.workspaceSize <= workspace_limit, "algorithm is not valid for workspace");
    const float alpha = 1.0f, beta = 0.0f;
    check(cublasLtMatmul(device->handle, entry->operation,
                        &alpha, a.data_ptr(), entry->a, b.data_ptr(), entry->b,
                        &beta, c.data_ptr(), entry->c, c.data_ptr(), entry->c,
                        &candidate.algo, workspace.data_ptr(), workspace_limit,
                        c10::cuda::getCurrentCUDAStream(a.get_device()).stream()), "cuBLASLt matmul");
}

void matmul_config(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor c,
                   torch::Tensor workspace, const std::vector<int>& configuration) {
    c10::cuda::CUDAGuard guard(a.device());
    TORCH_CHECK(configuration.size() == 7 || configuration.size() == 9,
                "configuration order: ID, tile, splitK, reduction, swizzle, custom, stages, optional inner-shape and cluster-shape");
    TORCH_CHECK(workspace.is_cuda() && workspace.device() == a.device() && workspace.is_contiguous()
                && workspace.scalar_type() == at::kByte, "workspace must be contiguous CUDA uint8 on the matrix device");
    const size_t workspace_limit = workspace.numel();
    auto [device, entry] = description(a, b, c, workspace_limit);
    cublasLtMatmulHeuristicResult_t candidate;
    {
        std::lock_guard<std::mutex> lock(metadata_mutex);
        auto found = entry->configured.find(configuration);
        if (found == entry->configured.end()) {
            cublasLtMatmulAlgo_t algorithm;
            const auto type = data_type(a);
            check(cublasLtMatmulAlgoInit(device->handle, compute_type(type), CUDA_R_32F,
                                       type, type, type, type, configuration[0], &algorithm), "initialize explicit algorithm");
            const std::array<cublasLtMatmulAlgoConfigAttributes_t, 6> attributes{{
                CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_STAGES_ID,
            }};
            for (size_t i = 0; i < attributes.size(); ++i) {
                check(cublasLtMatmulAlgoConfigSetAttribute(&algorithm, attributes[i],
                                                         &configuration[i + 1], sizeof(int)), "set explicit algorithm attribute");
            }
            if (configuration.size() == 9) {
                const uint16_t inner_shape = configuration[7], cluster_shape = configuration[8];
                check(cublasLtMatmulAlgoConfigSetAttribute(&algorithm, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
                                                         &inner_shape, sizeof(inner_shape)), "set inner shape");
                check(cublasLtMatmulAlgoConfigSetAttribute(&algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
                                                         &cluster_shape, sizeof(cluster_shape)), "set cluster shape");
            }
            cublasLtMatmulHeuristicResult_t result{};
            check(cublasLtMatmulAlgoCheck(device->handle, entry->operation, entry->a, entry->b,
                                        entry->c, entry->c, &algorithm, &result), "check explicit algorithm");
            TORCH_CHECK(result.state == CUBLAS_STATUS_SUCCESS && result.workspaceSize <= workspace_limit,
                        "explicit algorithm is not valid for this shape/workspace");
            result.algo = algorithm;
            found = entry->configured.emplace(configuration, result).first;
        }
        candidate = found->second;
    }
    const float alpha = 1.0f, beta = 0.0f;
    check(cublasLtMatmul(device->handle, entry->operation,
                        &alpha, a.data_ptr(), entry->a, b.data_ptr(), entry->b,
                        &beta, c.data_ptr(), entry->c, c.data_ptr(), entry->c,
                        &candidate.algo, workspace.data_ptr(), workspace_limit,
                        c10::cuda::getCurrentCUDAStream(a.get_device()).stream()), "explicit cuBLASLt matmul");
}

}  // namespace sol_lt

#ifndef SOL_LT_NO_BINDING
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("algorithms", &sol_lt::algorithms);
    module.def("matmul", &sol_lt::matmul);
    module.def("matmul_config", &sol_lt::matmul_config);
    module.attr("cublaslt_version") = cublasLtGetVersion();
}
#endif
