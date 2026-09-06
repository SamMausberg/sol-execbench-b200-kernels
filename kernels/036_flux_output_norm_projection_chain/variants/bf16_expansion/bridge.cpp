// SPDX-License-Identifier: Apache-2.0
// Experimental FP32 linear layer through two BF16 fragments per operand.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <array>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

void launch_split_pair(const float*,const float*,void*,void*,void*,void*,int64_t,int64_t,cudaStream_t);

namespace expansion {
constexpr size_t workspace_limit = 32 * 1024 * 1024;

void check(cublasStatus_t status, const char* where) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, where, ": cuBLAS status ", int(status));
}


struct Description {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t a{}, b{}, c{};
    cublasLtMatmulPreference_t preference{};
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms;
    Description(int64_t m, int64_t n, int64_t k, cudaDataType_t input_type,
                cublasComputeType_t compute, bool biased, cublasLtHandle_t handle) {
        check(cublasLtMatmulDescCreate(&operation, compute, CUDA_R_32F), "operation");
        cublasOperation_t trans = CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                           &trans, sizeof(trans)), "transpose");
        if (biased) {
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                               &epilogue, sizeof(epilogue)), "bias epilogue");
            cudaDataType_t bias_type = CUDA_R_32F;
            check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                               &bias_type, sizeof(bias_type)), "bias type");
        }
        // Row-major C[M,N] is viewed as column-major C[N,M].
        check(cublasLtMatrixLayoutCreate(&a, input_type, k, n, k), "weight layout");
        check(cublasLtMatrixLayoutCreate(&b, input_type, k, m, k), "input layout");
        check(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, n, m, n), "output layout");
        check(cublasLtMatmulPreferenceCreate(&preference), "preference");
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_limit, sizeof(workspace_limit)), "workspace preference");
        algorithms.resize(16);
        int count = 0;
        check(cublasLtMatmulAlgoGetHeuristic(handle, operation, a, b, c, c, preference,
                                            algorithms.size(), algorithms.data(), &count), "heuristics");
        algorithms.resize(count);
        TORCH_CHECK(count > 0, "No algorithm for this shape, dtype, and epilogue");
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
    std::map<std::tuple<int64_t,int64_t,int64_t,int,int,bool>,std::unique_ptr<Description>> descriptions;
    State() { check(cublasLtCreate(&handle), "handle"); }
    ~State() { descriptions.clear(); if (handle) cublasLtDestroy(handle); }
};
// Cache only CPU descriptors and algorithms. No input-derived GPU state is cached.
thread_local std::map<int,std::unique_ptr<State>> states;

State& state(int device) {
    auto& result = states[device];
    if (!result) result = std::make_unique<State>();
    return *result;
}

Description& description(State& state, int64_t m, int64_t n, int64_t k,
                         cudaDataType_t input_type, bool biased) {
    auto compute = input_type == CUDA_R_32F ? CUBLAS_COMPUTE_32F_EMULATED_16BFX9 : CUBLAS_COMPUTE_32F;
    auto key = std::make_tuple(m,n,k,int(input_type),int(compute),biased);
    auto& result = state.descriptions[key];
    if (!result) result = std::make_unique<Description>(m,n,k,input_type,compute,biased,state.handle);
    return *result;
}

void check_matrix(const torch::Tensor& value, at::ScalarType type, int device) {
    TORCH_CHECK(value.is_cuda() && value.get_device() == device && value.scalar_type() == type
                && value.is_contiguous() && value.dim() == 2, "Expected contiguous matrices on one CUDA device");
}

void split(torch::Tensor a, torch::Tensor b, torch::Tensor ah, torch::Tensor al,
           torch::Tensor bh, torch::Tensor bl) {
    c10::cuda::CUDAGuard guard(a.device());
    check_matrix(a,at::kFloat,a.get_device()); check_matrix(b,at::kFloat,a.get_device());
    for (const auto& item : {ah,al,bh,bl}) check_matrix(item,at::kBFloat16,a.get_device());
    TORCH_CHECK(ah.sizes()==a.sizes() && al.sizes()==a.sizes() && bh.sizes()==b.sizes() && bl.sizes()==b.sizes(),
                "Fragment shapes must match source matrices");
    auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
    launch_split_pair(a.data_ptr<float>(),b.data_ptr<float>(),ah.data_ptr(),al.data_ptr(),
                      bh.data_ptr(),bl.data_ptr(),a.numel(),b.numel(),stream);
    auto error=cudaGetLastError();TORCH_CHECK(error==cudaSuccess,"split failed: ",cudaGetErrorString(error));
}

void multiply(State& state, Description& desc, const torch::Tensor& input, const torch::Tensor& weight,
              const torch::Tensor& output, const torch::Tensor& workspace, cudaStream_t stream,
              int algorithm_index, float beta, const void* bias) {
    TORCH_CHECK(algorithm_index >= 0 && algorithm_index < int(desc.algorithms.size()), "algorithm index out of range");
    const auto& algorithm = desc.algorithms[algorithm_index];
    TORCH_CHECK(algorithm.state == CUBLAS_STATUS_SUCCESS && algorithm.workspaceSize <= workspace.numel(),
                "Unsupported algorithm/workspace");
    if (bias) check(cublasLtMatmulDescSetAttribute(desc.operation,CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                 &bias,sizeof(bias)),"bias pointer");
    const float alpha=1.0f;
    check(cublasLtMatmul(state.handle,desc.operation,&alpha,weight.data_ptr(),desc.a,input.data_ptr(),desc.b,&beta,
                        output.data_ptr(),desc.c,output.data_ptr(),desc.c,&algorithm.algo,
                        workspace.data_ptr(),workspace.numel(),stream),"matmul");
    if (bias) {
        const void* empty = nullptr;
        check(cublasLtMatmulDescSetAttribute(desc.operation,CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &empty,sizeof(empty)),"clear bias pointer");
    }
}

void check_output(torch::Tensor a, torch::Tensor b, torch::Tensor bias,
                   torch::Tensor output, torch::Tensor workspace) {
    check_matrix(output,at::kFloat,a.get_device());
    TORCH_CHECK(a.size(1)==b.size(1) && output.size(0)==a.size(0) && output.size(1)==b.size(0), "Shape mismatch");
    TORCH_CHECK(bias.is_cuda() && bias.device()==a.device() && bias.scalar_type()==at::kFloat
                && bias.is_contiguous() && bias.numel()==b.size(0), "Invalid FP32 bias");
    TORCH_CHECK(workspace.is_cuda() && workspace.device()==a.device() && workspace.scalar_type()==at::kByte
                && workspace.is_contiguous() && workspace.numel() >= workspace_limit,"Invalid workspace");
}

void products(torch::Tensor ah, torch::Tensor al, torch::Tensor bh, torch::Tensor bl,
               torch::Tensor bias, torch::Tensor output, torch::Tensor workspace,
               int terms, int plain_algorithm, int bias_algorithm) {
    c10::cuda::CUDAGuard guard(ah.device());
    for (const auto& item : {ah,al,bh,bl}) check_matrix(item,at::kBFloat16,ah.get_device());
    TORCH_CHECK(ah.sizes()==al.sizes() && bh.sizes()==bl.sizes(), "Fragment shapes differ");
    TORCH_CHECK(terms==3 || terms==4, "Expected three or four products");
    check_output(ah,bh,bias,output,workspace);
    auto& current = state(ah.get_device());
    auto& plain = description(current,ah.size(0),bh.size(0),ah.size(1),CUDA_R_16BF,false);
    auto& biased = description(current,ah.size(0),bh.size(0),ah.size(1),CUDA_R_16BF,true);
    auto stream = c10::cuda::getCurrentCUDAStream(ah.get_device()).stream();
    if (terms==4) multiply(current,plain,al,bl,output,workspace,stream,plain_algorithm,0.0f,nullptr);
    multiply(current,plain,al,bh,output,workspace,stream,plain_algorithm,terms==4 ? 1.0f : 0.0f,nullptr);
    multiply(current,plain,ah,bl,output,workspace,stream,plain_algorithm,1.0f,nullptr);
    multiply(current,biased,ah,bh,output,workspace,stream,bias_algorithm,1.0f,bias.data_ptr());
}

void expanded(torch::Tensor a, torch::Tensor b, torch::Tensor bias, torch::Tensor output,
               torch::Tensor ah, torch::Tensor al, torch::Tensor bh, torch::Tensor bl,
               torch::Tensor workspace, int terms, int plain_algorithm, int bias_algorithm) {
    split(a,b,ah,al,bh,bl);
    products(ah,al,bh,bl,bias,output,workspace,terms,plain_algorithm,bias_algorithm);
}

void baseline(torch::Tensor a, torch::Tensor b, torch::Tensor bias, torch::Tensor output,
               torch::Tensor workspace, int algorithm_index) {
    c10::cuda::CUDAGuard guard(a.device());
    check_matrix(a,at::kFloat,a.get_device());check_matrix(b,at::kFloat,a.get_device());
    check_output(a,b,bias,output,workspace);
    auto& current = state(a.get_device());
    auto& desc = description(current,a.size(0),b.size(0),a.size(1),CUDA_R_32F,true);
    multiply(current,desc,a,b,output,workspace,c10::cuda::getCurrentCUDAStream(a.get_device()).stream(),
             algorithm_index,0.0f,bias.data_ptr());
}

int attribute(const cublasLtMatmulAlgo_t& algorithm, cublasLtMatmulAlgoConfigAttributes_t name) {
    size_t written = 0;
    if (name==CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID || name==CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID) {
        uint16_t value=0;
        check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm,name,&value,sizeof(value),&written),"algorithm attribute");
        return value;
    }
    int value=0;
    check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm,name,&value,sizeof(value),&written),"algorithm attribute");
    return value;
}

pybind11::list algorithms(int64_t m,int64_t n,int64_t k,bool fp32,bool biased,int device) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    auto& desc = description(state(device),m,n,k,fp32 ? CUDA_R_32F : CUDA_R_16BF,biased);
    pybind11::list result;
    const std::array<std::pair<const char*,cublasLtMatmulAlgoConfigAttributes_t>,9> fields{{
        {"id",CUBLASLT_ALGO_CONFIG_ID},{"tile",CUBLASLT_ALGO_CONFIG_TILE_ID},
        {"split_k",CUBLASLT_ALGO_CONFIG_SPLITK_NUM},{"reduction",CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME},
        {"swizzle",CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING},{"custom",CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION},
        {"stages",CUBLASLT_ALGO_CONFIG_STAGES_ID},{"inner_shape",CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID},
        {"cluster_shape",CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID},
    }};
    for (size_t index=0;index<desc.algorithms.size();++index) {
        auto& candidate=desc.algorithms[index];
        pybind11::dict entry;
        entry["index"]=index;entry["workspace_bytes"]=candidate.workspaceSize;
        entry["estimated_waves"]=candidate.wavesCount;
        for (auto field:fields) entry[field.first]=attribute(candidate.algo,field.second);
        result.append(entry);
    }
    return result;
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("split",&expansion::split,pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("products",&expansion::products,pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("expanded",&expansion::expanded,pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("baseline",&expansion::baseline,pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("algorithms",&expansion::algorithms);
    module.attr("cublaslt_version")=cublasLtGetVersion();
}
