// SPDX-License-Identifier: Apache-2.0
// Experimental FP32 linear layer through three BF16 fragments and six triangular products.
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

void launch_split_pair(const float*,const float*,void*,void*,int64_t,int64_t,cudaStream_t);
void launch_reduce_five(const float*,float*,int64_t,cudaStream_t);

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
                cublasComputeType_t compute, bool biased, int batches, cublasLtHandle_t handle) {
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
        if (batches > 1) {
            int64_t strides[3] = {n*k,m*k,m*n};
            cublasLtMatrixLayout_t layouts[3] = {a,b,c};
            for (int i=0;i<3;++i) {
                check(cublasLtMatrixLayoutSetAttribute(layouts[i],CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                      &batches,sizeof(batches)),"batch count");
                check(cublasLtMatrixLayoutSetAttribute(layouts[i],CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                      &strides[i],sizeof(strides[i])),"batch stride");
            }
        }
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
    std::map<std::tuple<int64_t,int64_t,int64_t,int,int,bool,int>,std::unique_ptr<Description>> descriptions;
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
                         cudaDataType_t input_type, bool biased, int batches=1) {
    auto compute = input_type == CUDA_R_32F ? CUBLAS_COMPUTE_32F_EMULATED_16BFX9 : CUBLAS_COMPUTE_32F;
    auto key = std::make_tuple(m,n,k,int(input_type),int(compute),biased,batches);
    auto& result = state.descriptions[key];
    if (!result) result = std::make_unique<Description>(m,n,k,input_type,compute,biased,batches,state.handle);
    return *result;
}

void check_matrix(const torch::Tensor& value, at::ScalarType type, int device) {
    TORCH_CHECK(value.is_cuda() && value.get_device() == device && value.scalar_type() == type
                && value.is_contiguous() && value.dim() == 2, "Expected contiguous matrices on one CUDA device");
}

void split(torch::Tensor a, torch::Tensor b, torch::Tensor ap, torch::Tensor bp) {
    c10::cuda::CUDAGuard guard(a.device());
    check_matrix(a,at::kFloat,a.get_device());check_matrix(b,at::kFloat,a.get_device());
    for (const auto& item : {ap,bp})
        TORCH_CHECK(item.is_cuda() && item.device()==a.device() && item.scalar_type()==at::kBFloat16
                    && item.is_contiguous() && item.dim()==3 && item.size(0)==3,"Expected three BF16 planes");
    TORCH_CHECK(ap.size(1)==a.size(0) && ap.size(2)==a.size(1)
                && bp.size(1)==b.size(0) && bp.size(2)==b.size(1),"Fragment shape mismatch");
    launch_split_pair(a.data_ptr<float>(),b.data_ptr<float>(),ap.data_ptr(),bp.data_ptr(),a.numel(),b.numel(),
                       c10::cuda::getCurrentCUDAStream(a.get_device()).stream());
    auto error=cudaGetLastError();TORCH_CHECK(error==cudaSuccess,"split failed: ",cudaGetErrorString(error));
}

void multiply(State& state, Description& desc, const void* input, const void* weight,
              void* output, const torch::Tensor& workspace, cudaStream_t stream,
              int algorithm_index, float beta, const void* bias) {
    TORCH_CHECK(algorithm_index >= 0 && algorithm_index < int(desc.algorithms.size()), "algorithm index out of range");
    const auto& algorithm = desc.algorithms[algorithm_index];
    TORCH_CHECK(algorithm.state == CUBLAS_STATUS_SUCCESS && algorithm.workspaceSize <= workspace.numel(),
                "Unsupported algorithm/workspace");
    if (bias) check(cublasLtMatmulDescSetAttribute(desc.operation,CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                 &bias,sizeof(bias)),"bias pointer");
    const float alpha=1.0f;
    check(cublasLtMatmul(state.handle,desc.operation,&alpha,weight,desc.a,input,desc.b,&beta,
                        output,desc.c,output,desc.c,&algorithm.algo,
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

void products(torch::Tensor ap, torch::Tensor bp, torch::Tensor bias, torch::Tensor output,
               torch::Tensor partials, torch::Tensor workspace, int strategy,
               int plain_algorithm, int bias_algorithm) {
    c10::cuda::CUDAGuard guard(ap.device());
    TORCH_CHECK(strategy==0 || strategy==1,"Expected sequential or batched strategy");
    for (const auto& item : {ap,bp})
        TORCH_CHECK(item.is_cuda() && item.device()==ap.device() && item.scalar_type()==at::kBFloat16
                    && item.is_contiguous() && item.dim()==3 && item.size(0)==3,"Expected three BF16 planes");
    auto av=ap.select(0,0), bv=bp.select(0,0);
    check_output(av,bv,bias,output,workspace);
    int64_t m=ap.size(1),n=bp.size(1),k=ap.size(2);
    auto* a0=static_cast<const char*>(ap.data_ptr());
    auto* a1=a0+2*m*k;
    auto* a2=a1+2*m*k;
    auto* b2=static_cast<const char*>(bp.data_ptr());
    auto* b1=b2+2*n*k;
    auto* b0=b1+2*n*k;
    auto& current=state(ap.get_device());
    auto& biased=description(current,m,n,k,CUDA_R_16BF,true);
    auto stream=c10::cuda::getCurrentCUDAStream(ap.get_device()).stream();
    if (strategy==0) {
        auto& plain=description(current,m,n,k,CUDA_R_16BF,false);
        multiply(current,plain,a0,b2,output.data_ptr(),workspace,stream,plain_algorithm,0.0f,nullptr);
        multiply(current,plain,a1,b1,output.data_ptr(),workspace,stream,plain_algorithm,1.0f,nullptr);
        multiply(current,plain,a2,b0,output.data_ptr(),workspace,stream,plain_algorithm,1.0f,nullptr);
        multiply(current,plain,a0,b1,output.data_ptr(),workspace,stream,plain_algorithm,1.0f,nullptr);
        multiply(current,plain,a1,b0,output.data_ptr(),workspace,stream,plain_algorithm,1.0f,nullptr);
    } else {
        TORCH_CHECK(partials.is_cuda() && partials.device()==ap.device() && partials.scalar_type()==at::kFloat
                    && partials.is_contiguous() && partials.numel()==5*m*n,"Expected five FP32 partial matrices");
        auto& three=description(current,m,n,k,CUDA_R_16BF,false,3);
        auto& two=description(current,m,n,k,CUDA_R_16BF,false,2);
        auto* partial=partials.data_ptr<float>();
        multiply(current,three,a0,b2,partial,workspace,stream,plain_algorithm,0.0f,nullptr);
        multiply(current,two,a0,b1,partial+3*m*n,workspace,stream,plain_algorithm,0.0f,nullptr);
        launch_reduce_five(partial,output.data_ptr<float>(),m*n,stream);
        auto error=cudaGetLastError();TORCH_CHECK(error==cudaSuccess,"reduce failed: ",cudaGetErrorString(error));
    }
    multiply(current,biased,a0,b0,output.data_ptr(),workspace,stream,bias_algorithm,1.0f,bias.data_ptr());
}

void expanded(torch::Tensor a, torch::Tensor b, torch::Tensor bias, torch::Tensor output,
               torch::Tensor ap, torch::Tensor bp, torch::Tensor partials, torch::Tensor workspace,
               int strategy, int plain_algorithm, int bias_algorithm) {
    split(a,b,ap,bp);
    products(ap,bp,bias,output,partials,workspace,strategy,plain_algorithm,bias_algorithm);
}

void baseline(torch::Tensor a, torch::Tensor b, torch::Tensor bias, torch::Tensor output,
               torch::Tensor workspace, int algorithm_index) {
    c10::cuda::CUDAGuard guard(a.device());
    check_matrix(a,at::kFloat,a.get_device());check_matrix(b,at::kFloat,a.get_device());
    check_output(a,b,bias,output,workspace);
    auto& current = state(a.get_device());
    auto& desc = description(current,a.size(0),b.size(0),a.size(1),CUDA_R_32F,true);
    multiply(current,desc,a.data_ptr(),b.data_ptr(),output.data_ptr(),workspace,c10::cuda::getCurrentCUDAStream(a.get_device()).stream(),
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

pybind11::list algorithms(int64_t m,int64_t n,int64_t k,bool fp32,bool biased,int device,int batches) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    auto& desc = description(state(device),m,n,k,fp32 ? CUDA_R_32F : CUDA_R_16BF,biased,batches);
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
