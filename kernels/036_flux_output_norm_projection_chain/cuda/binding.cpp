// SPDX-License-Identifier: Apache-2.0
// Flux normalization chain: explicit FP32 kernels and per-operation cuBLASLt modes.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include "kernels.h"
#include <array>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace flux {
void check(cublasStatus_t status, const char* where) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, where, ": cuBLAS status ", int(status));
}

struct Description {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t a{}, b{}, c{};
    cublasLtMatmulPreference_t preference{};
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms;
    Description(int64_t m, int64_t n, int64_t k, cublasComputeType_t mode,
                cublasLtHandle_t handle, size_t workspace_bytes) {
        check(cublasLtMatmulDescCreate(&operation, mode, CUDA_R_32F), "operation");
        cublasOperation_t trans = CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                           &trans, sizeof(trans)), "transpose");
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                           &epilogue, sizeof(epilogue)), "bias epilogue");
        // C_col[N,M] = W_col[K,N]^T * X_col[K,M]. Bias is along N.
        check(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, k, n, k), "weight layout");
        check(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, k, m, k), "input layout");
        check(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, n, m, n), "output layout");
        check(cublasLtMatmulPreferenceCreate(&preference), "preference");
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_bytes, sizeof(workspace_bytes)), "workspace preference");
        algorithms.resize(16);int count=0;
        check(cublasLtMatmulAlgoGetHeuristic(handle, operation, a, b, c, c, preference,
                                            algorithms.size(), algorithms.data(), &count), "heuristics");
        algorithms.resize(count);
        TORCH_CHECK(count > 0, "No valid algorithm for the requested compute mode and bias epilogue");
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
    std::map<std::tuple<int64_t,int64_t,int64_t,int>,std::unique_ptr<Description>> descriptions;
    State() { check(cublasLtCreate(&handle), "handle"); }
    ~State() { descriptions.clear(); if (handle) cublasLtDestroy(handle); }
};
thread_local std::map<int,std::unique_ptr<State>> states;
constexpr size_t workspace_limit = 32 * 1024 * 1024;

Description& description(State& state, int64_t m, int64_t n, int64_t k, cublasComputeType_t mode) {
    auto key = std::make_tuple(m,n,k,int(mode));
    auto& entry=state.descriptions[key];
    if (!entry) entry=std::make_unique<Description>(m,n,k,mode,state.handle,workspace_limit);
    return *entry;
}

void set_bias(Description& desc, const void* pointer) {
    check(cublasLtMatmulDescSetAttribute(desc.operation,CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &pointer,sizeof(pointer)),"bias pointer");
}

void multiply(State& state, Description& desc, const torch::Tensor& input,
              const torch::Tensor& weight, torch::Tensor& output, torch::Tensor& workspace,
              cudaStream_t stream, int index) {
    TORCH_CHECK(index >= 0 && index < int(desc.algorithms.size()), "algorithm index out of range");
    const auto& algorithm=desc.algorithms[index];
    TORCH_CHECK(algorithm.state == CUBLAS_STATUS_SUCCESS && algorithm.workspaceSize <= workspace_limit,
                "unsupported algorithm/workspace");
    const float alpha=1.0f,beta=0.0f;
    check(cublasLtMatmul(state.handle,desc.operation,&alpha,
                        weight.data_ptr(),desc.a,input.data_ptr(),desc.b,&beta,
                        output.data_ptr(),desc.c,output.data_ptr(),desc.c,&algorithm.algo,
                        workspace.data_ptr(),workspace_limit,stream),"matmul");
}

void configured(torch::Tensor hidden, torch::Tensor temb, torch::Tensor weight,
                torch::Tensor bias, torch::Tensor projection_weight, torch::Tensor projection_bias,
                double eps, torch::Tensor output, int precision, int mod_algorithm, int out_algorithm) {
    c10::cuda::CUDAGuard guard(hidden.device());
    for (const auto& t : {hidden,temb,weight,bias,projection_weight,projection_bias,output})
        TORCH_CHECK(t.is_cuda() && t.device()==hidden.device() && t.scalar_type()==at::kFloat,
                    "Expected FP32 tensors on one CUDA device");
    if (!hidden.is_contiguous() || !temb.is_contiguous() || !weight.is_contiguous() ||
        !bias.is_contiguous() || !projection_weight.is_contiguous() || !projection_bias.is_contiguous() || !output.is_contiguous()) {
        auto temporary = torch::empty(output.sizes(), output.options());
        configured(hidden.contiguous(),temb.contiguous(),weight.contiguous(),bias.contiguous(),
                   projection_weight.contiguous(),projection_bias.contiguous(),eps,temporary,precision,mod_algorithm,out_algorithm);
        output.copy_(temporary);
        return;
    }
    TORCH_CHECK(hidden.dim()==3 && hidden.size(2)==3072 && projection_weight.size(0)==64,"Unsupported dimensions");
    int batch=hidden.size(0), sequence=hidden.size(1), rows=batch*sequence;
    auto silu=torch::empty_like(temb);
    auto modulation=torch::empty({batch,6144},hidden.options());
    auto adapted=torch::empty_like(hidden);
    auto workspace=torch::empty({int64_t(workspace_limit)},hidden.options().dtype(at::kByte));
    auto& state=states[hidden.get_device()];
    if (!state) state=std::make_unique<State>();
    auto mode=precision==0 ? CUBLAS_COMPUTE_32F : precision==1 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F_EMULATED_16BFX9;
    auto& modulation_desc=description(*state,batch,6144,3072,mode);
    auto& output_desc=description(*state,rows,64,3072,mode);
    set_bias(modulation_desc,bias.data_ptr());set_bias(output_desc,projection_bias.data_ptr());
    auto stream=c10::cuda::getCurrentCUDAStream(hidden.get_device()).stream();
    launch_silu(temb.data_ptr<float>(),silu.data_ptr<float>(),batch*3072,stream);
    multiply(*state,modulation_desc,silu,weight,modulation,workspace,stream,mod_algorithm);
    launch_norm(hidden.data_ptr<float>(),modulation.data_ptr<float>(),adapted.data_ptr<float>(),rows,sequence,float(eps),stream);
    multiply(*state,output_desc,adapted,projection_weight,output,workspace,stream,out_algorithm);
    set_bias(modulation_desc,nullptr);set_bias(output_desc,nullptr);
    auto error=cudaGetLastError();TORCH_CHECK(error==cudaSuccess,"CUDA kernel failure: ",cudaGetErrorString(error));
}

pybind11::dict algorithms(torch::Tensor hidden, int precision) {
    c10::cuda::CUDAGuard guard(hidden.device());
    auto& state=states[hidden.get_device()];
    if (!state) state=std::make_unique<State>();
    auto mode=precision==0 ? CUBLAS_COMPUTE_32F : precision==1 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F_EMULATED_16BFX9;
    pybind11::dict result;
    for (int kind=0;kind<2;++kind) {
        auto& desc=description(*state,kind ? hidden.size(0)*hidden.size(1) : hidden.size(0),kind ? 64 : 6144,3072,mode);
        pybind11::list entries;
        for (size_t i=0;i<desc.algorithms.size();++i) {
            auto& candidate=desc.algorithms[i];
            pybind11::dict entry;entry["index"]=i;entry["workspace_bytes"]=candidate.workspaceSize;entry["waves"]=candidate.wavesCount;
            int id=0;size_t written=0;
            check(cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written),"algorithm id");
            entry["id"]=id;entries.append(entry);
        }
        result[kind ? "output" : "modulation"]=entries;
    }
    result["cublaslt_version"]=cublasLtGetVersion();
    return result;
}

void run(torch::Tensor h,torch::Tensor t,torch::Tensor w,torch::Tensor b,torch::Tensor pw,
         torch::Tensor pb,double eps,torch::Tensor o) { configured(h,t,w,b,pw,pb,eps,o,2,0,0); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("algorithms",&flux::algorithms);
    module.def("run",&flux::run,pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("configured",&flux::configured,pybind11::call_guard<pybind11::gil_scoped_release>());
}
