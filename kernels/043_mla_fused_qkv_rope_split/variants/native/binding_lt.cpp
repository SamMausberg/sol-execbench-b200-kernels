// SPDX-License-Identifier: Apache-2.0
// Direct cuBLASLt projections and exact BF16 RMSNorm in one host call.
#define SOL_LT_NO_BINDING
#include "cublaslt_bridge.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

void mla_normalize(const void* latent, const void* weight, void* output,
                   int rows, float epsilon, cudaStream_t stream);

pybind11::tuple run_configurable(const torch::Tensor& hidden_states,
                   const torch::Tensor& q_a_proj_weight,
                   const torch::Tensor& q_a_layernorm_weight,
                   const torch::Tensor& q_b_proj_weight,
                   const torch::Tensor& kv_a_proj_weight,
                   double rms_norm_eps, const std::vector<int>& algorithms) {
    TORCH_CHECK(algorithms.size() == 3, "expected three algorithm indices");
    at::NoGradGuard no_grad;
    TORCH_CHECK(hidden_states.is_cuda() && hidden_states.scalar_type() == at::kBFloat16,
                "expected CUDA BF16 hidden states");
    TORCH_CHECK(hidden_states.dim() == 3 && hidden_states.size(2) == 7168,
                "expected hidden states [batch, sequence, 7168]");
    TORCH_CHECK(q_a_layernorm_weight.device() == hidden_states.device() &&
                q_a_layernorm_weight.scalar_type() == at::kBFloat16 &&
                q_a_layernorm_weight.numel() == 1536, "invalid norm weight");
    c10::cuda::CUDAGuard guard(hidden_states.device());
    const auto batch = hidden_states.size(0), sequence = hidden_states.size(1);
    const auto rows = batch * sequence;
    auto x = hidden_states.reshape({rows, 7168}).contiguous();
    auto qa_weight = q_a_proj_weight.contiguous();
    auto qb_weight = q_b_proj_weight.contiguous();
    auto kv_weight = kv_a_proj_weight.contiguous();
    auto weight = q_a_layernorm_weight.contiguous();
    auto latent = torch::empty({rows, 1536}, hidden_states.options());
    auto normalized = torch::empty_like(latent);
    auto q = torch::empty({rows, 24576}, hidden_states.options());
    auto kv = torch::empty({rows, 576}, hidden_states.options());
    auto workspace = torch::empty({32 * 1024 * 1024}, hidden_states.options().dtype(at::kByte));
    sol_lt::matmul(x, qa_weight, latent, workspace, algorithms[0]);
    mla_normalize(latent.data_ptr(), weight.data_ptr(), normalized.data_ptr(),
                  static_cast<int>(rows), static_cast<float>(rms_norm_eps),
                  c10::cuda::getCurrentCUDAStream(hidden_states.get_device()).stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    sol_lt::matmul(normalized, qb_weight, q, workspace, algorithms[1]);
    sol_lt::matmul(x, kv_weight, kv, workspace, algorithms[2]);
    q = q.view({batch, sequence, 128, 192});
    kv = kv.view({batch, sequence, 576});
    return pybind11::make_tuple(q.slice(3, 0, 128), q.slice(3, 128, 192),
                                kv.slice(2, 0, 512), kv.slice(2, 512, 576).unsqueeze(2));
}

pybind11::tuple run(const torch::Tensor& hidden_states,
                   const torch::Tensor& q_a_proj_weight,
                   const torch::Tensor& q_a_layernorm_weight,
                   const torch::Tensor& q_b_proj_weight,
                   const torch::Tensor& kv_a_proj_weight,
                   double rms_norm_eps) {
    return run_configurable(hidden_states, q_a_proj_weight, q_a_layernorm_weight,
                            q_b_proj_weight, kv_a_proj_weight, rms_norm_eps, {0, 0, 0});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run_configurable", &run_configurable);
    module.def("algorithms", &sol_lt::algorithms);
    module.def("matmul", &sol_lt::matmul);
    module.attr("cublaslt_version") = cublasLtGetVersion();
    module.def("run", &run);
}
