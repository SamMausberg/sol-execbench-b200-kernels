// SPDX-License-Identifier: Apache-2.0
// Three ordinary matrix products and exact BF16 RMSNorm in one host call.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

void mla_normalize(const void* latent, const void* weight, void* output,
                   int rows, float epsilon, cudaStream_t stream);

pybind11::tuple run(const torch::Tensor& hidden_states,
                   const torch::Tensor& q_a_proj_weight,
                   const torch::Tensor& q_a_layernorm_weight,
                   const torch::Tensor& q_b_proj_weight,
                   const torch::Tensor& kv_a_proj_weight,
                   double rms_norm_eps) {
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
    auto x = hidden_states.reshape({rows, 7168});
    auto weight = q_a_layernorm_weight.contiguous();
    auto latent = torch::empty({rows, 1536}, hidden_states.options());
    auto normalized = torch::empty_like(latent);
    auto q = torch::empty({rows, 24576}, hidden_states.options());
    auto kv = torch::empty({rows, 576}, hidden_states.options());
    at::mm_out(latent, x, q_a_proj_weight.t());
    mla_normalize(latent.data_ptr(), weight.data_ptr(), normalized.data_ptr(),
                  static_cast<int>(rows), static_cast<float>(rms_norm_eps),
                  c10::cuda::getCurrentCUDAStream(hidden_states.get_device()).stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::mm_out(q, normalized, q_b_proj_weight.t());
    at::mm_out(kv, x, kv_a_proj_weight.t());
    q = q.view({batch, sequence, 128, 192});
    kv = kv.view({batch, sequence, 576});
    return pybind11::make_tuple(q.slice(3, 0, 128), q.slice(3, 128, 192),
                                kv.slice(2, 0, 512), kv.slice(2, 512, 576).unsqueeze(2));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
}
