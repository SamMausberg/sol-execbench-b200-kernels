// SPDX-License-Identifier: Apache-2.0
// Batched products write each head into disjoint columns of the final output.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>

#include <map>
#include <memory>
#include <mutex>

void make_av_pointer_arrays(const void* weights, const void* values, void* output,
                            void* pointers, int batch, int sequence, cudaStream_t stream);

void check(cublasStatus_t status, const char* name) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, name, " failed with cuBLAS status ", int(status));
}

struct Handle {
    cublasHandle_t value = nullptr;
    std::mutex mutex;
    Handle() { check(cublasCreate(&value), "create handle"); }
    ~Handle() { if (value) cublasDestroy(value); }
};

std::map<int, std::unique_ptr<Handle>> handles;
std::mutex handles_mutex;

Handle* handle_for(int device) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto& entry = handles[device];
    if (!entry) entry = std::make_unique<Handle>();
    return entry.get();
}

torch::Tensor run(const torch::Tensor& attn_weights, const torch::Tensor& value_states) {
    at::NoGradGuard no_grad;
    TORCH_CHECK(attn_weights.is_cuda() && value_states.device() == attn_weights.device(),
                "expected tensors on one CUDA device");
    TORCH_CHECK(attn_weights.scalar_type() == at::kBFloat16 && value_states.scalar_type() == at::kBFloat16,
                "expected BF16 inputs");
    TORCH_CHECK(attn_weights.dim() == 4 && value_states.dim() == 4 && attn_weights.size(1) == 40,
                "expected attention tensors with 40 heads");
    const int batch = attn_weights.size(0), sequence = attn_weights.size(2);
    TORCH_CHECK(attn_weights.size(3) == sequence && value_states.size(0) == batch &&
                value_states.size(1) == 40 && value_states.size(2) == sequence && value_states.size(3) == 128,
                "invalid attention/value shape");
    if (!attn_weights.is_contiguous() || !value_states.is_contiguous() ||
        reinterpret_cast<uintptr_t>(attn_weights.data_ptr()) % 16 ||
        reinterpret_cast<uintptr_t>(value_states.data_ptr()) % 16) {
        return at::matmul(attn_weights, value_states).transpose(1, 2).reshape({batch, sequence, 5120});
    }
    c10::cuda::CUDAGuard guard(attn_weights.device());
    auto output = torch::empty({batch, sequence, 5120}, attn_weights.options());
    auto workspace = torch::empty({32 * 1024 * 1024}, attn_weights.options().dtype(at::kByte));
    auto* state = handle_for(attn_weights.get_device());
    std::lock_guard<std::mutex> lock(state->mutex);
    const auto stream = c10::cuda::getCurrentCUDAStream(attn_weights.get_device()).stream();
    check(cublasSetStream(state->value, stream), "set stream");
    check(cublasSetWorkspace(state->value, workspace.data_ptr(), workspace.numel()), "set workspace");
    const float alpha = 1.0f, beta = 0.0f;
    // Column-major [128, S] = V.T @ attention.T. The large C leading
    // dimension interleaves heads while keeping their actual elements disjoint.
    if (batch == 1) {
        check(cublasGemmStridedBatchedEx(
            state->value, CUBLAS_OP_N, CUBLAS_OP_N, 128, sequence, sequence,
            &alpha, value_states.data_ptr(), CUDA_R_16BF, 128, int64_t(sequence) * 128,
            attn_weights.data_ptr(), CUDA_R_16BF, sequence, int64_t(sequence) * sequence,
            &beta, output.data_ptr(), CUDA_R_16BF, 5120, 128, 40,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "strided batched AV");
    } else {
        auto pointers = torch::empty({3, batch * 40}, attn_weights.options().dtype(at::kLong));
        make_av_pointer_arrays(attn_weights.data_ptr(), value_states.data_ptr(), output.data_ptr(),
                               pointers.data_ptr(), batch, sequence, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        auto* table = pointers.data_ptr<int64_t>();
        check(cublasGemmBatchedEx(
            state->value, CUBLAS_OP_N, CUBLAS_OP_N, 128, sequence, sequence,
            &alpha, reinterpret_cast<const void* const*>(table), CUDA_R_16BF, 128,
            reinterpret_cast<const void* const*>(table + batch * 40), CUDA_R_16BF, sequence,
            &beta, reinterpret_cast<void* const*>(table + 2 * batch * 40), CUDA_R_16BF, 5120,
            batch * 40, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "pointer batched AV");
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
}
