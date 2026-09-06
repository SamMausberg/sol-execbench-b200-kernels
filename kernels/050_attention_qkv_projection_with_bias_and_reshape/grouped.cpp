#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

struct GroupedPlan {
    cublasHandle_t handle;
    cublasPointerMode_t previous_mode;
    int rows;
};

std::shared_ptr<GroupedPlan> prepare(int rows) {
    auto plan = std::make_shared<GroupedPlan>();
    plan->handle = at::cuda::getCurrentCUDABlasHandle();
    plan->rows = rows;
    TORCH_CHECK(cublasGetPointerMode(plan->handle, &plan->previous_mode) == CUBLAS_STATUS_SUCCESS,
                "Could not query cuBLAS pointer mode");
    TORCH_CHECK(cublasSetPointerMode(plan->handle, CUBLAS_POINTER_MODE_HOST) == CUBLAS_STATUS_SUCCESS,
                "Could not set cuBLAS pointer mode");
    return plan;
}

void execute(const std::shared_ptr<GroupedPlan>& plan, torch::Tensor pointers) {
    TORCH_CHECK(pointers.is_cuda() && pointers.scalar_type() == torch::kInt64
                && pointers.numel() == 9, "Expected nine CUDA pointers");
    int rows = plan->rows;
    const cublasOperation_t trans_a[] = {CUBLAS_OP_T, CUBLAS_OP_T};
    const cublasOperation_t trans_b[] = {CUBLAS_OP_N, CUBLAS_OP_N};
    const int m[] = {1024, 256}, n[] = {rows, rows}, k[] = {640, 640};
    const int lda[] = {640, 640}, ldb[] = {640, 640}, ldc[] = {1024, 256};
    const int group_size[] = {1, 2};
    const float alpha[] = {1.0f, 1.0f}, beta[] = {0.0f, 0.0f};
    auto pointer_data = pointers.data_ptr<int64_t>();
    auto status = cublasGemmGroupedBatchedEx(
        plan->handle, trans_a, trans_b, m, n, k, alpha,
        reinterpret_cast<const void* const*>(pointer_data), CUDA_R_16BF, lda,
        reinterpret_cast<const void* const*>(pointer_data + 3), CUDA_R_16BF, ldb,
        beta, reinterpret_cast<void* const*>(pointer_data + 6), CUDA_R_16BF, ldc,
        2, group_size, CUBLAS_COMPUTE_32F);
    auto restored = cublasSetPointerMode(plan->handle, plan->previous_mode);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS grouped GEMM failed: ", int(status));
    TORCH_CHECK(restored == CUBLAS_STATUS_SUCCESS, "Could not restore cuBLAS pointer mode");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<GroupedPlan, std::shared_ptr<GroupedPlan>>(module, "GroupedPlan");
    module.def("prepare", &prepare);
    module.def("execute", &execute);
}
