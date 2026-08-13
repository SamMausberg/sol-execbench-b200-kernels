// SPDX-License-Identifier: Apache-2.0

#include "kernel.cuh"

#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"

namespace sol179_detail {

using namespace cute;

constexpr int kHidden = 3584;
constexpr int kIntermediate = 18944;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentD = 8;

// The benchmark supplies ordinary contiguous [M, K/128] and [N/128, K/128]
// tensors. K-major compact scale layouts therefore match both buffers exactly.
template <class Arch>
using ScaleConfig = cutlass::detail::Sm1xxBlockwiseScaleConfig<
    1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;

#if defined(SOL_TARGET_SM100)

constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;

#if defined(SOL_DYNAMIC_SCHEDULER)
using Sm100TileScheduler = void;
#else
using Sm100TileScheduler = cutlass::gemm::StaticPersistentScheduler;
#endif

using FusedSiluMultiply = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementD, ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::epilogue::thread::SiLu, ElementD, ElementCompute, kRound>,
        cutlass::epilogue::fusion::Sm90SrcFetch<ElementD>>,
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::first, ElementD, ElementCompute, kRound>,
        cutlass::epilogue::fusion::Sm90AccFetch>>;

template <class TileShape, class ClusterShape, class MainloopSchedule,
          bool Fused = false>
struct Sm100GemmBuilder {
  using Scales = ScaleConfig<cutlass::arch::Sm100>;
  using LayoutSFA = decltype(Scales::deduce_layoutSFA());
  using LayoutSFB = decltype(Scales::deduce_layoutSFB());
  using ElementC = cute::conditional_t<Fused, ElementD, void>;
  static constexpr int AlignmentCForKernel = Fused ? AlignmentD : 1;
  using FusionOp = cute::conditional_t<
      Fused, FusedSiluMultiply,
      cutlass::epilogue::fusion::LinearCombination<
          ElementD, ElementCompute, ElementC, ElementCompute, kRound>>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentCForKernel,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      FusionOp>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
      ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue, Sm100TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

#ifndef SOL_CLUSTER_N
#define SOL_CLUSTER_N 4
#endif

#if SOL_CLUSTER_N == 1
#define SOL_CLUSTER_N_CUTE _1
#elif SOL_CLUSTER_N == 2
#define SOL_CLUSTER_N_CUTE _2
#elif SOL_CLUSTER_N == 4
#define SOL_CLUSTER_N_CUTE _4
#elif SOL_CLUSTER_N == 8
#define SOL_CLUSTER_N_CUTE _8
#else
#error "SOL_CLUSTER_N must be 1, 2, 4, or 8"
#endif

using Sm100One = Sm100GemmBuilder<
    Shape<_128, _128, _128>, Shape<_1, SOL_CLUSTER_N_CUTE, _1>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>;
using Sm100Two = Sm100GemmBuilder<
    Shape<_256, _128, _128>, Shape<_2, SOL_CLUSTER_N_CUTE, _1>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise2SmSm100>;
using Sm100OneFused = Sm100GemmBuilder<
    Shape<_128, _128, _128>, Shape<_1, SOL_CLUSTER_N_CUTE, _1>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100, true>;
using Sm100TwoFused = Sm100GemmBuilder<
    Shape<_256, _128, _128>, Shape<_2, SOL_CLUSTER_N_CUTE, _1>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise2SmSm100, true>;

#undef SOL_CLUSTER_N_CUTE

#endif

#if defined(SOL_TARGET_SM120)

template <class TileShape, class MainloopSchedule>
struct Sm120GemmBuilder {
  using ClusterShape = Shape<_1, _1, _1>;
  using Scales = ScaleConfig<cutlass::arch::Sm120>;
  using LayoutSFA = decltype(Scales::deduce_layoutSFA());
  using LayoutSFB = decltype(Scales::deduce_layoutSFB());

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      void, LayoutC, 1,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
      ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Sm120Cooperative = Sm120GemmBuilder<
    Shape<_128, _128, _128>, cutlass::gemm::KernelScheduleSm120Blockwise>;

#endif

template <class Config>
cutlass::Status run_gemm(
    int m,
    const ElementA* a,
    const ElementB* b,
    const float* scale_a,
    const float* scale_b,
    ElementD* d,
    cudaStream_t stream,
    const ElementD* c = nullptr) {
  using Gemm = typename Config::Gemm;
  using Scales = typename Config::Scales;

  const auto problem = cute::make_shape(m, kIntermediate, kHidden, 1);
  const auto stride_a = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideA{},
      cute::make_shape(m, kHidden, 1));
  const auto stride_b = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideB{},
      cute::make_shape(kIntermediate, kHidden, 1));
  const auto stride_c = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideC{},
      cute::make_shape(m, kIntermediate, 1));
  const auto stride_d = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideD{},
      cute::make_shape(m, kIntermediate, 1));
  const auto layout_scale_a = Scales::tile_atom_to_shape_SFA(problem);
  const auto layout_scale_b = Scales::tile_atom_to_shape_SFB(problem);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {a, stride_a, b, stride_b,
       scale_a, layout_scale_a, scale_b, layout_scale_b},
      {{}, c, stride_c, d, stride_d}};

  Gemm gemm;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  if (workspace_size != 0) {
    // CUTLASS workspace is invocation-local. Never retain device memory that
    // participates in numerical computation across evaluator calls.
    workspace = at::empty(
        {static_cast<int64_t>(workspace_size)},
        at::TensorOptions().device(at::kCUDA).dtype(at::kByte));
  }
  return gemm(arguments,
              workspace_size == 0 ? nullptr : workspace.data_ptr(), stream);
}

#if defined(SOL_TARGET_SM120)
__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ up_and_output,
    int64_t count) {
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       static_cast<int64_t>(threadIdx.x);
  if (pair * 2 >= count) {
    return;
  }

  const auto gate2 = reinterpret_cast<const __nv_bfloat162*>(gate)[pair];
  const auto up2 = reinterpret_cast<const __nv_bfloat162*>(up_and_output)[pair];
  const float2 g = __bfloat1622float2(gate2);
  const float2 u = __bfloat1622float2(up2);
  // F.silu preserves the BF16 dtype of gate_output. Reproduce that rounding
  // boundary before the BF16 multiplication rather than fusing in FP32.
  const __nv_bfloat162 activated_bf16 = __float22bfloat162_rn(make_float2(
      g.x / (1.0f + __expf(-g.x)),
      g.y / (1.0f + __expf(-g.y))));
  const float2 activated = __bfloat1622float2(activated_bf16);
  const float2 y = make_float2(
      activated.x * u.x,
      activated.y * u.y);
  reinterpret_cast<__nv_bfloat162*>(up_and_output)[pair] =
      __float22bfloat162_rn(y);
}
#endif

inline void check_cutlass(cutlass::Status status, const char* operation) {
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              operation, " failed: ", cutlassGetStatusString(status));
}

}  // namespace sol179_detail

void launch_fp8_mlp_gate_up(
    const at::Tensor& x,
    const at::Tensor& scale_x,
    const at::Tensor& gate_weight,
    const at::Tensor& scale_gate,
    const at::Tensor& up_weight,
    const at::Tensor& scale_up,
    at::Tensor output,
    cudaStream_t stream) {
  using namespace sol179_detail;
  const int m = static_cast<int>(x.size(0));
  const auto* a = reinterpret_cast<const ElementA*>(x.data_ptr());
  const auto* gate = reinterpret_cast<const ElementB*>(gate_weight.data_ptr());
  const auto* up = reinterpret_cast<const ElementB*>(up_weight.data_ptr());
  const auto* sx = scale_x.data_ptr<float>();
  const auto* sg = scale_gate.data_ptr<float>();
  const auto* su = scale_up.data_ptr<float>();
  auto* result = reinterpret_cast<ElementD*>(output.data_ptr());

  // This intermediate is recomputed from the current invocation's inputs and
  // cannot survive into a later evaluator call.
  at::Tensor gate_scratch = at::empty_like(output);
  auto* gate_result = reinterpret_cast<ElementD*>(gate_scratch.data_ptr());

#if defined(SOL_TARGET_SM100)
  if (((m / 128) & 1) == 0) {
    check_cutlass(run_gemm<Sm100Two>(m, a, gate, sx, sg, gate_result, stream),
                  "SM100 gate GEMM");
    check_cutlass(run_gemm<Sm100TwoFused>(
                      m, a, up, sx, su, result, stream, gate_result),
                  "SM100 fused up GEMM");
  } else {
    check_cutlass(run_gemm<Sm100One>(m, a, gate, sx, sg, gate_result, stream),
                  "SM100 gate GEMM");
    check_cutlass(run_gemm<Sm100OneFused>(
                      m, a, up, sx, su, result, stream, gate_result),
                  "SM100 fused up GEMM");
  }
#elif defined(SOL_TARGET_SM120)
  check_cutlass(run_gemm<Sm120Cooperative>(
                    m, a, gate, sx, sg, gate_result, stream),
                "SM120 gate GEMM");
  check_cutlass(run_gemm<Sm120Cooperative>(
                    m, a, up, sx, su, result, stream),
                "SM120 up GEMM");
#else
  TORCH_CHECK(false, "problem 179 requires SM100a or SM120");
#endif

#if defined(SOL_TARGET_SM120)
  const int64_t count = output.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((count / 2 + threads - 1) / threads);
  silu_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate_result),
      reinterpret_cast<__nv_bfloat16*>(result), count);
#endif

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "problem 179 CUDA launch failed: ", cudaGetErrorString(error));
}
