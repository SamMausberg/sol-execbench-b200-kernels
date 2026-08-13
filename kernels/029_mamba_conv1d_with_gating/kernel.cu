// SPDX-FileCopyrightText: 2026 sol-038-flux-rmsnorm contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernel.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

#if defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ == 1000)
#define SOL29_TARGET_SM100 1
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#endif

namespace sol29_detail {

constexpr int kIntermediate = 16384;
constexpr int kProjected = 32768;
constexpr int kHidden = 8192;
constexpr int kKernel = 4;

#if defined(SOL29_TARGET_SM100)

using namespace cute;

using GemmElement = cutlass::bfloat16_t;
using GemmAccumulator = float;
using GemmCompute = float;
using GemmLayoutA = cutlass::layout::RowMajor;
using GemmLayoutB = cutlass::layout::ColumnMajor;
using GemmLayoutC = cutlass::layout::RowMajor;
using GemmLayoutD = cutlass::layout::RowMajor;
constexpr int kGemmAlignment = 8;  // 16-byte vectors of BF16.
constexpr auto kGemmRound = cutlass::FloatRoundStyle::round_to_nearest;

template <class TileShape, class ClusterShape, class MainloopSchedule,
          class EpilogueSchedule>
struct Sm100ProjectionBuilder {
  // The projection bias is a [N] row vector.  Loading it through an EVT is
  // both legal for arbitrary M residues and avoids presenting a zero-stride
  // tensor to TMA (zero TMA strides are not generally valid descriptors).
  using BiasBroadcast = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0, TileShape, GemmElement, GemmCompute>;
  using BiasAdd = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::plus, GemmElement, GemmCompute, kGemmRound>,
      cutlass::epilogue::fusion::Sm90AccFetch,
      BiasBroadcast>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      GemmAccumulator, GemmCompute,
      void, GemmLayoutC, 1,
      GemmElement, GemmLayoutD, kGemmAlignment,
      EpilogueSchedule, BiasAdd>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      GemmElement, GemmLayoutA, kGemmAlignment,
      GemmElement, GemmLayoutB, kGemmAlignment,
      GemmAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// A single-CTA UMMA tile avoids wasting a partner CTA for M=128.  All larger
// benchmark M values use 2SM UMMA, which shares each 512 MiB weight matrix tile
// across the paired CTAs and is the throughput-oriented B200 path.
using Sm100Projection1Sm = Sm100ProjectionBuilder<
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::TmaWarpSpecialized1Sm>;
using Sm100Projection2Sm = Sm100ProjectionBuilder<
    Shape<_256, _128, _64>, Shape<_2, _2, _1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
    cutlass::epilogue::TmaWarpSpecialized2Sm>;

template <class Config>
cutlass::Status run_sm100_projection(
    int m,
    const GemmElement* hidden,
    const GemmElement* weight,
    const GemmElement* bias,
    GemmElement* projected,
    const at::Device& device,
    cudaStream_t stream) {
  using Gemm = typename Config::Gemm;
  const auto problem = cute::make_shape(m, kProjected, kHidden, 1);
  const auto stride_a = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideA{},
      cute::make_shape(m, kHidden, 1));
  // Column-major [K,N] aliases the supplied row-major [N,K] weight exactly.
  const auto stride_b = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideB{},
      cute::make_shape(kProjected, kHidden, 1));
  const auto stride_c = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideC{},
      cute::make_shape(m, kProjected, 1));
  const auto stride_d = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideD{},
      cute::make_shape(m, kProjected, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {hidden, stride_a, weight, stride_b},
      {{{}, {bias}, {}}, nullptr, stride_c, projected, stride_d}};

  Gemm gemm;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  if (workspace_size != 0) {
    // CUTLASS workspace is invocation-local. Never retain device memory that
    // participates in numerical computation across evaluator calls.
    workspace = at::empty(
        {static_cast<int64_t>(workspace_size)},
        at::TensorOptions().device(device).dtype(at::kByte));
  }
  const cutlass::Status support = gemm.can_implement(arguments);
  if (support != cutlass::Status::kSuccess) {
    return support;
  }
  return gemm(arguments,
              workspace_size == 0 ? nullptr : workspace.data_ptr(), stream);
}

#endif  // SOL29_TARGET_SM100

// A CTA transposes a [sequence, channel] projection tile through shared memory
// while applying the full depthwise tail.  A bank-skewed channel pitch is
// essential: it makes a warp that reads one channel across 32 sequence
// positions conflict-free instead of incurring a 16-way conflict.
constexpr int kThreads = 256;

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float value) {
  return __float2bfloat16_rn(value);
}

template <int TileSequence, int TileChannels, int SharedPitch>
__global__ __launch_bounds__(kThreads, 4)
void mamba_conv1d_tail_kernel(
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ attention_mask,
    const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ conv_bias,
    __nv_bfloat16* __restrict__ output,
    int sequence) {
  __shared__ __nv_bfloat16
      shared_projection[(TileSequence + kKernel - 1) * SharedPitch];
  __shared__ __nv_bfloat16 shared_weight[TileChannels * kKernel];
  __shared__ __nv_bfloat16 shared_bias[TileChannels];

  const int tid = static_cast<int>(threadIdx.x);
  const int batch = static_cast<int>(blockIdx.z);
  const int channel_base = static_cast<int>(blockIdx.x) * TileChannels;
  const int sequence_base = static_cast<int>(blockIdx.y) * TileSequence;

  // All benchmark channel counts are exact tiles.  Keeping the predicate here
  // costs nothing after constant folding and makes the launch helper robust.
  if (tid < TileChannels) {
    const int channel = channel_base + tid;
    if (channel < kIntermediate) {
      shared_bias[tid] = conv_bias[channel];
    }
  }
  if (tid < TileChannels * kKernel) {
    const int channel_local = tid >> 2;
    const int tap = tid & 3;
    const int channel = channel_base + channel_local;
    if (channel < kIntermediate) {
      shared_weight[channel_local * kKernel + tap] =
          conv_weight[channel * kKernel + tap];
    }
  }

  // Load the three-position halo plus the sequence tile in GEMM-friendly
  // order: every
  // participating warp reads one contiguous 64-byte channel row.  The mask
  // multiply is rounded to BF16 here, exactly where eager PyTorch materializes
  // its pre-convolution masked tensor.
  constexpr int kSharedValues =
      (TileSequence + kKernel - 1) * TileChannels;
  for (int index = tid; index < kSharedValues; index += kThreads) {
    const int row = index / TileChannels;
    const int channel_local = index & (TileChannels - 1);
    const int position = sequence_base + row - (kKernel - 1);
    const int channel = channel_base + channel_local;
    __nv_bfloat16 value = float_to_bf16(0.0f);
    if (position >= 0 && position < sequence && channel < kIntermediate) {
      const int64_t token =
          static_cast<int64_t>(batch) * sequence + position;
      const float projected_value =
          bf16_to_float(projected[token * kProjected + channel]);
      const float mask_value = bf16_to_float(attention_mask[token]);
      value = float_to_bf16(projected_value * mask_value);
    }
    shared_projection[row * SharedPitch + channel_local] = value;
  }
  __syncthreads();

  // Consecutive lanes own consecutive sequence positions.  Each warp thus
  // emits full coalesced output segments, while the padded shared-memory tile
  // supplies all four causal taps without rereading projected from HBM.
  constexpr int kOutputsPerBlock = TileSequence * TileChannels;
#pragma unroll
  for (int index = tid; index < kOutputsPerBlock; index += kThreads) {
    const int channel_local = index / TileSequence;
    const int position_local = index - channel_local * TileSequence;
    const int channel = channel_base + channel_local;
    const int position = sequence_base + position_local;
    if (channel < kIntermediate && position < sequence) {
      const int64_t mask_index =
          static_cast<int64_t>(batch) * sequence + position;
      const __nv_bfloat16 mask_bf16 = attention_mask[mask_index];
      __nv_bfloat16 result = float_to_bf16(0.0f);
      // The post-mask makes the stencil and activation dead for a masked
      // output.  For valid positions, preserve the reference's BF16 rounding
      // boundaries: conv1d -> sigmoid -> multiply.
      if (__hge(mask_bf16, float_to_bf16(0.5f))) {
        const __nv_bfloat16* x =
            shared_projection + position_local * SharedPitch + channel_local;
        const __nv_bfloat16* w =
            shared_weight + channel_local * kKernel;
        float accumulator = bf16_to_float(x[0 * SharedPitch]) *
                            bf16_to_float(w[0]);
        accumulator = fmaf(bf16_to_float(x[1 * SharedPitch]),
                           bf16_to_float(w[1]), accumulator);
        accumulator = fmaf(bf16_to_float(x[2 * SharedPitch]),
                           bf16_to_float(w[2]), accumulator);
        accumulator = fmaf(bf16_to_float(x[3 * SharedPitch]),
                           bf16_to_float(w[3]), accumulator);
        accumulator += bf16_to_float(shared_bias[channel_local]);
        const __nv_bfloat16 conv_bf16 = float_to_bf16(accumulator);
        const float conv = bf16_to_float(conv_bf16);
        const __nv_bfloat16 sigmoid_bf16 =
            float_to_bf16(1.0f / (1.0f + __expf(-conv)));
        result = float_to_bf16(conv * bf16_to_float(sigmoid_bf16));
      }
      const int64_t output_index =
          (static_cast<int64_t>(batch) * kIntermediate + channel) * sequence +
          position;
      output[output_index] = result;
    }
  }
}

}  // namespace sol29_detail

bool has_sm100_cutlass_projection() {
#if defined(SOL29_TARGET_SM100)
  return true;
#else
  return false;
#endif
}

void launch_sm100_cutlass_projection(
    const at::Tensor& hidden_states,
    const at::Tensor& in_proj_weight,
    const at::Tensor& in_proj_bias,
    at::Tensor& projected,
    cudaStream_t stream) {
#if defined(SOL29_TARGET_SM100)
  using namespace sol29_detail;
  const int m = static_cast<int>(hidden_states.size(0) * hidden_states.size(1));
  const auto* hidden =
      reinterpret_cast<const GemmElement*>(hidden_states.data_ptr());
  const auto* weight =
      reinterpret_cast<const GemmElement*>(in_proj_weight.data_ptr());
  const auto* bias =
      reinterpret_cast<const GemmElement*>(in_proj_bias.data_ptr());
  auto* output = reinterpret_cast<GemmElement*>(projected.data_ptr());

  cutlass::Status status;
  if (m == 128) {
    status = run_sm100_projection<Sm100Projection1Sm>(
        m, hidden, weight, bias, output, projected.device(), stream);
  } else {
    status = run_sm100_projection<Sm100Projection2Sm>(
        m, hidden, weight, bias, output, projected.device(), stream);
  }
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "problem 29 SM100 CUTLASS projection failed: ",
              cutlassGetStatusString(status));
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "problem 29 SM100 CUTLASS launch failed: ",
              cudaGetErrorString(error));
#else
  TORCH_CHECK(false,
              "SM100 CUTLASS projection was requested from a non-SM100 build");
#endif
}

void launch_mamba_conv1d_tail(
    const at::Tensor& projected,
    const at::Tensor& attention_mask,
    const at::Tensor& conv1d_weight,
    const at::Tensor& conv1d_bias,
    at::Tensor& output_hidden_states,
    cudaStream_t stream) {
  using namespace sol29_detail;
  const int batch = static_cast<int>(projected.size(0));
  const int sequence = static_cast<int>(projected.size(1));
  const dim3 block(kThreads);
  const auto* projected_ptr =
      reinterpret_cast<const __nv_bfloat16*>(projected.data_ptr());
  const auto* mask_ptr =
      reinterpret_cast<const __nv_bfloat16*>(attention_mask.data_ptr());
  const auto* weight_ptr =
      reinterpret_cast<const __nv_bfloat16*>(conv1d_weight.data_ptr());
  const auto* bias_ptr =
      reinterpret_cast<const __nv_bfloat16*>(conv1d_bias.data_ptr());
  auto* output_ptr =
      reinterpret_cast<__nv_bfloat16*>(output_hidden_states.data_ptr());

  // Thirty-two channels provide at least 512 CTAs even for B=1,L=128,
  // enough to fill the 148-SM B200.  Pitch 33 removes the otherwise severe
  // bank conflict when a warp reads one channel across 32 positions.
  const dim3 grid(
      kIntermediate / 32,
      (sequence + 128 - 1) / 128,
      batch);
  mamba_conv1d_tail_kernel<128, 32, 33><<<grid, block, 0, stream>>>(
      projected_ptr,
      mask_ptr,
      weight_ptr,
      bias_ptr,
      output_ptr,
      sequence);
}
