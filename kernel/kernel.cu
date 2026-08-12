// SPDX-FileCopyrightText: 2026 sol-038-flux-rmsnorm contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernel.cuh"

#include <cstdint>

namespace {

constexpr int kHeads = 48;
constexpr int kHeadSize = 128;
constexpr int kVecsPerHead = kHeadSize / 4;
constexpr float kInvHeadSize = 1.0f / static_cast<float>(kHeadSize);
constexpr unsigned kFullMask = 0xffffffffu;

union F32x2Bits {
  float2 f;
  unsigned long long u;
};

__device__ __forceinline__ float2 mul2(float2 lhs, float2 rhs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  F32x2Bits a;
  F32x2Bits b;
  F32x2Bits result;
  a.f = lhs;
  b.f = rhs;
  asm volatile("mul.f32x2 %0, %1, %2;"
               : "=l"(result.u)
               : "l"(a.u), "l"(b.u));
  return result.f;
#else
  return make_float2(lhs.x * rhs.x, lhs.y * rhs.y);
#endif
}

__device__ __forceinline__ float2 fma2(
    float2 lhs, float2 rhs, float2 addend) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  F32x2Bits a;
  F32x2Bits b;
  F32x2Bits c;
  F32x2Bits result;
  a.f = lhs;
  b.f = rhs;
  c.f = addend;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;"
               : "=l"(result.u)
               : "l"(a.u), "l"(b.u), "l"(c.u));
  return result.f;
#else
  return make_float2(
      fmaf(lhs.x, rhs.x, addend.x),
      fmaf(lhs.y, rhs.y, addend.y));
#endif
}

__device__ __forceinline__ float fast_rsqrt(float value) {
  float result;
  asm volatile("rsqrt.approx.f32 %0, %1;"
               : "=f"(result)
               : "f"(value));
  return result;
}

// The 128-token case is launch-sensitive on B200. Each half-warp handles one
// token/head row, and the two grid-x planes keep Q and K memory streams apart.
constexpr int kSmallWarps = 8;
constexpr int kSmallThreads = kSmallWarps * 32;
constexpr int kSmallTokenStreams = 64;

__global__ __launch_bounds__(kSmallThreads, 4)
void rmsnorm_qk_token_g16_128_kernel(
    const float4* __restrict__ query,
    const float4* __restrict__ key,
    const float4* __restrict__ weight_q,
    const float4* __restrict__ weight_k,
    float eps,
    float4* __restrict__ query_norm,
    float4* __restrict__ key_norm) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int token_group = lane >> 4;
  const int sublane = lane & 15;
  const bool is_key = (static_cast<int>(blockIdx.x) & 1) != 0;
  const int head_tile = static_cast<int>(blockIdx.x) >> 1;
  const int head = head_tile * kSmallWarps + warp;
  const int token = static_cast<int>(blockIdx.y) * 2 + token_group;

  const float4* __restrict__ input = is_key ? key : query;
  const float4* __restrict__ weight = is_key ? weight_k : weight_q;
  float4* __restrict__ output = is_key ? key_norm : query_norm;

  const int weight_base = head * kVecsPerHead + sublane;
  const float4 weight0 = weight[weight_base];
  const float4 weight1 = weight[weight_base + 16];
  const int input_base =
      (token * kHeads + head) * kVecsPerHead + sublane;
  const float4 input0 = input[input_base];
  const float4 input1 = input[input_base + 16];
  const float2 input01 = make_float2(input0.x, input0.y);
  const float2 input23 = make_float2(input0.z, input0.w);
  const float2 input45 = make_float2(input1.x, input1.y);
  const float2 input67 = make_float2(input1.z, input1.w);

  float2 packed_sum = mul2(input01, input01);
  packed_sum = fma2(input23, input23, packed_sum);
  packed_sum = fma2(input45, input45, packed_sum);
  packed_sum = fma2(input67, input67, packed_sum);
  float sum = packed_sum.x + packed_sum.y;
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(kFullMask, sum, offset, 16);
  }

  const float scale = fast_rsqrt(fmaf(sum, kInvHeadSize, eps));
  const float2 scale2 = make_float2(scale, scale);
  const float2 output01 =
      mul2(mul2(input01, scale2), make_float2(weight0.x, weight0.y));
  const float2 output23 =
      mul2(mul2(input23, scale2), make_float2(weight0.z, weight0.w));
  const float2 output45 =
      mul2(mul2(input45, scale2), make_float2(weight1.x, weight1.y));
  const float2 output67 =
      mul2(mul2(input67, scale2), make_float2(weight1.z, weight1.w));
  output[input_base] =
      make_float4(output01.x, output01.y, output23.x, output23.y);
  output[input_base + 16] =
      make_float4(output45.x, output45.y, output67.x, output67.y);
}

// One grid-y plane handles Q and the other handles K. This preserves one
// launch while halving each CTA's live input, reduction, and weight state.
// Adjacent rows are assigned inside a 48-head token, so no row modulo is
// executed in the hot path and memory remains contiguous across CTAs.
template <int RowsPerWarp, int WarpsPerBlock, int MinBlocksPerSm>
__global__ __launch_bounds__(WarpsPerBlock * 32, MinBlocksPerSm)
void rmsnorm_qk_grid_kernel(
    const float4* __restrict__ query,
    const float4* __restrict__ key,
    const float4* __restrict__ weight_q,
    const float4* __restrict__ weight_k,
    float eps,
    float4* __restrict__ query_norm,
    float4* __restrict__ key_norm) {
  constexpr int kRowsPerBlock = RowsPerWarp * WarpsPerBlock;
  constexpr int kBlocksPerToken = kHeads / kRowsPerBlock;
  static_assert(kHeads % kRowsPerBlock == 0,
                "CTA rows must evenly tile the 48 benchmark heads");

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int block = static_cast<int>(blockIdx.x);
  const int token = block / kBlocksPerToken;
  const int head_group = block - token * kBlocksPerToken;
  const int head_base =
      head_group * kRowsPerBlock + warp * RowsPerWarp;
  const int row_base = token * kHeads + head_base;
  const bool is_key = blockIdx.y != 0;

  const float4* __restrict__ input = is_key ? key : query;
  const float4* __restrict__ weight = is_key ? weight_k : weight_q;
  float4* __restrict__ output = is_key ? key_norm : query_norm;
  float4 values[RowsPerWarp];
  float sums[RowsPerWarp];

#pragma unroll
  for (int row_offset = 0; row_offset < RowsPerWarp; ++row_offset) {
    const int vector_offset =
        (row_base + row_offset) * kVecsPerHead + lane;
    values[row_offset] = input[vector_offset];
    const float4 value = values[row_offset];
    const float2 value01 = make_float2(value.x, value.y);
    const float2 value23 = make_float2(value.z, value.w);
    float2 packed_sum = mul2(value01, value01);
    packed_sum = fma2(value23, value23, packed_sum);
    sums[row_offset] = packed_sum.x + packed_sum.y;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int row_offset = 0; row_offset < RowsPerWarp; ++row_offset) {
      sums[row_offset] +=
          __shfl_xor_sync(kFullMask, sums[row_offset], offset);
    }
  }

#pragma unroll
  for (int row_offset = 0; row_offset < RowsPerWarp; ++row_offset) {
    const int head = head_base + row_offset;
    const int vector_offset =
        (row_base + row_offset) * kVecsPerHead + lane;
    const float4 value = values[row_offset];
    const float4 weights = weight[head * kVecsPerHead + lane];
    const float scale =
        fast_rsqrt(fmaf(sums[row_offset], kInvHeadSize, eps));
    const float2 scale2 = make_float2(scale, scale);
    const float2 output01 = mul2(
        mul2(make_float2(value.x, value.y), scale2),
        make_float2(weights.x, weights.y));
    const float2 output23 = mul2(
        mul2(make_float2(value.z, value.w), scale2),
        make_float2(weights.z, weights.w));
    output[vector_offset] =
        make_float4(output01.x, output01.y, output23.x, output23.y);
  }
}

template <int RowsPerWarp, int WarpsPerBlock, int MinBlocksPerSm>
void launch_grid_variant(
    const float4* query,
    const float4* key,
    const float4* weight_q,
    const float4* weight_k,
    float eps,
    int tokens,
    float4* query_norm,
    float4* key_norm,
    cudaStream_t stream) {
  constexpr int kRowsPerBlock = RowsPerWarp * WarpsPerBlock;
  constexpr int kBlocksPerToken = kHeads / kRowsPerBlock;
  const dim3 grid(tokens * kBlocksPerToken, 2);
  rmsnorm_qk_grid_kernel<RowsPerWarp, WarpsPerBlock, MinBlocksPerSm>
      <<<grid, WarpsPerBlock * 32, 0, stream>>>(
          query, key, weight_q, weight_k, eps, query_norm, key_norm);
}

}  // namespace

void launch_flux_rmsnorm_qk(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_k,
    float eps,
    torch::Tensor query_norm,
    torch::Tensor key_norm,
    cudaStream_t stream) {
  const int tokens = static_cast<int>(
      query.numel() / (kHeads * kHeadSize));
  const auto* q = reinterpret_cast<const float4*>(query.data_ptr<float>());
  const auto* k = reinterpret_cast<const float4*>(key.data_ptr<float>());
  const auto* wq =
      reinterpret_cast<const float4*>(weight_q.data_ptr<float>());
  const auto* wk =
      reinterpret_cast<const float4*>(weight_k.data_ptr<float>());
  auto* oq = reinterpret_cast<float4*>(query_norm.data_ptr<float>());
  auto* ok = reinterpret_cast<float4*>(key_norm.data_ptr<float>());

  if (tokens == 128) {
    constexpr int kHeadTiles = kHeads / kSmallWarps;
    const dim3 grid(2 * kHeadTiles, kSmallTokenStreams);
    rmsnorm_qk_token_g16_128_kernel<<<grid, kSmallThreads, 0, stream>>>(
        q, k, wq, wk, eps, oq, ok);
  } else if (tokens <= 2048) {
    launch_grid_variant<2, 4, 4>(
        q, k, wq, wk, eps, tokens, oq, ok, stream);
  } else {
    launch_grid_variant<2, 8, 3>(
        q, k, wq, wk, eps, tokens, oq, ok, stream);
  }

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "flux RMSNorm kernel launch failed: ",
              cudaGetErrorString(error));
}
