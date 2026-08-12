// SPDX-FileCopyrightText: 2026 sol-038-flux-rmsnorm contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernel.cuh"

#include <cstdint>
#include <limits>

namespace {

constexpr int kHeads = 48;
constexpr int kHeadSize = 128;
constexpr int kPacksPerHead = 16;
constexpr int kHeadGroupsPerWarp = 8;
constexpr unsigned kFullWarp = 0xffffffffu;
constexpr float kInvHeadSize = 1.0f / static_cast<float>(kHeadSize);

#ifndef SOL38_WARPS_PER_BLOCK
#define SOL38_WARPS_PER_BLOCK 6
#endif

#ifndef SOL38_TARGET_X_CTAS
#define SOL38_TARGET_X_CTAS 256
#endif

#ifndef SOL38_MIN_BLOCKS_PER_SM
#define SOL38_MIN_BLOCKS_PER_SM 3
#endif

#ifndef SOL38_STREAMING_NOALLOC
#define SOL38_STREAMING_NOALLOC 0
#endif

constexpr int kWarpsPerBlock = SOL38_WARPS_PER_BLOCK;
constexpr int kHeadsPerBlock = kWarpsPerBlock * kHeadGroupsPerWarp;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;

static_assert(kWarpsPerBlock > 0, "SOL38_WARPS_PER_BLOCK must be positive");
static_assert(kHeadsPerBlock <= kHeads,
              "SOL38_WARPS_PER_BLOCK assigns more than 48 heads");
static_assert(kHeads % kHeadsPerBlock == 0,
              "SOL38_WARPS_PER_BLOCK must evenly divide 48 heads");

struct alignas(32) Pack256 {
  unsigned long long x;
  unsigned long long y;
  unsigned long long z;
  unsigned long long w;
};

static_assert(sizeof(Pack256) == 32, "Pack256 must be 256 bits");

union U64Float2 {
  unsigned long long u;
  float2 f;
};

__device__ __forceinline__ unsigned long long pack_f2(float2 value) {
  U64Float2 packed;
  packed.f = value;
  return packed.u;
}

__device__ __forceinline__ float2 unpack_f2(unsigned long long value) {
  U64Float2 packed;
  packed.u = value;
  return packed.f;
}

__device__ __forceinline__ float2 add2(float2 lhs, float2 rhs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  unsigned long long result;
  asm volatile("add.f32x2 %0, %1, %2;"
               : "=l"(result)
               : "l"(pack_f2(lhs)), "l"(pack_f2(rhs)));
  return unpack_f2(result);
#else
  return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
#endif
}

__device__ __forceinline__ float2 mul2(float2 lhs, float2 rhs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  unsigned long long result;
  asm volatile("mul.f32x2 %0, %1, %2;"
               : "=l"(result)
               : "l"(pack_f2(lhs)), "l"(pack_f2(rhs)));
  return unpack_f2(result);
#else
  return make_float2(lhs.x * rhs.x, lhs.y * rhs.y);
#endif
}

__device__ __forceinline__ float2 fma2(
    float2 lhs, float2 rhs, float2 addend) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  unsigned long long result;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(pack_f2(lhs)), "l"(pack_f2(rhs)),
                 "l"(pack_f2(addend)));
  return unpack_f2(result);
#else
  return make_float2(
      fmaf(lhs.x, rhs.x, addend.x), fmaf(lhs.y, rhs.y, addend.y));
#endif
}

__device__ __forceinline__ float fast_rsqrt(float value) {
#if defined(__CUDA_ARCH__)
  float result;
  asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
#else
  return rsqrtf(value);
#endif
}

__device__ __forceinline__ Pack256 load256(const Pack256* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  Pack256 result;
  asm volatile("ld.global.v4.b64 {%0, %1, %2, %3}, [%4];"
               : "=l"(result.x), "=l"(result.y), "=l"(result.z),
                 "=l"(result.w)
               : "l"(address)
               : "memory");
  return result;
#else
  return *address;
#endif
}

__device__ __forceinline__ Pack256 load256_streaming(
    const Pack256* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  Pack256 result;
#if SOL38_STREAMING_NOALLOC
  asm volatile("ld.global.L1::no_allocate.v4.b64 {%0, %1, %2, %3}, [%4];"
               : "=l"(result.x), "=l"(result.y), "=l"(result.z),
                 "=l"(result.w)
               : "l"(address)
               : "memory");
#else
  asm volatile("ld.global.v4.b64 {%0, %1, %2, %3}, [%4];"
               : "=l"(result.x), "=l"(result.y), "=l"(result.z),
                 "=l"(result.w)
               : "l"(address)
               : "memory");
#endif
  return result;
#else
  return *address;
#endif
}

__device__ __forceinline__ void store256_streaming(
    Pack256* address, Pack256 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#if SOL38_STREAMING_NOALLOC
  asm volatile("st.global.L1::no_allocate.v4.b64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(address), "l"(value.x), "l"(value.y), "l"(value.z),
                 "l"(value.w)
               : "memory");
#else
  asm volatile("st.global.v4.b64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(address), "l"(value.x), "l"(value.y), "l"(value.z),
                 "l"(value.w)
               : "memory");
#endif
#else
  *address = value;
#endif
}

__device__ __forceinline__ void accumulate_square(
    Pack256 value, float2& acc0, float2& acc1, float2& acc2, float2& acc3) {
  const float2 value0 = unpack_f2(value.x);
  const float2 value1 = unpack_f2(value.y);
  const float2 value2 = unpack_f2(value.z);
  const float2 value3 = unpack_f2(value.w);
  acc0 = fma2(value0, value0, acc0);
  acc1 = fma2(value1, value1, acc1);
  acc2 = fma2(value2, value2, acc2);
  acc3 = fma2(value3, value3, acc3);
}

__device__ __forceinline__ Pack256 scale_pack(
    Pack256 input, Pack256 weight, float scale) {
  const float2 scale2 = make_float2(scale, scale);
  Pack256 output;
  output.x = pack_f2(
      mul2(mul2(unpack_f2(input.x), scale2), unpack_f2(weight.x)));
  output.y = pack_f2(
      mul2(mul2(unpack_f2(input.y), scale2), unpack_f2(weight.y)));
  output.z = pack_f2(
      mul2(mul2(unpack_f2(input.z), scale2), unpack_f2(weight.z)));
  output.w = pack_f2(
      mul2(mul2(unpack_f2(input.w), scale2), unpack_f2(weight.w)));
  return output;
}

__global__ __launch_bounds__(kThreadsPerBlock, SOL38_MIN_BLOCKS_PER_SM)
void rmsnorm_qk_stream4_vec32_kernel(
    const Pack256* __restrict__ query,
    const Pack256* __restrict__ key,
    const Pack256* __restrict__ weight_q,
    const Pack256* __restrict__ weight_k,
    float eps,
    int rows,
    int rows_per_cta,
    Pack256* __restrict__ query_norm,
    Pack256* __restrict__ key_norm) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int subgroup = lane >> 2;
  const int sublane = lane & 3;
  const int local_head = warp * kHeadGroupsPerWarp + subgroup;
  const int head = static_cast<int>(blockIdx.z) * kHeadsPerBlock + local_head;
  const bool is_key = blockIdx.y != 0;

  const Pack256* __restrict__ input = is_key ? key : query;
  const Pack256* __restrict__ weight = is_key ? weight_k : weight_q;
  Pack256* __restrict__ output = is_key ? key_norm : query_norm;

  const Pack256* weight_ptr = weight + head * kPacksPerHead;
  const Pack256 weight0 = load256(weight_ptr + sublane);
  const Pack256 weight1 = load256(weight_ptr + sublane + 4);
  const Pack256 weight2 = load256(weight_ptr + sublane + 8);
  const Pack256 weight3 = load256(weight_ptr + sublane + 12);

  const int row_begin = static_cast<int>(blockIdx.x) * rows_per_cta;
  const int row_end = min(row_begin + rows_per_cta, rows);

  for (int row = row_begin; row < row_end; ++row) {
    const int64_t base =
        (static_cast<int64_t>(row) * kHeads + head) * kPacksPerHead;
    const Pack256* input_ptr = input + base;

    const Pack256 input0 = load256_streaming(input_ptr + sublane);
    const Pack256 input1 = load256_streaming(input_ptr + sublane + 4);
    const Pack256 input2 = load256_streaming(input_ptr + sublane + 8);
    const Pack256 input3 = load256_streaming(input_ptr + sublane + 12);

    float2 acc0 = make_float2(0.0f, 0.0f);
    float2 acc1 = make_float2(0.0f, 0.0f);
    float2 acc2 = make_float2(0.0f, 0.0f);
    float2 acc3 = make_float2(0.0f, 0.0f);
    accumulate_square(input0, acc0, acc1, acc2, acc3);
    accumulate_square(input1, acc0, acc1, acc2, acc3);
    accumulate_square(input2, acc0, acc1, acc2, acc3);
    accumulate_square(input3, acc0, acc1, acc2, acc3);

    const float2 sum01 = add2(acc0, acc1);
    const float2 sum23 = add2(acc2, acc3);
    const float2 packed_sum = add2(sum01, sum23);
    float sum = packed_sum.x + packed_sum.y;
    sum += __shfl_xor_sync(kFullWarp, sum, 2, 4);
    sum += __shfl_xor_sync(kFullWarp, sum, 1, 4);

    float scale = 0.0f;
    if (sublane == 0) {
      scale = fast_rsqrt(fmaf(sum, kInvHeadSize, eps));
    }
    scale = __shfl_sync(kFullWarp, scale, 0, 4);

    Pack256* output_ptr = output + base;
    store256_streaming(
        output_ptr + sublane, scale_pack(input0, weight0, scale));
    store256_streaming(
        output_ptr + sublane + 4, scale_pack(input1, weight1, scale));
    store256_streaming(
        output_ptr + sublane + 8, scale_pack(input2, weight2, scale));
    store256_streaming(
        output_ptr + sublane + 12, scale_pack(input3, weight3, scale));
  }
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
  const int64_t rows64 = query.numel() / (kHeads * kHeadSize);
  TORCH_CHECK(rows64 > 0, "empty input is unsupported");
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(), "too many rows");
  const int rows = static_cast<int>(rows64);

  int rows_per_cta = rows / SOL38_TARGET_X_CTAS;
  if (rows_per_cta < 1) {
    rows_per_cta = 1;
  }
  const int grid_x = (rows + rows_per_cta - 1) / rows_per_cta;

  const dim3 block(kThreadsPerBlock);
  const dim3 grid(grid_x, 2, kHeads / kHeadsPerBlock);
  rmsnorm_qk_stream4_vec32_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const Pack256*>(query.data_ptr<float>()),
      reinterpret_cast<const Pack256*>(key.data_ptr<float>()),
      reinterpret_cast<const Pack256*>(weight_q.data_ptr<float>()),
      reinterpret_cast<const Pack256*>(weight_k.data_ptr<float>()),
      eps,
      rows,
      rows_per_cta,
      reinterpret_cast<Pack256*>(query_norm.data_ptr<float>()),
      reinterpret_cast<Pack256*>(key_norm.data_ptr<float>()));

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "rmsnorm_qk_stream4_vec32_kernel launch failed: ",
              cudaGetErrorString(error));
}
