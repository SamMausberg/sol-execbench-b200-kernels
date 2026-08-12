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
constexpr int kFlatVecsPerHead = kHeadSize / 4;
constexpr int kFlatSubgroupWidth = 16;
constexpr int kFlatWarpsPerBlock = 32;
constexpr int kFlatThreadsPerBlock = kFlatWarpsPerBlock * 32;
constexpr int kTokenG16WarpsPerBlock = 8;
constexpr int kTokenG16ThreadsPerBlock = kTokenG16WarpsPerBlock * 32;
constexpr int kTokenG16Streams = 64;
constexpr int kG8Width = 8;
constexpr int kG8HeadPairsPerWarp = 2;
constexpr int kG8HeadsPerBlock = 24;
constexpr int kG8WarpsPerBlock =
    kG8HeadsPerBlock / kG8HeadPairsPerWarp;
constexpr int kG8ThreadsPerBlock = kG8WarpsPerBlock * 32;
constexpr int kStream4MaxRows = 256;
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

// These coherent cache operators are used only by the measured 256-row G8
// path. They change replacement priority, not data identity or semantics.
__device__ __forceinline__ Pack256 load256_g8_input(
    const Pack256* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  Pack256 result;
  asm volatile(
      "ld.global.L1::no_allocate.L2::evict_first.v4.b64 "
      "{%0, %1, %2, %3}, [%4];"
      : "=l"(result.x), "=l"(result.y), "=l"(result.z), "=l"(result.w)
      : "l"(address)
      : "memory");
  return result;
#else
  return *address;
#endif
}

__device__ __forceinline__ Pack256 load256_g8_weight(
    const Pack256* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  Pack256 result;
  asm volatile(
      "ld.global.L1::evict_last.L2::evict_last.v4.b64 "
      "{%0, %1, %2, %3}, [%4];"
      : "=l"(result.x), "=l"(result.y), "=l"(result.z), "=l"(result.w)
      : "l"(address)
      : "memory");
  return result;
#else
  return *address;
#endif
}

__device__ __forceinline__ void store256_g8_output(
    Pack256* address, Pack256 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile(
      "st.global.L1::no_allocate.L2::evict_first.v4.b64 "
      "[%0], {%1, %2, %3, %4};"
      :
      : "l"(address), "l"(value.x), "l"(value.y), "l"(value.z),
        "l"(value.w)
      : "memory");
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

__device__ __forceinline__ void accumulate_square_2way(
    Pack256 value, float2& even_acc, float2& odd_acc) {
  const float2 value0 = unpack_f2(value.x);
  const float2 value1 = unpack_f2(value.y);
  const float2 value2 = unpack_f2(value.z);
  const float2 value3 = unpack_f2(value.w);
  even_acc = fma2(value0, value0, even_acc);
  odd_acc = fma2(value1, value1, odd_acc);
  even_acc = fma2(value2, value2, even_acc);
  odd_acc = fma2(value3, value3, odd_acc);
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

__global__ __launch_bounds__(kTokenG16ThreadsPerBlock, 4)
void rmsnorm_qk_token_g16_128_kernel(
    const float4* __restrict__ query,
    const float4* __restrict__ key,
    const float4* __restrict__ weight_q,
    const float4* __restrict__ weight_k,
    float eps,
    float4* __restrict__ query_norm,
    float4* __restrict__ key_norm) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int token_group = lane >> 4;
  const int sublane = lane & 15;
  const bool is_key = (static_cast<int>(blockIdx.x) & 1) != 0;
  const int head_tile = static_cast<int>(blockIdx.x) >> 1;
  const int head = head_tile * kTokenG16WarpsPerBlock + warp;
  const int token = static_cast<int>(blockIdx.y) * 2 + token_group;

  const float4* __restrict__ input = is_key ? key : query;
  const float4* __restrict__ weight = is_key ? weight_k : weight_q;
  float4* __restrict__ output = is_key ? key_norm : query_norm;

  const int weight_base = head * kFlatVecsPerHead + sublane;
  const float4 weight0 = weight[weight_base];
  const float4 weight1 = weight[weight_base + 16];

  const int64_t base =
      (static_cast<int64_t>(token) * kHeads + head) * kFlatVecsPerHead +
      sublane;
  const float4 input0 = input[base];
  const float4 input1 = input[base + 16];
  const float2 input01 = make_float2(input0.x, input0.y);
  const float2 input23 = make_float2(input0.z, input0.w);
  const float2 input45 = make_float2(input1.x, input1.y);
  const float2 input67 = make_float2(input1.z, input1.w);

  float2 sum0 = fma2(input01, input01, make_float2(0.0f, 0.0f));
  sum0 = fma2(input45, input45, sum0);
  float2 sum1 = fma2(input23, input23, make_float2(0.0f, 0.0f));
  sum1 = fma2(input67, input67, sum1);
  const float2 packed_sum = add2(sum0, sum1);
  float sum = packed_sum.x + packed_sum.y;
  sum += __shfl_xor_sync(kFullWarp, sum, 8, 16);
  sum += __shfl_xor_sync(kFullWarp, sum, 4, 16);
  sum += __shfl_xor_sync(kFullWarp, sum, 2, 16);
  sum += __shfl_xor_sync(kFullWarp, sum, 1, 16);

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

  output[base] =
      make_float4(output01.x, output01.y, output23.x, output23.y);
  output[base + 16] =
      make_float4(output45.x, output45.y, output67.x, output67.y);
}

// B200-measured 256-row path from the supplied optimization bundle. Each warp
// computes two complete Q/K head pairs using four independent 8-lane groups.
__global__ __launch_bounds__(kG8ThreadsPerBlock, 3)
void rmsnorm_qk_g8_256_kernel(
    const Pack256* __restrict__ query,
    const Pack256* __restrict__ key,
    const Pack256* __restrict__ weight_q,
    const Pack256* __restrict__ weight_k,
    float eps,
    Pack256* __restrict__ query_norm,
    Pack256* __restrict__ key_norm) {
  const int token = static_cast<int>(blockIdx.x);
  const int head_tile = static_cast<int>(blockIdx.y);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int subgroup = lane >> 3;
  const int sublane = lane & (kG8Width - 1);
  const int pair_in_warp = subgroup >> 1;
  const bool is_key = (subgroup & 1) != 0;
  const int head =
      head_tile * kG8HeadsPerBlock + warp * kG8HeadPairsPerWarp +
      pair_in_warp;

  const Pack256* __restrict__ input = is_key ? key : query;
  const Pack256* __restrict__ weight = is_key ? weight_k : weight_q;
  Pack256* __restrict__ output = is_key ? key_norm : query_norm;

  const int weight_base = head * kPacksPerHead + sublane;
  const int base =
      token * (kHeads * kPacksPerHead) + weight_base;
  const Pack256 input0 = load256_g8_input(input + base);
  const Pack256 input1 = load256_g8_input(input + base + kG8Width);

  float2 even_acc = make_float2(0.0f, 0.0f);
  float2 odd_acc = make_float2(0.0f, 0.0f);
  accumulate_square_2way(input0, even_acc, odd_acc);
  accumulate_square_2way(input1, even_acc, odd_acc);
  const float2 packed_sum = add2(even_acc, odd_acc);
  float sum = packed_sum.x + packed_sum.y;
  sum += __shfl_xor_sync(kFullWarp, sum, 4, kG8Width);
  sum += __shfl_xor_sync(kFullWarp, sum, 2, kG8Width);
  sum += __shfl_xor_sync(kFullWarp, sum, 1, kG8Width);
  const float scale = fast_rsqrt(fmaf(sum, kInvHeadSize, eps));

  Pack256 weight_pack = load256_g8_weight(weight + weight_base);
  store256_g8_output(
      output + base, scale_pack(input0, weight_pack, scale));
  weight_pack =
      load256_g8_weight(weight + weight_base + kG8Width);
  store256_g8_output(
      output + base + kG8Width,
      scale_pack(input1, weight_pack, scale));
}

__global__ __launch_bounds__(kFlatThreadsPerBlock, 1)
void rmsnorm_qk_flat_halfwarp_kernel(
    const float4* __restrict__ query,
    const float4* __restrict__ key,
    const float4* __restrict__ weight_q,
    const float4* __restrict__ weight_k,
    float eps,
    int total_head_rows,
    float4* __restrict__ query_norm,
    float4* __restrict__ key_norm) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head_row =
      static_cast<int>(blockIdx.x) * kFlatWarpsPerBlock + warp;
  if (head_row >= total_head_rows) {
    return;
  }

  const int sublane = lane & (kFlatSubgroupWidth - 1);
  const bool is_key = lane >= kFlatSubgroupWidth;
  const int head = head_row % kHeads;
  const int64_t base =
      static_cast<int64_t>(head_row) * kFlatVecsPerHead + sublane;
  const int weight_base = head * kFlatVecsPerHead + sublane;

  const float4* __restrict__ input = is_key ? key : query;
  const float4* __restrict__ weight = is_key ? weight_k : weight_q;
  float4* __restrict__ output = is_key ? key_norm : query_norm;

  const float4 input0 = input[base];
  const float4 input1 = input[base + kFlatSubgroupWidth];
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
    sum += __shfl_down_sync(kFullWarp, sum, offset, kFlatSubgroupWidth);
  }

  float scale_slot = 0.0f;
  if (sublane == 0) {
    scale_slot = fast_rsqrt(fmaf(sum, kInvHeadSize, eps));
  }
  const float scale =
      __shfl_sync(kFullWarp, scale_slot, 0, kFlatSubgroupWidth);
  const float2 scale2 = make_float2(scale, scale);

  const float4 weight0 = __ldg(weight + weight_base);
  const float2 output01 =
      mul2(mul2(input01, scale2), make_float2(weight0.x, weight0.y));
  const float2 output23 =
      mul2(mul2(input23, scale2), make_float2(weight0.z, weight0.w));
  output[base] =
      make_float4(output01.x, output01.y, output23.x, output23.y);

  const float4 weight1 =
      __ldg(weight + weight_base + kFlatSubgroupWidth);
  const float2 output45 =
      mul2(mul2(input45, scale2), make_float2(weight1.x, weight1.y));
  const float2 output67 =
      mul2(mul2(input67, scale2), make_float2(weight1.z, weight1.w));
  output[base + kFlatSubgroupWidth] =
      make_float4(output45.x, output45.y, output67.x, output67.y);
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
  TORCH_CHECK(
      rows64 <= std::numeric_limits<int>::max() / kHeads, "too many rows");
  const int rows = static_cast<int>(rows64);

  if (rows == 128) {
    constexpr int kHeadTiles = kHeads / kTokenG16WarpsPerBlock;
    const dim3 grid(2 * kHeadTiles, kTokenG16Streams);
    rmsnorm_qk_token_g16_128_kernel
        <<<grid, kTokenG16ThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const float4*>(query.data_ptr<float>()),
            reinterpret_cast<const float4*>(key.data_ptr<float>()),
            reinterpret_cast<const float4*>(weight_q.data_ptr<float>()),
            reinterpret_cast<const float4*>(weight_k.data_ptr<float>()),
            eps,
            reinterpret_cast<float4*>(query_norm.data_ptr<float>()),
            reinterpret_cast<float4*>(key_norm.data_ptr<float>()));
  } else if (rows == 256) {
    const dim3 grid(rows, kHeads / kG8HeadsPerBlock);
    rmsnorm_qk_g8_256_kernel<<<grid, kG8ThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const Pack256*>(query.data_ptr<float>()),
        reinterpret_cast<const Pack256*>(key.data_ptr<float>()),
        reinterpret_cast<const Pack256*>(weight_q.data_ptr<float>()),
        reinterpret_cast<const Pack256*>(weight_k.data_ptr<float>()),
        eps,
        reinterpret_cast<Pack256*>(query_norm.data_ptr<float>()),
        reinterpret_cast<Pack256*>(key_norm.data_ptr<float>()));
  } else if (rows <= kStream4MaxRows) {
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
  } else {
    const int total_head_rows = rows * kHeads;
    const int blocks =
        (total_head_rows + kFlatWarpsPerBlock - 1) / kFlatWarpsPerBlock;
    rmsnorm_qk_flat_halfwarp_kernel<<<blocks, kFlatThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(query.data_ptr<float>()),
        reinterpret_cast<const float4*>(key.data_ptr<float>()),
        reinterpret_cast<const float4*>(weight_q.data_ptr<float>()),
        reinterpret_cast<const float4*>(weight_k.data_ptr<float>()),
        eps,
        total_head_rows,
        reinterpret_cast<float4*>(query_norm.data_ptr<float>()),
        reinterpret_cast<float4*>(key_norm.data_ptr<float>()));
  }

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "flux RMSNorm kernel launch failed: ",
              cudaGetErrorString(error));
}
