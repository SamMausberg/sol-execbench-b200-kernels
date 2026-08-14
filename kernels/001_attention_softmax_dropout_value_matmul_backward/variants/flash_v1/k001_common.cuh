// SPDX-License-Identifier: Apache-2.0
// Shared dims, constants and index helpers for kernel 001.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>

namespace k001 {

using bf16 = cutlass::bfloat16_t;

static constexpr int H = 80;
static constexpr int KVH = 8;
static constexpr int G = 10;
static constexpr int D = 128;
static constexpr int TILE = 128;
static constexpr int NTHREADS = 128;

struct Dims {
  int B, Sq, Skv;
  int n_tiles;
  int m_tiles;
  int split;
  float inv;
};

// --------------------------------------------------------------------------
// Shared index helpers (validated on the dev GPU through the fallback path).
// --------------------------------------------------------------------------
__host__ __device__ inline int64_t delta_part_idx(const Dims& d, int nt, int b,
                                                  int h, int i) {
  return ((int64_t)nt * d.B * H + (int64_t)b * H + h) * d.Sq + i;
}
__host__ __device__ inline int64_t wgt_off(const Dims& d, int b, int h, int i,
                                           int n) {
  return (((int64_t)b * H + h) * d.Sq + i) * d.Skv + n;
}
__host__ __device__ inline int64_t dO_off(const Dims& d, int b, int i, int h,
                                          int dd) {
  return (((int64_t)b * d.Sq + i) * H + h) * D + dd;
}
__host__ __device__ inline int64_t v_off(const Dims& d, int b, int kvh, int n,
                                         int dd) {
  return (((int64_t)b * KVH + kvh) * d.Skv + n) * D + dd;
}

__device__ inline float ds_math(float dp, float w, bool m, float inv,
                                float delta) {
  float dw = m ? dp * inv : 0.0f;
  return w * (dw - delta);
}


// SM100 launchers (defined in k1_sm100.cu / k2_sm100.cu).
int k001_sm100_available();
void k001_launch_k1(const void* Wd, const void* dO, const void* V, void* dV,
                    float* delta_part, float* dv_ws, Dims dims, int kv_aligned,
                    cudaStream_t stream);
void k001_launch_k2(const void* W, const void* mask, const void* dO,
                    const void* V, const float* delta_part, void* dS,
                    const float* dv_ws, void* dV, Dims dims, int kv_aligned,
                    cudaStream_t stream);

}  // namespace k001
