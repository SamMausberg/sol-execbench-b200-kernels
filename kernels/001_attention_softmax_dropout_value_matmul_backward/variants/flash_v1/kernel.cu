// SPDX-License-Identifier: Apache-2.0
//
// SOL-ExecBench problem 001: attention softmax + dropout + value-matmul
// backward. Host dispatch + fallback kernels; the SM100 tcgen05 kernels live
// in k1_sm100.cu / k2_sm100.cu (see k001_sm100.cuh for the shared machinery
// and k001_common.cuh for the math/indexing contract).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "k001_common.cuh"

namespace k001 {

// ==========================================================================
// Fallback path (correctness reference; runs on any arch, incl. SM120).
// ==========================================================================

__global__ void k1_fallback(const bf16* __restrict__ Wd,
                            const bf16* __restrict__ W,
                            const uint8_t* __restrict__ mask,
                            const bf16* __restrict__ dO,
                            const bf16* __restrict__ Vv, bf16* __restrict__ dV,
                            float* __restrict__ delta_part,
                            float* __restrict__ dv_ws, Dims dims) {
  const int nt = blockIdx.x;
  const int kvh = blockIdx.y % KVH;
  const int sp = blockIdx.y / KVH;
  const int b = blockIdx.z;
  const int n0 = nt * TILE;
  const int g_per = G / dims.split;
  const int g_begin = sp * g_per;
  const int cols_valid = max(0, min(TILE, dims.Skv - n0));

  extern __shared__ float smem_acc[];  // dV tile accumulator [TILE][D]
  for (int idx = threadIdx.x; idx < TILE * D; idx += NTHREADS)
    smem_acc[idx] = 0.0f;
  __syncthreads();

  for (int g = g_begin; g < g_begin + g_per; ++g) {
    const int h = kvh * G + g;
    for (int it = 0; it < dims.m_tiles; ++it) {
      const int i0 = it * TILE;
      const int rows_valid = max(0, min(TILE, dims.Sq - i0));
      // Phase A: one row per thread -> DeltaPart.
      const int r = threadIdx.x;
      if (r < rows_valid) {
        const int i = i0 + r;
        const bf16* dO_row = dO + dO_off(dims, b, i, h, 0);
        const bf16* w_row = W + wgt_off(dims, b, h, i, 0);
        const uint8_t* m_row = mask + wgt_off(dims, b, h, i, 0);
        float delta_acc = 0.0f;
        for (int n = 0; n < cols_valid; ++n) {
          if (!m_row[n0 + n]) continue;
          const bf16* v_row = Vv + v_off(dims, b, kvh, n0 + n, 0);
          float dp = 0.0f;
          for (int dd = 0; dd < D; ++dd)
            dp += float(dO_row[dd]) * float(v_row[dd]);
          delta_acc += dp * float(w_row[n0 + n]);
        }
        // Exact Delta: rowsum(dP * m * W) * inv (no Wd rounding involved).
        delta_part[delta_part_idx(dims, nt, b, h, i)] = delta_acc * dims.inv;
      }
      __syncthreads();
      // Phase B: dV accumulation, parallel over outputs (no races).
      for (int idx = threadIdx.x; idx < cols_valid * D; idx += NTHREADS) {
        const int n = idx / D, dd = idx % D;
        float acc = 0.0f;
        for (int r2 = 0; r2 < rows_valid; ++r2) {
          const int i = i0 + r2;
          acc += float(Wd[wgt_off(dims, b, h, i, n0 + n)]) *
                 float(dO[dO_off(dims, b, i, h, dd)]);
        }
        smem_acc[n * D + dd] += acc;
      }
      __syncthreads();
    }
  }

  for (int idx = threadIdx.x; idx < cols_valid * D; idx += NTHREADS) {
    const int n = idx / D, dd = idx % D;
    if (dims.split == 1)
      dV[v_off(dims, b, kvh, n0 + n, dd)] = bf16(smem_acc[n * D + dd]);
    else
      atomicAdd(&dv_ws[v_off(dims, b, kvh, n0 + n, dd)], smem_acc[n * D + dd]);
  }
}

__global__ void k2_fallback(const bf16* __restrict__ W,
                            const uint8_t* __restrict__ mask,
                            const bf16* __restrict__ dO,
                            const bf16* __restrict__ Vv,
                            const float* __restrict__ delta_part,
                            bf16* __restrict__ dS,
                            const float* __restrict__ dv_ws,
                            bf16* __restrict__ dV, Dims dims) {
  const int64_t m_ctas = (int64_t)dims.m_tiles * dims.B * H;
  const int64_t cta = blockIdx.x;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  if (cta >= m_ctas) {
    const int64_t total = (int64_t)dims.B * KVH * dims.Skv * D;
    const int64_t n_tail = (int64_t)gridDim.x - m_ctas;
    for (int64_t e = (cta - m_ctas) * NTHREADS + threadIdx.x; e < total;
         e += n_tail * NTHREADS)
      dV[e] = bf16(dv_ws[e]);
    return;
  }
  const int mt = (int)(cta % dims.m_tiles);
  const int bh = (int)(cta / dims.m_tiles);
  const int h = bh % H;
  const int b = bh / H;
  const int kvh = h / G;
  const int i = mt * TILE + threadIdx.x;
  if (i >= dims.Sq) return;

  float delta = 0.0f;
  for (int nt = 0; nt < dims.n_tiles; ++nt)
    delta += delta_part[delta_part_idx(dims, nt, b, h, i)];

  const bf16* dO_row = dO + dO_off(dims, b, i, h, 0);
  const bf16* w_row = W + wgt_off(dims, b, h, i, 0);
  const uint8_t* m_row = mask + wgt_off(dims, b, h, i, 0);
  bf16* ds_row = dS + wgt_off(dims, b, h, i, 0);
  for (int nn = 0; nn < dims.Skv; ++nn) {
    const bf16* v_row = Vv + v_off(dims, b, kvh, nn, 0);
    float dp = 0.0f;
    for (int dd = 0; dd < D; ++dd) dp += float(dO_row[dd]) * float(v_row[dd]);
    ds_row[nn] =
        bf16(ds_math(dp, float(w_row[nn]), m_row[nn] != 0, dims.inv, delta));
  }
}

// --------------------------------------------------------------------------
// Host dispatch
// --------------------------------------------------------------------------

static int pick_split(const Dims& d) {
  const int64_t base = (int64_t)d.n_tiles * KVH * d.B;
  for (int split : {1, 2, 5, 10})
    if (base * split >= 296 || split == 10) return split;
  return 1;
}

void run(torch::Tensor grad_attn_output, torch::Tensor attn_weights,
         torch::Tensor attn_weights_dropped, torch::Tensor value_states,
         torch::Tensor dropout_mask, double attention_dropout,
         torch::Tensor grad_attn_scores, torch::Tensor grad_value_states) {
  const at::cuda::CUDAGuard guard(grad_attn_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(grad_attn_output.is_contiguous() && attn_weights.is_contiguous() &&
              attn_weights_dropped.is_contiguous() &&
              value_states.is_contiguous() && dropout_mask.is_contiguous(),
              "expected contiguous inputs");

  Dims dims;
  dims.B = (int)grad_attn_output.size(0);
  dims.Sq = (int)grad_attn_output.size(1);
  dims.Skv = (int)value_states.size(2);
  dims.n_tiles = (dims.Skv + TILE - 1) / TILE;
  dims.m_tiles = (dims.Sq + TILE - 1) / TILE;
  const float p = (float)attention_dropout;
  dims.inv = p > 0.f ? 1.0f / (1.0f - p) : 1.0f;
  dims.split = pick_split(dims);

  auto opts_f32 = grad_attn_output.options().dtype(torch::kFloat32);
  torch::Tensor delta_part =
      torch::empty({(int64_t)dims.n_tiles * dims.B * H * dims.Sq}, opts_f32);
  torch::Tensor dv_ws;
  float* dv_ws_ptr = nullptr;
  if (dims.split > 1) {
    dv_ws = torch::zeros({(int64_t)dims.B * KVH * dims.Skv * D}, opts_f32);
    dv_ws_ptr = dv_ws.data_ptr<float>();
  }

  const bf16* dO = reinterpret_cast<const bf16*>(grad_attn_output.data_ptr());
  const bf16* W = reinterpret_cast<const bf16*>(attn_weights.data_ptr());
  const bf16* Wd =
      reinterpret_cast<const bf16*>(attn_weights_dropped.data_ptr());
  const bf16* V = reinterpret_cast<const bf16*>(value_states.data_ptr());
  const uint8_t* mask =
      reinterpret_cast<const uint8_t*>(dropout_mask.data_ptr());
  bf16* dS = reinterpret_cast<bf16*>(grad_attn_scores.data_ptr());
  bf16* dV = reinterpret_cast<bf16*>(grad_value_states.data_ptr());

  const int kv_aligned = (dims.Skv % 16 == 0) ? 1 : 0;
  static const bool debug_tma = getenv("K001_BUILD_TMA") != nullptr;
  if (debug_tma && kv_aligned) k001_debug_build_tma(Wd, W, mask, dO, V, dims);
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // Skv < 128: degenerate/peaked-softmax shapes go through the exact-Delta
  // fallback (they are tiny anyway; no benchmark workload hits this).
  const bool use_tc = (prop->major == 10) && dims.Skv >= 128;

  const int64_t m_ctas = (int64_t)dims.m_tiles * dims.B * H;
  const int64_t tail = (dims.split > 1) ? 148 : 0;

  if (!use_tc) {
    dim3 g1(dims.n_tiles, KVH * dims.split, dims.B);
    size_t smem1 = TILE * D * sizeof(float);
    static bool fb_attr = false;
    if (!fb_attr) {
      cudaFuncSetAttribute(k1_fallback,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem1);
      fb_attr = true;
    }
    k1_fallback<<<g1, NTHREADS, smem1, stream>>>(
        Wd, W, mask, dO, V, dV, delta_part.data_ptr<float>(), dv_ws_ptr, dims);
    k2_fallback<<<dim3((unsigned)(m_ctas + tail)), NTHREADS, 0, stream>>>(
        W, mask, dO, V, delta_part.data_ptr<float>(), dS, dv_ws_ptr, dV, dims);
    return;
  }

  if (k001_sm100_available()) {
    k001_launch_k1(Wd, dO, V, dV, delta_part.data_ptr<float>(), dv_ws_ptr,
                   dims, kv_aligned, stream);
    k001_launch_k2(W, mask, dO, V, delta_part.data_ptr<float>(), dS, dv_ws_ptr,
                   dV, dims, kv_aligned, stream);
  } else {
    TORCH_CHECK(false, "SM100 device but binary built without SM100 support");
  }
}

}  // namespace k001

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &k001::run, "kernel 001 attention backward (SOL-ExecBench)");
}
