// SPDX-License-Identifier: Apache-2.0
//
// SOL-ExecBench problem 001 -- "safe" variant: cuBLAS batched GEMMs (via ATen)
// plus one fused elementwise kernel. No CUTLASS/tcgen05 dependencies.
//
//   dO_t = transpose(dO)                        (ATen copy kernel)
//   dV   = bmm(Wd_cat^T, dO_cat)                (GQA folded into K = 10*Sq)
//   dP   = bmm(dO_cat, V^T) -> bf16 workspace
//   fused: Delta = rowsum(dP*m*W)*inv ; dS = W*(dP*m*inv - Delta)
//          one smem-staged pass over dP/W/m per row block.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace k001s {

static constexpr int H = 80;
static constexpr int KVH = 8;
static constexpr int G = 10;
static constexpr int D = 128;

using bf16 = __nv_bfloat16;

// One CTA processes ROWS_PER_CTA consecutive rows of the flattened
// [B*H*Sq, Skv] problem; each warp owns one row staged in shared memory.
template <int THREADS, bool ALIGNED>
__global__ void fused_delta_ds(const bf16* __restrict__ dP,
                               const bf16* __restrict__ W,
                               const uint8_t* __restrict__ mask,
                               bf16* __restrict__ dS, int64_t n_rows, int skv,
                               float inv) {
  constexpr int WARPS = THREADS / 32;
  extern __shared__ char smem_raw[];
  // Layout per warp: dP row (bf16), W row (bf16), m row (uint8)
  const int64_t row = (int64_t)blockIdx.x * WARPS + threadIdx.x / 32;
  if (row >= n_rows) return;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int skv_pad = (skv + 15) & ~15;
  bf16* s_dp = reinterpret_cast<bf16*>(smem_raw) + (int64_t)warp * skv_pad;
  bf16* s_w = reinterpret_cast<bf16*>(smem_raw) + (int64_t)WARPS * skv_pad +
              (int64_t)warp * skv_pad;
  uint8_t* s_m = reinterpret_cast<uint8_t*>(
                     reinterpret_cast<bf16*>(smem_raw) + (int64_t)2 * WARPS * skv_pad) +
                 (int64_t)warp * skv_pad;

  const bf16* dp_row = dP + row * skv;
  const bf16* w_row = W + row * skv;
  const uint8_t* m_row = mask + row * skv;

  float delta = 0.0f;
  if (ALIGNED) {
    for (int c8 = lane * 8; c8 < skv; c8 += 32 * 8) {
      uint4 dpv = *reinterpret_cast<const uint4*>(dp_row + c8);
      uint4 wv = *reinterpret_cast<const uint4*>(w_row + c8);
      uint2 mv = *reinterpret_cast<const uint2*>(m_row + c8);
      *reinterpret_cast<uint4*>(s_dp + c8) = dpv;
      *reinterpret_cast<uint4*>(s_w + c8) = wv;
      *reinterpret_cast<uint2*>(reinterpret_cast<uint8_t*>(s_m) + c8) = mv;
      const bf16* dpe = reinterpret_cast<const bf16*>(&dpv);
      const bf16* we = reinterpret_cast<const bf16*>(&wv);
      const uint8_t* me = reinterpret_cast<const uint8_t*>(&mv);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (me[j]) delta += __bfloat162float(dpe[j]) * __bfloat162float(we[j]);
    }
  } else {
    for (int c = lane; c < skv; c += 32) {
      bf16 dpv = dp_row[c];
      bf16 wv = w_row[c];
      uint8_t mv = m_row[c];
      s_dp[c] = dpv;
      s_w[c] = wv;
      s_m[c] = mv;
      if (mv) delta += __bfloat162float(dpv) * __bfloat162float(wv);
    }
  }
  // Warp reduction; then Delta includes the inv factor.
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    delta += __shfl_xor_sync(0xffffffff, delta, off);
  delta *= inv;

  __syncwarp();
  bf16* ds_row = dS + row * skv;
  if (ALIGNED) {
    for (int c8 = lane * 8; c8 < skv; c8 += 32 * 8) {
      uint4 out;
      bf16* oe = reinterpret_cast<bf16*>(&out);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int c = c8 + j;
        float wv = __bfloat162float(s_w[c]);
        float dw = s_m[c] ? __bfloat162float(s_dp[c]) * inv : 0.0f;
        oe[j] = __float2bfloat16(wv * (dw - delta));
      }
      *reinterpret_cast<uint4*>(ds_row + c8) = out;
    }
  } else {
    for (int c = lane; c < skv; c += 32) {
      float wv = __bfloat162float(s_w[c]);
      float dw = s_m[c] ? __bfloat162float(s_dp[c]) * inv : 0.0f;
      ds_row[c] = __float2bfloat16(wv * (dw - delta));
    }
  }
}

void run(torch::Tensor grad_attn_output, torch::Tensor attn_weights,
         torch::Tensor attn_weights_dropped, torch::Tensor value_states,
         torch::Tensor dropout_mask, double attention_dropout,
         torch::Tensor grad_attn_scores, torch::Tensor grad_value_states) {
  const at::cuda::CUDAGuard guard(grad_attn_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t B = grad_attn_output.size(0);
  const int64_t Sq = grad_attn_output.size(1);
  const int64_t Skv = value_states.size(2);
  const float p = (float)attention_dropout;
  const float inv = p > 0.f ? 1.0f / (1.0f - p) : 1.0f;

  auto dO_t = grad_attn_output.transpose(1, 2).contiguous();  // [B,H,Sq,D]
  auto dO_cat = dO_t.view({B * KVH, G * Sq, D});
  auto Wd_cat = attn_weights_dropped.view({B * KVH, G * Sq, Skv});

  // dV = Wd_cat^T @ dO_cat  (fp32 accumulation inside cuBLAS)
  auto dV_view = grad_value_states.view({B * KVH, Skv, D});
  at::bmm_out(dV_view, Wd_cat.transpose(1, 2), dO_cat);

  // dP = dO_cat @ V^T  -> bf16 workspace
  auto dP = at::bmm(dO_cat, value_states.view({B * KVH, Skv, D}).transpose(1, 2));

  const int64_t n_rows = B * H * Sq;
  constexpr int THREADS = 256;
  constexpr int WARPS = THREADS / 32;
  const int skv_pad = (int)((Skv + 15) & ~15);
  const size_t smem = (size_t)WARPS * skv_pad * (2 + 2 + 1);
  const bool aligned = (Skv % 8 == 0);
  const int64_t blocks = (n_rows + WARPS - 1) / WARPS;

  auto* dPp = reinterpret_cast<const bf16*>(dP.data_ptr());
  auto* Wp = reinterpret_cast<const bf16*>(attn_weights.data_ptr());
  auto* mp = reinterpret_cast<const uint8_t*>(dropout_mask.data_ptr());
  auto* dSp = reinterpret_cast<bf16*>(grad_attn_scores.data_ptr());

  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(fused_delta_ds<THREADS, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    cudaFuncSetAttribute(fused_delta_ds<THREADS, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    attr_set = true;
  }
  TORCH_CHECK(smem <= 227 * 1024, "Skv too large for safe variant smem");
  if (aligned)
    fused_delta_ds<THREADS, true><<<(unsigned)blocks, THREADS, smem, stream>>>(
        dPp, Wp, mp, dSp, n_rows, (int)Skv, inv);
  else
    fused_delta_ds<THREADS, false><<<(unsigned)blocks, THREADS, smem, stream>>>(
        dPp, Wp, mp, dSp, n_rows, (int)Skv, inv);
}

}  // namespace k001s

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &k001s::run, "kernel 001 attention backward (safe variant)");
}
