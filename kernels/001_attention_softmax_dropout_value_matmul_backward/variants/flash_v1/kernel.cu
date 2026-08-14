// SPDX-License-Identifier: Apache-2.0
//
// SOL-ExecBench problem 001: attention softmax + dropout + value-matmul backward.
//
// Math (per batch b, head h; W = attn_weights, Wd = attn_weights_dropped,
// m = dropout_mask, p = attention_dropout, inv = 1/(1-p)):
//   dP    = dO @ V^T                       [Sq, Skv]  fp32 accum
//   Delta = rowsum(dP * Wd)                [Sq]
//   dS    = W * (dP * m * inv - Delta)     -> bf16 out
//   dV    = sum_g Wd[g]^T @ dO[g]          -> bf16 out (GQA sum over G=10)
// Delta uses Wd == bf16(W * m * inv): the flash-attention D-trick for this
// graph. The substitution error is ~bf16 eps of one factor, far inside the
// tolerance (atol 1e-5 / rtol 5% / matched-ratio 0.99).
//
// K1 grid (nt, kvh*split, b): streams Wd once; accumulates
//    dV[b,kvh,nt*128:+128,:] over K = G*Sq (GQA folded into the contraction)
//    and writes DeltaPart[nt,b,h,i] (this tile's 128 kv columns of Delta).
// K2 grid (mt * B*H [+ dV-convert tail]): dP = dO @ V^T per tile, reduces
//    DeltaPart over nt, applies the fused epilogue, writes dS. PDL launch;
//    only DeltaPart / dv_ws reads sit behind the grid dependency sync.
//
// SM100: tcgen05 MMAs via CuTe atoms + TMA/mbarrier pipeline.
// Other archs (SM120 dev GPU): same grid/indexing/predication/epilogue math
// with scalar fp32 compute — validates everything but tcgen05 itself.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

namespace k001 {

using namespace cute;

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

// ==========================================================================
// Fallback path (correctness reference; runs on any arch, incl. SM120).
// ==========================================================================

__global__ void k1_fallback(const bf16* __restrict__ Wd,
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
        const bf16* wd_row = Wd + wgt_off(dims, b, h, i, 0);
        float delta_acc = 0.0f;
        for (int n = 0; n < cols_valid; ++n) {
          const bf16* v_row = Vv + v_off(dims, b, kvh, n0 + n, 0);
          float dp = 0.0f;
          for (int dd = 0; dd < D; ++dd)
            dp += float(dO_row[dd]) * float(v_row[dd]);
          delta_acc += dp * float(wd_row[n0 + n]);
        }
        delta_part[delta_part_idx(dims, nt, b, h, i)] = delta_acc;
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

// ==========================================================================
// SM100 tcgen05 path.
// ==========================================================================
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#define K001_HAS_SM100 1
#endif

#if defined(K001_HAS_SM100)

// MMA_DP: dP[q,kv] = dO[q,d] @ V[kv,d]^T  -> A K-major, B K-major
using MmaDP = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_SS<bf16, bf16, float, TILE, TILE, UMMA::Major::K,
                         UMMA::Major::K>{}));
// MMA_DV: dV[kv,d] += Wd^T[kv,q] @ dO[q,d] -> A MN-major (smem tile stored
// (q,kv) with kv contiguous), B MN-major (smem tile stored (q,d), d contig).
using MmaDV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_SS<bf16, bf16, float, TILE, D, UMMA::Major::MN,
                         UMMA::Major::MN>{}));

inline auto sWd_layout() {  // A of MmaDV: logical (kv "M", q "K")
  return UMMA::tile_to_mma_shape(
      UMMA::Layout_MN_SW128_Atom<bf16>{},
      partition_shape_A(MmaDV{}, make_shape(Int<TILE>{}, Int<TILE>{})));
}
inline auto sDOmn_layout() {  // B of MmaDV: logical (d "N", q "K")
  return UMMA::tile_to_mma_shape(
      UMMA::Layout_MN_SW128_Atom<bf16>{},
      partition_shape_B(MmaDV{}, make_shape(Int<D>{}, Int<TILE>{})));
}
inline auto sDOk_layout() {  // A of MmaDP: logical (q "M", d "K")
  return UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<bf16>{},
      partition_shape_A(MmaDP{}, make_shape(Int<TILE>{}, Int<D>{})));
}
inline auto sV_layout() {  // B of MmaDP: logical (kv "N", d "K")
  return UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<bf16>{},
      partition_shape_B(MmaDP{}, make_shape(Int<TILE>{}, Int<D>{})));
}
inline auto sW_layout() {  // epilogue-only, swizzled vs bank conflicts
  return tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{},
                       Shape<Int<TILE>, Int<TILE>>{});
}
inline auto sM_layout() {
  return tile_to_shape(UMMA::Layout_K_SW128_Atom<uint8_t>{},
                       Shape<Int<TILE>, Int<TILE>>{});
}

using SWdLayout = decltype(sWd_layout());
using SDOmnLayout = decltype(sDOmn_layout());
using SDOkLayout = decltype(sDOk_layout());
using SVLayout = decltype(sV_layout());
using SWLayout = decltype(sW_layout());
using SMLayout = decltype(sM_layout());

static constexpr int kStages = 2;

struct K1Smem {
  alignas(1024) cute::array<bf16, cosize_v<SVLayout>> v;
  alignas(1024) cute::array<bf16, kStages * cosize_v<SWdLayout>> wd;
  alignas(1024) cute::array<bf16, kStages * cosize_v<SDOkLayout>> dok;
  alignas(1024) cute::array<bf16, kStages * cosize_v<SDOmnLayout>> domn;
  alignas(16) uint64_t op_full[kStages];
  alignas(16) uint64_t mma_done[kStages];
  alignas(16) uint64_t epi_free[kStages];
  alignas(16) uint64_t v_full;
  alignas(16) uint32_t tmem_base;
};

// Padded staging tile for coalesced bf16 writeout: [TILE rows][D+8 cols].
// Pitch 136 keeps 16-byte alignment of 8-element chunks in every row.
static constexpr int kStagePitch = D + 8;

struct K2Smem {
  alignas(1024) cute::array<bf16, cosize_v<SDOkLayout>> dok;
  alignas(1024) cute::array<bf16, kStages * cosize_v<SVLayout>> v;
  alignas(1024) cute::array<bf16, kStages * cosize_v<SWLayout>> w;
  alignas(1024) cute::array<uint8_t, kStages * cosize_v<SMLayout>> m;
  alignas(1024) cute::array<bf16, TILE * kStagePitch> stage;
  alignas(16) float delta[TILE];
  alignas(16) uint64_t op_full[kStages];
  alignas(16) uint64_t epi_full[kStages];
  alignas(16) uint64_t mma_done[kStages];
  alignas(16) uint64_t epi_free[kStages];
  alignas(16) uint64_t do_full;
  alignas(16) uint32_t tmem_base;
};

static_assert(sizeof(K1Smem) <= 232448, "K1 smem exceeds SM100 limit");
static_assert(sizeof(K2Smem) <= 232448, "K2 smem exceeds SM100 limit");

// Logical (m, k) coordinate into an MMA-shaped ((AtomM, AtomK), RestM, RestK)
// smem tensor with AtomM = 128, AtomK = 16 (bf16).
template <class T>
__device__ inline auto& smem_mk(T& t, int m, int k) {
  return t(make_coord(make_coord(m, k % 16), _0{}, k / 16));
}

template <class T>
static auto make_gmem_4d(const T* p, int64_t s0, int64_t s1, int64_t s2,
                         int64_t s3, int d0, int d1, int d2, int d3) {
  return make_tensor(make_gmem_ptr(p), make_shape(d0, d1, d2, d3),
                     make_stride(s0, s1, s2, s3));
}

inline auto make_tma_wd(const bf16* p, const Dims& d) {
  auto g = make_gmem_4d(p, (int64_t)1, (int64_t)d.Skv, (int64_t)d.Sq * d.Skv,
                        (int64_t)H * d.Sq * d.Skv, d.Skv, d.Sq, H, d.B);
  return make_tma_atom(SM90_TMA_LOAD{}, g, SWdLayout{},
                       make_shape(Int<TILE>{}, Int<TILE>{}));
}
inline auto make_tma_domn(const bf16* p, const Dims& d) {
  auto g = make_gmem_4d(p, (int64_t)1, (int64_t)H * D, (int64_t)D,
                        (int64_t)d.Sq * H * D, D, d.Sq, H, d.B);
  return make_tma_atom(SM90_TMA_LOAD{}, g, SDOmnLayout{},
                       make_shape(Int<D>{}, Int<TILE>{}));
}
inline auto make_tma_dok(const bf16* p, const Dims& d) {
  auto g = make_gmem_4d(p, (int64_t)H * D, (int64_t)1, (int64_t)D,
                        (int64_t)d.Sq * H * D, d.Sq, D, H, d.B);
  return make_tma_atom(SM90_TMA_LOAD{}, g, SDOkLayout{},
                       make_shape(Int<TILE>{}, Int<D>{}));
}
inline auto make_tma_v(const bf16* p, const Dims& d) {
  auto g = make_gmem_4d(p, (int64_t)D, (int64_t)1, (int64_t)d.Skv * D,
                        (int64_t)KVH * d.Skv * D, d.Skv, D, KVH, d.B);
  return make_tma_atom(SM90_TMA_LOAD{}, g, SVLayout{},
                       make_shape(Int<TILE>{}, Int<D>{}));
}
inline auto make_tma_w(const bf16* p, const Dims& d) {
  auto g = make_gmem_4d(p, (int64_t)1, (int64_t)d.Skv, (int64_t)d.Sq * d.Skv,
                        (int64_t)H * d.Sq * d.Skv, d.Skv, d.Sq, H, d.B);
  return make_tma_atom(SM90_TMA_LOAD{}, g, SWLayout{},
                       make_shape(Int<TILE>{}, Int<TILE>{}));
}
inline auto make_tma_m(const uint8_t* p, const Dims& d) {
  auto g = make_gmem_4d(p, (int64_t)1, (int64_t)d.Skv, (int64_t)d.Sq * d.Skv,
                        (int64_t)H * d.Sq * d.Skv, d.Skv, d.Sq, H, d.B);
  return make_tma_atom(SM90_TMA_LOAD{}, g, SMLayout{},
                       make_shape(Int<TILE>{}, Int<TILE>{}));
}

static const Dims kDummyDims{1, TILE, TILE, 1, 1, 1, 1.f};
using TmaWd = decltype(make_tma_wd(nullptr, kDummyDims));
using TmaDOmn = decltype(make_tma_domn(nullptr, kDummyDims));
using TmaDOk = decltype(make_tma_dok(nullptr, kDummyDims));
using TmaV = decltype(make_tma_v(nullptr, kDummyDims));
using TmaW = decltype(make_tma_w(nullptr, kDummyDims));
using TmaM = decltype(make_tma_m(nullptr, kDummyDims));

struct K1Params {
  TmaWd tma_wd;
  TmaDOk tma_dok;
  TmaDOmn tma_domn;
  TmaV tma_v;
  const bf16* wd;
  bf16* dV;
  float* delta_part;
  float* dv_ws;
  Dims dims;
  int kv_aligned;
};

struct K2Params {
  TmaDOk tma_dok;
  TmaV tma_v;
  TmaW tma_w;
  TmaM tma_m;
  const bf16* w;
  const uint8_t* mask;
  const float* delta_part;
  bf16* dS;
  const float* dv_ws;
  bf16* dV;
  Dims dims;
  int kv_aligned;
};

__device__ inline void mbar_arrive(uint64_t* bar) {
  uint32_t addr = cute::cast_smem_ptr_to_uint(bar);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr));
}

__global__ void __launch_bounds__(NTHREADS) k1_sm100(
    const __grid_constant__ K1Params p) {
  const Dims dims = p.dims;
  const int nt = blockIdx.x;
  const int kvh = blockIdx.y % KVH;
  const int sp = blockIdx.y / KVH;
  const int b = blockIdx.z;
  const int n0 = nt * TILE;
  const int g_per = G / dims.split;
  const int g_begin = sp * g_per;
  const int total_iters = g_per * dims.m_tiles;

  extern __shared__ char smem_raw[];
  K1Smem& smem = *reinterpret_cast<K1Smem*>(smem_raw);
  const int tid = threadIdx.x;
  const bool warp0 = (tid < 32);

  if (warp0 && elect_one_sync()) {
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(smem.op_full[s], 1);
      cute::initialize_barrier(smem.mma_done[s], 1);
      cute::initialize_barrier(smem.epi_free[s], NTHREADS);
    }
    cute::initialize_barrier(smem.v_full, 1);
  }
  __syncthreads();

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (warp0) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &smem.tmem_base);
  }
  __syncthreads();
  const uint32_t tmem_base = smem.tmem_base;

  Tensor sV = make_tensor(make_smem_ptr(smem.v.begin()), SVLayout{});
  auto sWd_stage = [&](int s) {
    return make_tensor(make_smem_ptr(smem.wd.begin() + s * cosize_v<SWdLayout>),
                       SWdLayout{});
  };
  auto sDOk_stage = [&](int s) {
    return make_tensor(
        make_smem_ptr(smem.dok.begin() + s * cosize_v<SDOkLayout>),
        SDOkLayout{});
  };
  auto sDOmn_stage = [&](int s) {
    return make_tensor(
        make_smem_ptr(smem.domn.begin() + s * cosize_v<SDOmnLayout>),
        SDOmnLayout{});
  };

  MmaDP mma_dp{};
  MmaDV mma_dv{};
  auto thr_dp = mma_dp.get_slice(_0{});
  auto thr_dv = mma_dv.get_slice(_0{});

  // Accumulator fragments in TMEM: dV at column 0, dP double-buffered after.
  Tensor cDV_dummy = make_tensor(
      make_gmem_ptr((float*)nullptr),
      make_layout(Shape<Int<TILE>, Int<D>>{}, LayoutRight{}));
  Tensor tDV = thr_dv.make_fragment_C(thr_dv.partition_C(cDV_dummy));
  tDV.data() = tmem_base;
  Tensor cDP_dummy = make_tensor(
      make_gmem_ptr((float*)nullptr),
      make_layout(Shape<Int<TILE>, Int<TILE>>{}, LayoutRight{}));
  Tensor tDP0 = thr_dp.make_fragment_C(thr_dp.partition_C(cDP_dummy));
  Tensor tDP1 = thr_dp.make_fragment_C(thr_dp.partition_C(cDP_dummy));
  tDP0.data() = tmem_base + D;
  tDP1.data() = tmem_base + D + TILE;

  // Gmem coord tensors.
  Tensor mWd =
      p.tma_wd.get_tma_tensor(make_shape(dims.Skv, dims.Sq, Int<H>{}, dims.B));
  Tensor mDOk =
      p.tma_dok.get_tma_tensor(make_shape(dims.Sq, Int<D>{}, Int<H>{}, dims.B));
  Tensor mDOmn = p.tma_domn.get_tma_tensor(
      make_shape(Int<D>{}, dims.Sq, Int<H>{}, dims.B));
  Tensor mV = p.tma_v.get_tma_tensor(
      make_shape(dims.Skv, Int<D>{}, Int<KVH>{}, dims.B));

  // V tile: load once.
  {
    Tensor gV = local_tile(mV(_, _, kvh, b), Shape<Int<TILE>, Int<D>>{},
                           make_coord(nt, 0));
    Tensor tCgV = thr_dp.partition_B(gV);
    if (warp0 && elect_one_sync()) {
      cute::set_barrier_transaction_bytes(
          smem.v_full, sizeof(bf16) * cosize_v<SVLayout>);
      auto [tg, ts] = tma_partition(p.tma_v, Int<0>{}, Layout<_1>{},
                                    group_modes<0, 3>(sV),
                                    group_modes<0, 3>(tCgV));
      copy(p.tma_v.with(smem.v_full), tg, ts);
    }
  }
  cute::wait_barrier(smem.v_full, 0);
  Tensor frgV = thr_dp.make_fragment_B(sV);

  constexpr uint32_t kOpBytesAligned =
      sizeof(bf16) * (cosize_v<SWdLayout> + cosize_v<SDOkLayout> +
                      cosize_v<SDOmnLayout>);
  constexpr uint32_t kOpBytesUnaligned =
      sizeof(bf16) * (cosize_v<SDOkLayout> + cosize_v<SDOmnLayout>);

  int phase_op = 0, phase_mma = 0, phase_epi = 0;
  // per-slot phases packed: slot s phase = (x >> s) & 1
  int ph_op = 0, ph_mma = 0, ph_epi = 0;

  mma_dv.accumulate_ = UMMA::ScaleOut::Zero;

  for (int iter = 0; iter < total_iters + 1; ++iter) {
    const int slot = iter % kStages;
    if (iter < total_iters) {
      const int g = g_begin + iter / dims.m_tiles;
      const int it = iter % dims.m_tiles;
      const int h = kvh * G + g;
      // wait for slot's previous epilogue to release smem
      if (iter >= kStages) {
        cute::wait_barrier(smem.epi_free[slot], (ph_epi >> slot) & 1);
        ph_epi ^= (1 << slot);
      }
      if (warp0 && elect_one_sync()) {
        cute::set_barrier_transaction_bytes(
            smem.op_full[slot],
            p.kv_aligned ? kOpBytesAligned : kOpBytesUnaligned);
        if (p.kv_aligned) {
          Tensor gWd = local_tile(mWd(_, _, h, b),
                                  Shape<Int<TILE>, Int<TILE>>{},
                                  make_coord(nt, it));
          Tensor tCgWd = thr_dv.partition_A(gWd);
          auto sWd_s = sWd_stage(slot);
          auto [tg, ts] = tma_partition(p.tma_wd, Int<0>{}, Layout<_1>{},
                                        group_modes<0, 3>(sWd_s),
                                        group_modes<0, 3>(tCgWd));
          copy(p.tma_wd.with(smem.op_full[slot]), tg, ts);
        }
        {
          Tensor gDOk = local_tile(mDOk(_, _, h, b), Shape<Int<TILE>, Int<D>>{},
                                   make_coord(it, 0));
          Tensor tCg = thr_dp.partition_A(gDOk);
          auto s_s = sDOk_stage(slot);
          auto [tg, ts] = tma_partition(p.tma_dok, Int<0>{}, Layout<_1>{},
                                        group_modes<0, 3>(s_s),
                                        group_modes<0, 3>(tCg));
          copy(p.tma_dok.with(smem.op_full[slot]), tg, ts);
        }
        {
          Tensor gDOmn = local_tile(mDOmn(_, _, h, b),
                                    Shape<Int<D>, Int<TILE>>{},
                                    make_coord(0, it));
          Tensor tCg = thr_dv.partition_B(gDOmn);
          auto s_s = sDOmn_stage(slot);
          auto [tg, ts] = tma_partition(p.tma_domn, Int<0>{}, Layout<_1>{},
                                        group_modes<0, 3>(s_s),
                                        group_modes<0, 3>(tCg));
          copy(p.tma_domn.with(smem.op_full[slot]), tg, ts);
        }
      }
      if (!p.kv_aligned) {
        const int i0 = it * TILE;
        const int rows_valid = max(0, min(TILE, dims.Sq - i0));
        const int cols_valid = max(0, min(TILE, dims.Skv - n0));
        auto sWd_s = sWd_stage(slot);
        for (int idx = tid; idx < TILE * TILE; idx += NTHREADS) {
          const int r = idx / TILE, c = idx % TILE;  // r = q row, c = kv col
          bf16 val = bf16(0.f);
          if (r < rows_valid && c < cols_valid)
            val = p.wd[wgt_off(dims, b, h, i0 + r, n0 + c)];
          // logical A coord (m = kv = c, k = q = r)
          smem_mk(sWd_s, c, r) = val;
        }
        __syncthreads();
      }
      // MMA issue (whole warp0; cute elects internally)
      if (warp0) {
        cute::wait_barrier(smem.op_full[slot], (ph_op >> slot) & 1);
        ph_op ^= (1 << slot);
        auto sWd_s = sWd_stage(slot);
        auto sDOk_s = sDOk_stage(slot);
        auto sDOmn_s = sDOmn_stage(slot);
        Tensor frgWd = thr_dv.make_fragment_A(sWd_s);
        Tensor frgDOmn = thr_dv.make_fragment_B(sDOmn_s);
        Tensor frgDOk = thr_dp.make_fragment_A(sDOk_s);
        Tensor tDP = (slot == 0) ? tDP0 : tDP1;
        for (int k = 0; k < size<2>(frgWd); ++k)
          gemm(mma_dv, frgWd(_, _, k), frgDOmn(_, _, k), tDV);
        mma_dv.accumulate_ = UMMA::ScaleOut::One;
        mma_dp.accumulate_ = UMMA::ScaleOut::Zero;
        for (int k = 0; k < size<2>(frgDOk); ++k) {
          gemm(mma_dp, frgDOk(_, _, k), frgV(_, _, k), tDP);
          mma_dp.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&smem.mma_done[slot]);
      }
    }
    // Epilogue for iter-1: DeltaPart.
    const int eiter = iter - 1;
    if (eiter >= 0) {
      const int eslot = eiter % kStages;
      const int g = g_begin + eiter / dims.m_tiles;
      const int it = eiter % dims.m_tiles;
      const int h = kvh * G + g;
      const int i0 = it * TILE;
      cute::wait_barrier(smem.mma_done[eslot], (ph_mma >> eslot) & 1);
      ph_mma ^= (1 << eslot);
      Tensor tDP = (eslot == 0) ? tDP0 : tDP1;
      auto tiled_t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tDP);
      auto thr_t2r = tiled_t2r.get_slice(tid);
      Tensor tDtP = thr_t2r.partition_S(tDP);
      Tensor cP = make_identity_tensor(Shape<Int<TILE>, Int<TILE>>{});
      Tensor tDcP = thr_t2r.partition_D(thr_dp.partition_C(cP));
      Tensor tDrP = make_tensor<float>(shape(tDcP));
      copy(tiled_t2r, tDtP, tDrP);
      cutlass::arch::fence_view_async_tmem_load();
      auto sWd_s = sWd_stage(eslot);
      const int my_row = get<0>(tDcP(_0{}));
      float delta_acc = 0.0f;
      CUTE_UNROLL
      for (int e = 0; e < size(tDrP); ++e) {
        const int c = get<1>(tDcP(e));
        delta_acc += tDrP(e) * float(smem_mk(sWd_s, c, my_row));
      }
      const int i = i0 + my_row;
      if (i < dims.Sq)
        p.delta_part[delta_part_idx(dims, nt, b, h, i)] = delta_acc;
      __syncthreads();
      mbar_arrive(&smem.epi_free[eslot]);
    }
  }

  // dV writeout (staged through the now-idle Wd smem for coalesced stores;
  // split > 1 uses fp32 atomics into the workspace instead).
  {
    auto tiled_t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tDV);
    auto thr_t2r = tiled_t2r.get_slice(tid);
    Tensor tDtA = thr_t2r.partition_S(tDV);
    Tensor cA = make_identity_tensor(Shape<Int<TILE>, Int<D>>{});
    Tensor tDcA = thr_t2r.partition_D(thr_dv.partition_C(cA));
    Tensor tDrA = make_tensor<float>(shape(tDcA));
    copy(tiled_t2r, tDtA, tDrA);
    cutlass::arch::fence_view_async_tmem_load();
    const int my_kv = get<0>(tDcA(_0{}));
    const int nn = n0 + my_kv;
    if (dims.split == 1) {
      bf16* stage = reinterpret_cast<bf16*>(smem.wd.begin());
      bf16* stage_row = stage + my_kv * kStagePitch;
      CUTE_UNROLL
      for (int e = 0; e < size(tDrA); ++e)
        stage_row[get<1>(tDcA(e))] = bf16(tDrA(e));
      __syncthreads();
      const int rows_valid = min(TILE, dims.Skv - n0);
      for (int idx = tid; idx < TILE * (D / 8); idx += NTHREADS) {
        const int r = idx / (D / 8), c8 = (idx % (D / 8)) * 8;
        if (r < rows_valid) {
          uint4 val = *reinterpret_cast<const uint4*>(stage + r * kStagePitch + c8);
          *reinterpret_cast<uint4*>(p.dV + v_off(dims, b, kvh, n0 + r, c8)) = val;
        }
      }
    } else if (nn < dims.Skv) {
      float* out = p.dv_ws + v_off(dims, b, kvh, nn, 0);
      CUTE_UNROLL
      for (int e = 0; e < size(tDrA); ++e)
        atomicAdd(&out[get<1>(tDcA(e))], tDrA(e));
    }
  }
  __syncthreads();
  if (warp0) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base, TmemAllocator::Sm100TmemCapacityColumns);
  }
  __threadfence();
  cudaTriggerProgrammaticLaunchCompletion();
}

__global__ void __launch_bounds__(NTHREADS) k2_sm100(
    const __grid_constant__ K2Params p) {
  const Dims dims = p.dims;
  const int64_t m_ctas = (int64_t)dims.m_tiles * dims.B * H;
  const int64_t cta = blockIdx.x;
  const int tid = threadIdx.x;

  if (cta >= m_ctas) {
    cudaGridDependencySynchronize();
    const int64_t total = (int64_t)dims.B * KVH * dims.Skv * D;
    const int64_t n_tail = (int64_t)gridDim.x - m_ctas;
    for (int64_t e = (cta - m_ctas) * NTHREADS + tid; e < total;
         e += n_tail * NTHREADS)
      p.dV[e] = bf16(p.dv_ws[e]);
    return;
  }

  const int mt = (int)(cta % dims.m_tiles);
  const int bh = (int)(cta / dims.m_tiles);
  const int h = bh % H;
  const int b = bh / H;
  const int kvh = h / G;
  const int i0 = mt * TILE;

  extern __shared__ char smem_raw[];
  K2Smem& smem = *reinterpret_cast<K2Smem*>(smem_raw);
  const bool warp0 = (tid < 32);

  if (warp0 && elect_one_sync()) {
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(smem.op_full[s], 1);
      cute::initialize_barrier(smem.epi_full[s], 1);
      cute::initialize_barrier(smem.mma_done[s], 1);
      cute::initialize_barrier(smem.epi_free[s], NTHREADS);
    }
    cute::initialize_barrier(smem.do_full, 1);
  }
  __syncthreads();

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (warp0) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &smem.tmem_base);
  }
  __syncthreads();
  const uint32_t tmem_base = smem.tmem_base;

  Tensor sDO = make_tensor(make_smem_ptr(smem.dok.begin()), SDOkLayout{});
  auto sV_stage = [&](int s) {
    return make_tensor(make_smem_ptr(smem.v.begin() + s * cosize_v<SVLayout>),
                       SVLayout{});
  };
  auto sW_stage = [&](int s) {
    return make_tensor(make_smem_ptr(smem.w.begin() + s * cosize_v<SWLayout>),
                       SWLayout{});
  };
  auto sM_stage = [&](int s) {
    return make_tensor(make_smem_ptr(smem.m.begin() + s * cosize_v<SMLayout>),
                       SMLayout{});
  };

  MmaDP mma_dp{};
  auto thr_dp = mma_dp.get_slice(_0{});
  Tensor cDP_dummy = make_tensor(
      make_gmem_ptr((float*)nullptr),
      make_layout(Shape<Int<TILE>, Int<TILE>>{}, LayoutRight{}));
  Tensor tDP0 = thr_dp.make_fragment_C(thr_dp.partition_C(cDP_dummy));
  Tensor tDP1 = thr_dp.make_fragment_C(thr_dp.partition_C(cDP_dummy));
  tDP0.data() = tmem_base;
  tDP1.data() = tmem_base + TILE;

  Tensor mDOk =
      p.tma_dok.get_tma_tensor(make_shape(dims.Sq, Int<D>{}, Int<H>{}, dims.B));
  Tensor mV = p.tma_v.get_tma_tensor(
      make_shape(dims.Skv, Int<D>{}, Int<KVH>{}, dims.B));
  Tensor mW =
      p.tma_w.get_tma_tensor(make_shape(dims.Skv, dims.Sq, Int<H>{}, dims.B));
  Tensor mM =
      p.tma_m.get_tma_tensor(make_shape(dims.Skv, dims.Sq, Int<H>{}, dims.B));

  // Prologue: dO tile (before the PDL dependency sync — it's an input).
  {
    Tensor gDO = local_tile(mDOk(_, _, h, b), Shape<Int<TILE>, Int<D>>{},
                            make_coord(mt, 0));
    Tensor tCg = thr_dp.partition_A(gDO);
    if (warp0 && elect_one_sync()) {
      cute::set_barrier_transaction_bytes(
          smem.do_full, sizeof(bf16) * cosize_v<SDOkLayout>);
      auto [tg, ts] = tma_partition(p.tma_dok, Int<0>{}, Layout<_1>{},
                                    group_modes<0, 3>(sDO),
                                    group_modes<0, 3>(tCg));
      copy(p.tma_dok.with(smem.do_full), tg, ts);
    }
  }

  cudaGridDependencySynchronize();
  {
    const int i = i0 + tid;
    float acc = 0.0f;
    if (i < dims.Sq)
      for (int nt = 0; nt < dims.n_tiles; ++nt)
        acc += p.delta_part[delta_part_idx(dims, nt, b, h, i)];
    smem.delta[tid] = acc;
  }
  cute::wait_barrier(smem.do_full, 0);
  __syncthreads();
  Tensor frgDO = thr_dp.make_fragment_A(sDO);

  int ph_op = 0, ph_epif = 0, ph_mma = 0, ph_epi = 0;
  const float inv = dims.inv;
  const int n_iters = dims.n_tiles;

  for (int iter = 0; iter < n_iters + 1; ++iter) {
    const int slot = iter % kStages;
    if (iter < n_iters) {
      if (iter >= kStages) {
        cute::wait_barrier(smem.epi_free[slot], (ph_epi >> slot) & 1);
        ph_epi ^= (1 << slot);
      }
      if (warp0 && elect_one_sync()) {
        cute::set_barrier_transaction_bytes(
            smem.op_full[slot], sizeof(bf16) * cosize_v<SVLayout>);
        Tensor gV = local_tile(mV(_, _, kvh, b), Shape<Int<TILE>, Int<D>>{},
                               make_coord(iter, 0));
        Tensor tCg = thr_dp.partition_B(gV);
        auto sV_s = sV_stage(slot);
        auto [tg, ts] = tma_partition(p.tma_v, Int<0>{}, Layout<_1>{},
                                      group_modes<0, 3>(sV_s),
                                      group_modes<0, 3>(tCg));
        copy(p.tma_v.with(smem.op_full[slot]), tg, ts);
        if (p.kv_aligned) {
          cute::set_barrier_transaction_bytes(
              smem.epi_full[slot],
              (uint32_t)(sizeof(bf16) * cosize_v<SWLayout> +
                         cosize_v<SMLayout>));
          Tensor gW = local_tile(mW(_, _, h, b), Shape<Int<TILE>, Int<TILE>>{},
                                 make_coord(iter, mt));
          auto sW_s = sW_stage(slot);
          auto [twg, tws] = tma_partition(p.tma_w, Int<0>{}, Layout<_1>{},
                                          group_modes<0, 2>(sW_s),
                                          group_modes<0, 2>(gW));
          copy(p.tma_w.with(smem.epi_full[slot]), twg, tws);
          Tensor gM = local_tile(mM(_, _, h, b), Shape<Int<TILE>, Int<TILE>>{},
                                 make_coord(iter, mt));
          auto sM_s = sM_stage(slot);
          auto [tmg, tms] = tma_partition(p.tma_m, Int<0>{}, Layout<_1>{},
                                          group_modes<0, 2>(sM_s),
                                          group_modes<0, 2>(gM));
          copy(p.tma_m.with(smem.epi_full[slot]), tmg, tms);
        }
      }
      if (!p.kv_aligned) {
        const int n0i = iter * TILE;
        const int rows_valid = max(0, min(TILE, dims.Sq - i0));
        const int cols_valid = max(0, min(TILE, dims.Skv - n0i));
        auto sW_s = sW_stage(slot);
        auto sM_s = sM_stage(slot);
        for (int idx = tid; idx < TILE * TILE; idx += NTHREADS) {
          const int r = idx / TILE, c = idx % TILE;
          bf16 wv = bf16(0.f);
          uint8_t mv = 0;
          if (r < rows_valid && c < cols_valid) {
            wv = p.w[wgt_off(dims, b, h, i0 + r, n0i + c)];
            mv = p.mask[wgt_off(dims, b, h, i0 + r, n0i + c)];
          }
          sW_s(make_coord(r, c)) = wv;
          sM_s(make_coord(r, c)) = mv;
        }
        __syncthreads();
      }
      if (warp0) {
        cute::wait_barrier(smem.op_full[slot], (ph_op >> slot) & 1);
        ph_op ^= (1 << slot);
        Tensor frgV = thr_dp.make_fragment_B(sV_stage(slot));
        Tensor tDP = (slot == 0) ? tDP0 : tDP1;
        mma_dp.accumulate_ = UMMA::ScaleOut::Zero;
        for (int k = 0; k < size<2>(frgDO); ++k) {
          gemm(mma_dp, frgDO(_, _, k), frgV(_, _, k), tDP);
          mma_dp.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&smem.mma_done[slot]);
      }
    }
    const int eiter = iter - 1;
    if (eiter >= 0) {
      const int eslot = eiter % kStages;
      const int n0i = eiter * TILE;
      cute::wait_barrier(smem.mma_done[eslot], (ph_mma >> eslot) & 1);
      ph_mma ^= (1 << eslot);
      if (p.kv_aligned) {
        cute::wait_barrier(smem.epi_full[eslot], (ph_epif >> eslot) & 1);
        ph_epif ^= (1 << eslot);
      }
      Tensor tDP = (eslot == 0) ? tDP0 : tDP1;
      auto tiled_t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tDP);
      auto thr_t2r = tiled_t2r.get_slice(tid);
      Tensor tDtP = thr_t2r.partition_S(tDP);
      Tensor cP = make_identity_tensor(Shape<Int<TILE>, Int<TILE>>{});
      Tensor tDcP = thr_t2r.partition_D(thr_dp.partition_C(cP));
      Tensor tDrP = make_tensor<float>(shape(tDcP));
      copy(tiled_t2r, tDtP, tDrP);
      cutlass::arch::fence_view_async_tmem_load();
      const int my_row = get<0>(tDcP(_0{}));
      auto sW_s = sW_stage(eslot);
      auto sM_s = sM_stage(eslot);
      const float delta = smem.delta[my_row];
      bf16* stage_row = smem.stage.begin() + my_row * kStagePitch;
      CUTE_UNROLL
      for (int e = 0; e < size(tDrP); ++e) {
        const int c = get<1>(tDcP(e));
        float wv = float(sW_s(make_coord(my_row, c)));
        bool mv = sM_s(make_coord(my_row, c)) != 0;
        stage_row[c] = bf16(ds_math(tDrP(e), wv, mv, inv, delta));
      }
      __syncthreads();
      // Cooperative coalesced writeout of the staged dS tile.
      {
        const int rows_valid = min(TILE, dims.Sq - i0);
        const int cols_valid = max(0, min(TILE, dims.Skv - n0i));
        if (p.kv_aligned) {
          for (int idx = tid; idx < TILE * (TILE / 8); idx += NTHREADS) {
            const int r = idx / (TILE / 8), c8 = (idx % (TILE / 8)) * 8;
            if (r < rows_valid && c8 < cols_valid) {
              uint4 val =
                  *reinterpret_cast<const uint4*>(smem.stage.begin() +
                                                  r * kStagePitch + c8);
              *reinterpret_cast<uint4*>(
                  p.dS + wgt_off(dims, b, h, i0 + r, n0i + c8)) = val;
            }
          }
        } else {
          for (int idx = tid; idx < TILE * TILE; idx += NTHREADS) {
            const int r = idx / TILE, c = idx % TILE;
            if (r < rows_valid && c < cols_valid)
              p.dS[wgt_off(dims, b, h, i0 + r, n0i + c)] =
                  smem.stage[r * kStagePitch + c];
          }
        }
      }
      __syncthreads();
      mbar_arrive(&smem.epi_free[eslot]);
    }
  }

  __syncthreads();
  if (warp0) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

#endif  // K001_HAS_SM100

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
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  const bool use_tc = (prop->major == 10);

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
        Wd, dO, V, dV, delta_part.data_ptr<float>(), dv_ws_ptr, dims);
    k2_fallback<<<dim3((unsigned)(m_ctas + tail)), NTHREADS, 0, stream>>>(
        W, mask, dO, V, delta_part.data_ptr<float>(), dS, dv_ws_ptr, dV, dims);
    return;
  }

#if defined(K001_HAS_SM100)
  K1Params p1{make_tma_wd(Wd, dims),
              make_tma_dok(dO, dims),
              make_tma_domn(dO, dims),
              make_tma_v(V, dims),
              Wd,
              dV,
              delta_part.data_ptr<float>(),
              dv_ws_ptr,
              dims,
              kv_aligned};
  K2Params p2{make_tma_dok(dO, dims),
              make_tma_v(V, dims),
              make_tma_w(W, dims),
              make_tma_m(mask, dims),
              W,
              mask,
              delta_part.data_ptr<float>(),
              dS,
              dv_ws_ptr,
              dV,
              dims,
              kv_aligned};

  static bool attrs_set = false;
  if (!attrs_set) {
    cudaFuncSetAttribute(k1_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sizeof(K1Smem));
    cudaFuncSetAttribute(k2_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sizeof(K2Smem));
    attrs_set = true;
  }

  {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(dims.n_tiles, KVH * dims.split, dims.B);
    cfg.blockDim = dim3(NTHREADS, 1, 1);
    cfg.dynamicSmemBytes = sizeof(K1Smem);
    cfg.stream = stream;
    cfg.numAttrs = 0;
    cudaLaunchKernelEx(&cfg, k1_sm100, p1);
  }
  {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((unsigned)(m_ctas + tail), 1, 1);
    cfg.blockDim = dim3(NTHREADS, 1, 1);
    cfg.dynamicSmemBytes = sizeof(K2Smem);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, k2_sm100, p2);
  }
#else
  TORCH_CHECK(false, "SM100 device but binary built without SM100 support");
#endif
}

}  // namespace k001

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &k001::run, "kernel 001 attention backward (SOL-ExecBench)");
}
