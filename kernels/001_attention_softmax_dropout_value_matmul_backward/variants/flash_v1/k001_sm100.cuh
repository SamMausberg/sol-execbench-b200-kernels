// SPDX-License-Identifier: Apache-2.0
// Shared CuTe/tcgen05 machinery for kernel 001 (SM100 path).
#pragma once
#include "k001_common.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

namespace k001 {
using namespace cute;

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

// NOTE: the contiguous mode of every gmem tensor must carry a *static* _1
// stride -- a runtime int64 stride of 1 is not provably contiguous to CuTe and
// produces an invalid TMA descriptor (2-byte stride on a non-inner dim).

inline auto make_tma_wd(const bf16* p, const Dims& d) {
  // (kv, q, h, b), kv contiguous -- matches the MN-major (kv-contiguous) A tile.
  auto g = make_tensor(make_gmem_ptr(p), make_shape(d.Skv, d.Sq, Int<H>{}, d.B),
                       make_stride(_1{}, (int64_t)d.Skv, (int64_t)d.Sq * d.Skv,
                                   (int64_t)H * d.Sq * d.Skv));
  return make_tma_atom(SM90_TMA_LOAD{}, g, SWdLayout{},
                       make_shape(Int<TILE>{}, Int<TILE>{}));
}
inline auto make_tma_domn(const bf16* p, const Dims& d) {
  // (d, q, h, b), d contiguous -- MN-major B tile of MmaDV.
  auto g = make_tensor(make_gmem_ptr(p), make_shape(Int<D>{}, d.Sq, Int<H>{}, d.B),
                       make_stride(_1{}, (int64_t)H * D, (int64_t)D,
                                   (int64_t)d.Sq * H * D));
  return make_tma_atom(SM90_TMA_LOAD{}, g, SDOmnLayout{},
                       make_shape(Int<D>{}, Int<TILE>{}));
}
inline auto make_tma_dok(const bf16* p, const Dims& d) {
  // (q, d, h, b), d contiguous -- K-major A tile of MmaDP.
  auto g = make_tensor(make_gmem_ptr(p), make_shape(d.Sq, Int<D>{}, Int<H>{}, d.B),
                       make_stride((int64_t)H * D, _1{}, (int64_t)D,
                                   (int64_t)d.Sq * H * D));
  return make_tma_atom(SM90_TMA_LOAD{}, g, SDOkLayout{},
                       make_shape(Int<TILE>{}, Int<D>{}));
}
inline auto make_tma_v(const bf16* p, const Dims& d) {
  // (kv, d, kvh, b), d contiguous -- K-major B tile of MmaDP.
  auto g = make_tensor(make_gmem_ptr(p),
                       make_shape(d.Skv, Int<D>{}, Int<KVH>{}, d.B),
                       make_stride((int64_t)D, _1{}, (int64_t)d.Skv * D,
                                   (int64_t)KVH * d.Skv * D));
  return make_tma_atom(SM90_TMA_LOAD{}, g, SVLayout{},
                       make_shape(Int<TILE>{}, Int<D>{}));
}
inline auto make_tma_w(const bf16* p, const Dims& d) {
  // (q, kv, h, b), kv contiguous -- matches SWLayout's (row=q, col=kv) modes
  // read by the epilogue as (my_row, c).
  auto g = make_tensor(make_gmem_ptr(p), make_shape(d.Sq, d.Skv, Int<H>{}, d.B),
                       make_stride((int64_t)d.Skv, _1{}, (int64_t)d.Sq * d.Skv,
                                   (int64_t)H * d.Sq * d.Skv));
  return make_tma_atom(SM90_TMA_LOAD{}, g, SWLayout{},
                       make_shape(Int<TILE>{}, Int<TILE>{}));
}
inline auto make_tma_m(const uint8_t* p, const Dims& d) {
  auto g = make_tensor(make_gmem_ptr(p), make_shape(d.Sq, d.Skv, Int<H>{}, d.B),
                       make_stride((int64_t)d.Skv, _1{}, (int64_t)d.Sq * d.Skv,
                                   (int64_t)H * d.Sq * d.Skv));
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


#endif  // K001_HAS_SM100
}  // namespace k001
