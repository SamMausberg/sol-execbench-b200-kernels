// SPDX-License-Identifier: Apache-2.0
#include "k001_sm100.cuh"

namespace k001 {
#if defined(K001_HAS_SM100)
using namespace cute;

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
        // Make the generic-proxy stores visible to the tcgen05 async proxy.
        cutlass::arch::fence_view_async_shared();
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


#endif  // K001_HAS_SM100

void k001_debug_build_tma(const void* Wd, const void* W, const void* mask,
                          const void* dO, const void* V, Dims dims) {
#if defined(K001_HAS_SM100)
  auto a0 = make_tma_wd((const bf16*)Wd, dims);
  auto a1 = make_tma_dok((const bf16*)dO, dims);
  auto a2 = make_tma_domn((const bf16*)dO, dims);
  auto a3 = make_tma_v((const bf16*)V, dims);
  auto a4 = make_tma_w((const bf16*)W, dims);
  auto a5 = make_tma_m((const uint8_t*)mask, dims);
  (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
#endif
}

int k001_sm100_available() {
#if defined(K001_HAS_SM100)
  return 1;
#else
  return 0;
#endif
}

void k001_launch_k1(const void* Wd, const void* dO, const void* V, void* dV,
                    float* delta_part, float* dv_ws, Dims dims, int kv_aligned,
                    cudaStream_t stream) {
#if defined(K001_HAS_SM100)
  K1Params p1{make_tma_wd((const bf16*)Wd, dims),
              make_tma_dok((const bf16*)dO, dims),
              make_tma_domn((const bf16*)dO, dims),
              make_tma_v((const bf16*)V, dims),
              (const bf16*)Wd,
              (bf16*)dV,
              delta_part,
              dv_ws,
              dims,
              kv_aligned};
  static bool attrs_set = false;
  if (!attrs_set) {
    cudaFuncSetAttribute(k1_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sizeof(K1Smem));
    attrs_set = true;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(dims.n_tiles, KVH * dims.split, dims.B);
  cfg.blockDim = dim3(NTHREADS, 1, 1);
  cfg.dynamicSmemBytes = sizeof(K1Smem);
  cfg.stream = stream;
  cfg.numAttrs = 0;
  cudaLaunchKernelEx(&cfg, k1_sm100, p1);
#endif
}

}  // namespace k001
