// SPDX-License-Identifier: Apache-2.0
#include "k001_sm100.cuh"

namespace k001 {
#if defined(K001_HAS_SM100)
using namespace cute;

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
      p.tma_w.get_tma_tensor(make_shape(dims.Sq, dims.Skv, Int<H>{}, dims.B));
  Tensor mM =
      p.tma_m.get_tma_tensor(make_shape(dims.Sq, dims.Skv, Int<H>{}, dims.B));

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
                                 make_coord(mt, iter));
          auto sW_s = sW_stage(slot);
          auto [twg, tws] = tma_partition(p.tma_w, Int<0>{}, Layout<_1>{},
                                          group_modes<0, 2>(sW_s),
                                          group_modes<0, 2>(gW));
          copy(p.tma_w.with(smem.epi_full[slot]), twg, tws);
          Tensor gM = local_tile(mM(_, _, h, b), Shape<Int<TILE>, Int<TILE>>{},
                                 make_coord(mt, iter));
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

void k001_launch_k2(const void* W, const void* mask, const void* dO,
                    const void* V, const float* delta_part, void* dS,
                    const float* dv_ws, void* dV, Dims dims, int kv_aligned,
                    cudaStream_t stream) {
#if defined(K001_HAS_SM100)
  K2Params p2{make_tma_dok((const bf16*)dO, dims),
              make_tma_v((const bf16*)V, dims),
              make_tma_w((const bf16*)W, dims),
              make_tma_m((const uint8_t*)mask, dims),
              (const bf16*)W,
              (const uint8_t*)mask,
              delta_part,
              (bf16*)dS,
              dv_ws,
              (bf16*)dV,
              dims,
              kv_aligned};
  static bool attrs_set = false;
  if (!attrs_set) {
    cudaFuncSetAttribute(k2_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sizeof(K2Smem));
    attrs_set = true;
  }
  const int64_t m_ctas = (int64_t)dims.m_tiles * dims.B * H;
  const int64_t tail = (dims.split > 1) ? 148 : 0;
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
#endif
}

}  // namespace k001
