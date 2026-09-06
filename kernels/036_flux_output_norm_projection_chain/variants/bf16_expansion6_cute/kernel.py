# SPDX-License-Identifier: Apache-2.0
"""Six independent BF16 products with FP32-to-BF16 input conversion in shared memory.

Component experiment only: A[8192,3072] @ B[64,3072].T + bias[64].
The K dimension is split into four actual 768-element reductions. No fragments
or input-derived values are retained across invocations.
"""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
import torch


BM, BN, BK = 128, 64, 64
STAGES = 2
THREADS = 128
SPLITS = 4
TERMS = 6


@cute.struct
class SharedStorage:
    ab_barriers: cute.struct.MemRange[cutlass.Int64, STAGES * 2]
    tmem_pointer: cutlass.Int32


@cute.jit
def convert_three(source: cute.Tensor, target0: cute.Tensor, target1: cute.Tensor,
                  target2: cute.Tensor, global_copy: cute.TiledCopy,
                  shared_copy: cute.TiledCopy, tidx: cutlass.Int32):
    g_part = global_copy.get_slice(tidx).partition_S(source)
    s_part0 = shared_copy.get_slice(tidx).partition_D(target0)
    s_part1 = shared_copy.get_slice(tidx).partition_D(target1)
    s_part2 = shared_copy.get_slice(tidx).partition_D(target2)
    values = cute.make_rmem_tensor(g_part.shape, cutlass.Float32)
    rounded = cute.make_rmem_tensor(g_part.shape, cutlass.BFloat16)
    cute.copy(global_copy, g_part, values)
    original = values.load()
    high = original.to(cutlass.BFloat16)
    rounded.store(high)
    cute.copy(shared_copy, rounded, s_part0)
    residual = original - high.to(cutlass.Float32)
    middle = residual.to(cutlass.BFloat16)
    rounded.store(middle)
    cute.copy(shared_copy, rounded, s_part1)
    low = (residual - middle.to(cutlass.Float32)).to(cutlass.BFloat16)
    rounded.store(low)
    cute.copy(shared_copy, rounded, s_part2)


@cute.kernel
def products_kernel(a: cute.Tensor, b: cute.Tensor, partials: cute.Tensor,
                    tiled_mma: cute.TiledMma, a_layout: cute.ComposedLayout,
                    b_layout: cute.ComposedLayout, global_copy: cute.TiledCopy,
                    shared_copy: cute.TiledCopy):
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    block_m, split, _ = cute.arch.block_idx()
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sa0 = smem.allocate_tensor(cutlass.BFloat16, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
    sa1 = smem.allocate_tensor(cutlass.BFloat16, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
    sa2 = smem.allocate_tensor(cutlass.BFloat16, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
    sb0 = smem.allocate_tensor(cutlass.BFloat16, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)
    sb1 = smem.allocate_tensor(cutlass.BFloat16, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)
    sb2 = smem.allocate_tensor(cutlass.BFloat16, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)
    natural_a = cute.make_layout((BM, BK, STAGES), stride=(BK, 1, BM * BK))
    natural_b = cute.make_layout((BN, BK, STAGES), stride=(BK, 1, BN * BK))
    va0 = cute.make_tensor(sa0.iterator, natural_a)
    va1 = cute.make_tensor(sa1.iterator, natural_a)
    va2 = cute.make_tensor(sa2.iterator, natural_a)
    vb0 = cute.make_tensor(sb0.iterator, natural_b)
    vb1 = cute.make_tensor(sb1.iterator, natural_b)
    vb2 = cute.make_tensor(sb2.iterator, natural_b)
    fa0 = tiled_mma.make_fragment_A(sa0)
    fa1 = tiled_mma.make_fragment_A(sa1)
    fa2 = tiled_mma.make_fragment_A(sa2)
    fb0 = tiled_mma.make_fragment_B(sb0)
    fb1 = tiled_mma.make_fragment_B(sb1)
    fb2 = tiled_mma.make_fragment_B(sb2)

    tmem = utils.TmemAllocator(
        storage.tmem_pointer,
        barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS),
    )
    tmem.allocate(512)
    ab_pipeline = pipeline.PipelineAsyncUmma.create(
        num_stages=STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        barrier_storage=storage.ab_barriers.data_ptr(),
    )
    producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
    consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    accumulator_shape = tiled_mma.partition_shape_C((BM, BN))
    fake_accumulator = tiled_mma.make_fragment_C(cute.append(accumulator_shape, TERMS))
    accumulators = cute.make_tensor(tmem_ptr, fake_accumulator.layout)
    c0 = accumulators[None, None, None, 0]
    c1 = accumulators[None, None, None, 1]
    c2 = accumulators[None, None, None, 2]
    c3 = accumulators[None, None, None, 3]
    c4 = accumulators[None, None, None, 4]
    c5 = accumulators[None, None, None, 5]

    for tile_k in cutlass.range(a.shape[1] // SPLITS // BK):
        ab_pipeline.producer_acquire(producer)
        global_k = split * (a.shape[1] // SPLITS // BK) + tile_k
        ga = cute.local_tile(a, (BM, BK), (block_m, global_k))
        gb = cute.local_tile(b, (BN, BK), (0, global_k))
        convert_three(ga, va0[None, None, producer.index], va1[None, None, producer.index],
                      va2[None, None, producer.index], global_copy, shared_copy, tidx)
        convert_three(gb, vb0[None, None, producer.index], vb1[None, None, producer.index],
                      vb2[None, None, producer.index], global_copy, shared_copy, tidx)
        cute.arch.fence_proxy("async.shared", space="cta")
        ab_pipeline.producer_commit(producer)
        producer.advance()
        if warp == 0:
            ab_pipeline.consumer_wait(consumer)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, tile_k > 0)
            for inner_k in cutlass.range_constexpr(cute.size(fa0, mode=[2])):
                index = (None, None, inner_k, consumer.index)
                cute.gemm(tiled_mma, c0, fa0[index], fb2[index], c0)
                cute.gemm(tiled_mma, c1, fa1[index], fb1[index], c1)
                cute.gemm(tiled_mma, c2, fa2[index], fb0[index], c2)
                cute.gemm(tiled_mma, c3, fa0[index], fb1[index], c3)
                cute.gemm(tiled_mma, c4, fa1[index], fb0[index], c4)
                cute.gemm(tiled_mma, c5, fa0[index], fb0[index], c5)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_pipeline.consumer_release(consumer)
            consumer.advance()

    # Producer-tail waits for the final UMMA release before the TMEM reads.
    ab_pipeline.producer_tail(producer)
    tmem.relinquish_alloc_permit()
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_copy = tcgen05.make_tmem_copy(tmem_atom, c0)
    thread_copy = tmem_copy.get_slice(tidx)
    for term in cutlass.range_constexpr(TERMS):
        gp = cute.local_tile(partials[split, term, None, None], (BM, BN), (block_m, 0))
        mma_gp = tiled_mma.get_slice(0).partition_C(gp)
        source = thread_copy.partition_S(accumulators[None, None, None, term])
        target = thread_copy.partition_D(mma_gp)
        values = cute.make_rmem_tensor(target.shape, cutlass.Float32)
        cute.copy(tmem_copy, source, values)
        cute.autovec_copy(values, target)
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.kernel
def reduce_kernel(partials: cute.Tensor, bias: cute.Tensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * THREADS + tidx
    if index < cute.size(out.shape):
        row = index // out.shape[1]
        column = index % out.shape[1]
        lower = cutlass.Float32(0.0)
        for term in cutlass.range_constexpr(5):
            value = partials[0, term, row, column]
            for split in cutlass.range_constexpr(1, SPLITS):
                value = value + partials[split, term, row, column]
            lower = lower + value
        major = partials[0, 5, row, column]
        for split in cutlass.range_constexpr(1, SPLITS):
            major = major + partials[split, 5, row, column]
        out[row, column] = (major + lower) + bias[column]


@cute.jit
def compiled_entry(a: cute.Tensor, b: cute.Tensor, bias: cute.Tensor,
                   partials: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
    mma = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
        cutlass.BFloat16, cutlass.Float32, (BM, BN, 16), tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
    ))
    a_layout = sm100_utils.make_smem_layout_a(mma, (BM, BN, BK), cutlass.BFloat16, STAGES)
    b_layout = sm100_utils.make_smem_layout_b(mma, (BM, BN, BK), cutlass.BFloat16, STAGES)
    thread_layout = cute.make_layout((8, 16), stride=(16, 1))
    value_layout = cute.make_layout((1, 4))
    global_copy = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128),
        thread_layout, value_layout,
    )
    shared_copy = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64),
        thread_layout, value_layout,
    )
    products_kernel(a, b, partials, mma, a_layout, b_layout, global_copy, shared_copy).launch(
        grid=(a.shape[0] // BM, SPLITS, 1), block=(THREADS, 1, 1), stream=stream,
    )
    reduce_kernel(partials, bias, out).launch(
        grid=(cute.ceil_div(cute.size(out.shape), THREADS), 1, 1),
        block=(THREADS, 1, 1), stream=stream,
    )


_compiled = {}


def launch(a, b, bias, out, partials):
    assert tuple(a.shape) == (8192, 3072) and tuple(b.shape) == (64, 3072)
    assert tuple(bias.shape) == (64,) and tuple(out.shape) == (8192, 64)
    assert tuple(partials.shape) == (4, 6, 8192, 64)
    assert all(t.dtype == torch.float32 and t.is_contiguous() for t in (a, b, bias, out, partials))
    tensors = [from_dlpack(t, assumed_align=16) for t in (a, b, bias, partials, out)]
    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    key = a.device.index
    compiled = _compiled.get(key)
    if compiled is None:
        compiled = cute.compile(compiled_entry, *tensors, stream)
        _compiled[key] = compiled
    compiled(*tensors, stream)
