# SPDX-License-Identifier: Apache-2.0
"""Pad only the reduction axis to permit TMA loads for odd sequence lengths."""
import torch
import triton
import triton.language as tl
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import HardwareInfo
from cute_av import AttentionValueGemm
from cute_kernel import _launch, _launch_reverse

_compiled = {}


@triton.jit
def _pad(A, V, AP, VP, SEQ: tl.constexpr, PAD: tl.constexpr,
         NA: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    a_blocks = triton.cdiv(NA, BLOCK)
    if pid < a_blocks:
        index = pid * BLOCK + local
        column = index % PAD
        row = index // PAD
        value = tl.load(A + row * SEQ + column, (index < NA) & (column < SEQ), 0)
        tl.store(AP + index, value, index < NA)
    else:
        index = (pid - a_blocks) * BLOCK + local
        column = index % 128
        row = (index // 128) % PAD
        head = index // (PAD * 128)
        value = tl.load(V + (head * SEQ + row) * 128 + column,
                        (index < NV) & (row < SEQ), 0)
        tl.store(VP + index, value, index < NV)


@torch.no_grad()
def launch(weights, values, output, bm=128, bn=128, reverse=False):
    batch, heads, sequence, _ = weights.shape
    padded = triton.cdiv(sequence, 8) * 8
    a_storage = torch.empty((batch, heads, sequence, padded), dtype=weights.dtype, device=weights.device)
    v_storage = torch.empty((batch, heads, padded, 128), dtype=values.dtype, device=values.device)
    a = from_dlpack(a_storage.permute(2, 3, 1, 0), assumed_align=16)
    b = from_dlpack(v_storage.permute(3, 2, 1, 0), assumed_align=16)
    c = from_dlpack(output.view(batch, sequence, 40, 128).permute(1, 3, 2, 0), assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream(weights.device).cuda_stream)
    key = (weights.device.index, batch, sequence, padded, bm, bn, reverse)
    compiled = _compiled.get(key)
    if compiled is None:
        gemm = AttentionValueGemm(cutlass.Float32, False, (bm, bn), (1, 1), True)
        clusters = HardwareInfo().get_max_active_clusters(1)
        compiled = cute.compile(_launch_reverse if reverse else _launch, gemm, a, b, c, clusters, stream)
        _compiled[key] = compiled
    # Build fresh TMA execution arguments before the first GPU launch. The
    # tensors and adapters remain alive until both operations are enqueued.
    execution_args, adapters = compiled.generate_execution_args(a, b, c, stream)
    count_a, count_v = a_storage.numel(), v_storage.numel()
    _pad[(triton.cdiv(count_a, 2048) + triton.cdiv(count_v, 2048),)](
        weights, values, a_storage, v_storage, sequence, padded, count_a, count_v, 2048, num_warps=4)
    compiled.run_compiled_program(execution_args)


@torch.no_grad()
def run(weights, values):
    batch, _, sequence, _ = weights.shape
    output = torch.empty((batch, sequence, 5120), dtype=weights.dtype, device=weights.device)
    launch(weights, values, output)
    return output
