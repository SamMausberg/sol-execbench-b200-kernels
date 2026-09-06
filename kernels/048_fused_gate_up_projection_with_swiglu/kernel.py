# SPDX-License-Identifier: Apache-2.0
"""Paired BF16 projections with the reference GELU-tanh rounding stages."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _activation(gate, up):
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    cube = (gate * gate) * gate
    inner = 0.7978845608028654 * (gate + 0.044715 * cube)
    activated = (0.5 * gate) * (1.0 + libdevice.tanh(inner))
    activated = activated.to(tl.bfloat16).to(tl.float32)
    return (activated * up).to(tl.bfloat16)


@triton.jit
def _paired(X, Gate, Up, Output, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    group = pid // (GROUP_M * tiles_n)
    first_m = group * GROUP_M
    active_m = tl.minimum(tiles_m - first_m, GROUP_M)
    tile_m = first_m + pid % active_m
    tile_n = (pid % (GROUP_M * tiles_n)) // active_m
    gate_acc = tl.full((BM, BN), 0.0, tl.float32)
    up_acc = tl.full((BM, BN), 0.0, tl.float32)
    for k in range(tl.cdiv(K, BK)):
        x = X.load([tile_m * BM, k * BK])
        gate = Gate.load([tile_n * BN, k * BK])
        up = Up.load([tile_n * BN, k * BK])
        gate_acc = tl.dot(x, tl.trans(gate), gate_acc)
        up_acc = tl.dot(x, tl.trans(up), up_acc)
    Output.store([tile_m * BM, tile_n * BN], _activation(gate_acc, up_acc))


@triton.jit
def _combine(Gate, Up, Output, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(Gate + offset, offset < COUNT, other=0).to(tl.float32)
    up = tl.load(Up + offset, offset < COUNT, other=0).to(tl.float32)
    tl.store(Output + offset, _activation(gate, up), offset < COUNT)


@torch.no_grad()
def library_run(x, gate_proj, up_proj, output):
    gate = torch.matmul(x, gate_proj.t())
    up = torch.matmul(x, up_proj.t())
    _combine[(triton.cdiv(output.numel(), 1024),)](
        gate, up, output, output.numel(), 1024, num_warps=4, enable_fp_fusion=False,
    )


@torch.no_grad()
def configured_run(x, gate_proj, up_proj, output, bm=64, bn=128, bk=64, warps=4, stages=3):
    if not all(value.is_contiguous() for value in (x, gate_proj, up_proj, output)):
        return library_run(x, gate_proj, up_proj, output)
    k = x.shape[-1]
    m = x.numel() // k
    n = gate_proj.shape[0]
    a = TensorDescriptor.from_tensor(x.view(m, k), [bm, bk])
    gate = TensorDescriptor.from_tensor(gate_proj, [bn, bk])
    up = TensorDescriptor.from_tensor(up_proj, [bn, bk])
    out = TensorDescriptor.from_tensor(output.view(m, n), [bm, bn])
    _paired[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a, gate, up, out, m, n, k, bm, bn, bk, 8,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False,
    )


def run(x, gate_proj, up_proj, output):
    configured_run(x, gate_proj, up_proj, output)
