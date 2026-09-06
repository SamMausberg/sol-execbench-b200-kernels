# SPDX-License-Identifier: Apache-2.0
"""Experimental padded expert rows with tensor-map GEMM loads."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from kernel import _assign_routes, _compact_routes, _dot, _gather_hidden

FORWARD_PRECISION = "tf32x3"
BACKWARD_PRECISION = "tf32"
PROJECT_TILE = (64, 128, 64, 3)
WEIGHT_TILE = (128, 256, 32, 1)
INPUT_TILE = (64, 128, 32, 1)


@triton.jit
def _padded_prefix(COUNTS, OFFSETS, E: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.arange(0, BLOCK)
    count = tl.load(COUNTS + expert, expert < E, 0)
    padded = tl.cdiv(count, 32) * 32
    end = tl.cumsum(padded)
    tl.store(OFFSETS + expert, end - padded, expert < E)
    tl.store(OFFSETS + E, tl.sum(padded, 0))


@triton.jit
def _prepare(X, G, WEIGHT, MAP, XR, GR, GW,
             H: tl.constexpr, TOP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    route = tl.load(MAP + row)
    token = route // TOP
    x = tl.load(X + token * H + col, (route >= 0) & (col < H), 0)
    g = tl.load(G + token * H + col, (route >= 0) & (col < H), 0)
    weight = tl.load(WEIGHT + route, route >= 0, 0)
    tl.store(XR + row * H + col, x, col < H)
    tl.store(GR + row * H + col, g, col < H)
    tl.store(GW + row * H + col, g * weight, col < H)


@triton.jit
def _project_tma(X, W, COUNTS, OFFSETS, OUT,
                 K: tl.constexpr, N: tl.constexpr, TRANSPOSE: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 PRECISION: tl.constexpr):
    expert = tl.program_id(1)
    count = tl.load(COUNTS + expert)
    start = tl.load(OFFSETS + expert)
    n = tl.program_id(0) * BN
    for tile in range(tl.cdiv(count, BM)):
        row = tile * BM
        acc = tl.full((BM, BN), 0, tl.float32)
        for block in range(triton.cdiv(K, BK)):
            col = block * BK
            a = X.load([start + row, col])
            if TRANSPOSE:
                b = W.load([expert * N + n, col]).T
            else:
                b = W.load([expert * K + col, n])
            acc = _dot(a, b, acc, PRECISION)
        m = row + tl.arange(0, BM)
        nn = n + tl.arange(0, BN)
        tl.store(OUT + (start + m[:, None]) * N + nn[None, :], acc,
                 (m[:, None] < count) & (nn[None, :] < N))


@triton.jit
def _activation(GATE, UP, BASE, WEIGHT, MAP, INTER, DGATE, DUP, DW,
                I: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    route = tl.load(MAP + row)
    valid = (route >= 0) & (col < I)
    gate = tl.load(GATE + row * I + col, valid, 0)
    up = tl.load(UP + row * I + col, valid, 0)
    base = tl.load(BASE + row * I + col, valid, 0)
    weight = tl.load(WEIGHT + route, route >= 0, 0)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    gate_out = gate * sigmoid
    inter = gate_out * up
    grad_inter = base * weight
    dgate = (grad_inter * up) * (sigmoid * (1.0 + gate * (1.0 - sigmoid)))
    dup = grad_inter * gate_out
    tl.store(INTER + row * I + col, inter, col < I)
    tl.store(DGATE + row * I + col, dgate, col < I)
    tl.store(DUP + row * I + col, dup, col < I)
    tl.store(DW + route, tl.sum(base * inter, 0), route >= 0)


@triton.jit
def _weight_tma(A, B, O, COUNTS, OFFSETS,
                M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                PRECISION: tl.constexpr):
    expert = tl.program_id(2)
    count = tl.load(COUNTS + expert)
    start = tl.load(OFFSETS + expert)
    m = tl.program_id(0) * BM
    n = tl.program_id(1) * BN
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(count, BK)):
        row = start + block * BK
        a = A.load([row, m]).T
        b = B.load([row, n])
        acc = _dot(a, b, acc, PRECISION)
    O.store([expert * M + m, n], acc)


@triton.jit
def _input_tma(DG, DU, WG, WU, COUNTS, OFFSETS, OUT,
               H: tl.constexpr, I: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               PRECISION: tl.constexpr):
    expert = tl.program_id(1)
    count = tl.load(COUNTS + expert)
    start = tl.load(OFFSETS + expert)
    n = tl.program_id(0) * BN
    for tile in range(tl.cdiv(count, BM)):
        row = tile * BM
        accg = tl.full((BM, BN), 0, tl.float32)
        accu = tl.full((BM, BN), 0, tl.float32)
        for block in range(triton.cdiv(I, BK)):
            k = block * BK
            ag = DG.load([start + row, k])
            au = DU.load([start + row, k])
            bg = WG.load([expert * I + k, n])
            bu = WU.load([expert * I + k, n])
            accg = _dot(ag, bg, accg, PRECISION)
            accu = _dot(au, bu, accu, PRECISION)
        m = row + tl.arange(0, BM)
        nn = n + tl.arange(0, BN)
        tl.store(OUT + (start + m[:, None]) * H + nn[None, :], accg + accu,
                 (m[:, None] < count) & (nn[None, :] < H))


def desc(tensor, rows, cols, block):
    return TensorDescriptor(tensor, [rows, cols], [cols, 1], list(block))


@torch.no_grad()
def run(grad_output, hidden_states, topk_indices, topk_weights,
        gate_weights, up_weights, down_weights,
        grad_hidden_states, grad_topk_weights,
        grad_gate_weights, grad_up_weights, grad_down_weights):
    tokens, hidden = hidden_states.shape
    experts, intermediate, _ = gate_weights.shape
    top = topk_indices.shape[1]
    routes = tokens * top
    capacity = routes + experts * 31
    device = hidden_states.device
    counts = torch.zeros(experts, dtype=torch.int32, device=device)
    local = torch.empty(routes, dtype=torch.int32, device=device)
    offsets = torch.empty(experts + 1, dtype=torch.int32, device=device)
    route_map = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    inverse = torch.empty(routes, dtype=torch.int32, device=device)
    gate, up, base, activated, dgate, dup = [
        torch.empty((capacity, intermediate), device=device) for _ in range(6)
    ]
    xr, gr, gw, route_grad = [
        torch.empty((capacity, hidden), device=device) for _ in range(4)
    ]
    _assign_routes[(triton.cdiv(routes, 256),)](topk_indices, counts, local, routes, 256)
    _padded_prefix[(1,)](counts, offsets, experts, triton.next_power_of_2(experts))
    _compact_routes[(triton.cdiv(routes, 256),)](
        topk_indices, local, offsets, route_map, inverse, routes, 256,
    )
    _prepare[(capacity, triton.cdiv(hidden, 1024))](
        hidden_states, grad_output, topk_weights, route_map, xr, gr, gw,
        hidden, top, 1024, num_warps=4,
    )
    pm, pn, pk, ps = PROJECT_TILE
    xdesc = desc(xr, capacity, hidden, (pm, pk))
    gdesc = desc(gr, capacity, hidden, (pm, pk))
    for weight, output in ((gate_weights, gate), (up_weights, up)):
        wdesc = desc(weight, experts * intermediate, hidden, (pn, pk))
        _project_tma[(triton.cdiv(intermediate, pn), experts)](
            xdesc, wdesc, counts, offsets, output, hidden, intermediate, True,
            pm, pn, pk, FORWARD_PRECISION, num_warps=4, num_stages=ps,
        )
    wdesc = desc(down_weights, experts * hidden, intermediate, (pk, pn))
    _project_tma[(triton.cdiv(intermediate, pn), experts)](
        gdesc, wdesc, counts, offsets, base, hidden, intermediate, False,
        pm, pn, pk, FORWARD_PRECISION, num_warps=4, num_stages=ps,
    )
    _activation[(capacity,)](
        gate, up, base, topk_weights, route_map, activated, dgate, dup, grad_topk_weights,
        intermediate, triton.next_power_of_2(intermediate),
        num_warps=4, enable_fp_fusion=False,
    )
    wm, wn, wk, ws = WEIGHT_TILE
    for left, right, output, m, n in (
        (dgate, xr, grad_gate_weights, intermediate, hidden),
        (dup, xr, grad_up_weights, intermediate, hidden),
        (gw, activated, grad_down_weights, hidden, intermediate),
    ):
        _weight_tma[(triton.cdiv(m, wm), triton.cdiv(n, wn), experts)](
            desc(left, capacity, m, (wk, wm)), desc(right, capacity, n, (wk, wn)),
            desc(output, experts * m, n, (wm, wn)), counts, offsets, m, n,
            wm, wn, wk, BACKWARD_PRECISION, num_warps=4, num_stages=ws,
        )
    im, inn, ik, ins = INPUT_TILE
    _input_tma[(triton.cdiv(hidden, inn), experts)](
        desc(dgate, capacity, intermediate, (im, ik)),
        desc(dup, capacity, intermediate, (im, ik)),
        desc(gate_weights, experts * intermediate, hidden, (ik, inn)),
        desc(up_weights, experts * intermediate, hidden, (ik, inn)),
        counts, offsets, route_grad, hidden, intermediate,
        im, inn, ik, BACKWARD_PRECISION, num_warps=4, num_stages=ins,
    )
    _gather_hidden[(tokens, triton.cdiv(hidden, 1024))](
        topk_indices, inverse, route_grad, grad_hidden_states,
        hidden, top, 1024, num_warps=4,
    )
