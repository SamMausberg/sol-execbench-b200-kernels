# SPDX-License-Identifier: Apache-2.0
"""Grouped expert backward with compact routing and direct gradient stores."""

import torch
import triton
import triton.language as tl

FORWARD_PRECISION = "tf32x3"
BACKWARD_PRECISION = "tf32x3"
PROJECT_TILE = (32, 64, 64, 3)
WEIGHT_TILE = (64, 128, 32, 2)
INPUT_TILE = (32, 64, 64, 3)
GEMM_ARCH = "sm100"


@triton.jit
def _dot(a, b, acc, PRECISION: tl.constexpr):
    if PRECISION == "bf16":
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
    elif PRECISION == "bf16x3":
        ah = a.to(tl.bfloat16)
        bh = b.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(al, bh, acc)
        acc = tl.dot(ah, bl, acc)
        return tl.dot(ah, bh, acc)
    else:
        return tl.dot(a, b, acc, input_precision=PRECISION)


@triton.jit
def _assign_routes(EXPERT, COUNTS, LOCAL, ROUTES: tl.constexpr,
                   BLOCK: tl.constexpr):
    route = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(EXPERT + route, route < ROUTES, 0).to(tl.int32)
    local = tl.atomic_add(COUNTS + expert, 1, route < ROUTES, sem="relaxed")
    tl.store(LOCAL + route, local, route < ROUTES)


@triton.jit
def _prefix_counts(COUNTS, OFFSETS, EXPERTS: tl.constexpr,
                    BLOCK: tl.constexpr):
    expert = tl.arange(0, BLOCK)
    count = tl.load(COUNTS + expert, expert < EXPERTS, 0)
    end = tl.cumsum(count)
    tl.store(OFFSETS + expert, end - count, expert < EXPERTS)
    tl.store(OFFSETS + EXPERTS, tl.sum(count, 0))


@triton.jit
def _compact_routes(EXPERT, LOCAL, OFFSETS, MAP, INVERSE,
                     ROUTES: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(EXPERT + route, route < ROUTES, 0).to(tl.int32)
    local = tl.load(LOCAL + route, route < ROUTES, 0)
    start = tl.load(OFFSETS + expert)
    compact = start + local
    tl.store(MAP + compact, route, route < ROUTES)
    tl.store(INVERSE + route, compact, route < ROUTES)


@triton.jit
def _grouped_project(
    X, W, MAP, COUNTS, OFFSETS, OUT,
    K: tl.constexpr, N: tl.constexpr, TOP: tl.constexpr,
    TRANSPOSE_WEIGHT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    expert = tl.program_id(1)
    count = tl.load(COUNTS + expert)
    start = tl.load(OFFSETS + expert)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    for tile in range(tl.cdiv(count, BM)):
        row = tile * BM + tl.arange(0, BM)
        route = tl.load(MAP + start + row, row < count, 0)
        token = route // TOP
        acc = tl.full((BM, BN), 0, tl.float32)
        for block in range(triton.cdiv(K, BK)):
            col = block * BK + k
            a = tl.load(
                X + token[:, None] * K + col[None, :],
                (row[:, None] < count) & (col[None, :] < K), 0,
            )
            if TRANSPOSE_WEIGHT:
                offset_w = n[None, :] * K + col[:, None]
            else:
                offset_w = col[:, None] * N + n[None, :]
            b = tl.load(
                W + expert * K * N + offset_w,
                (col[:, None] < K) & (n[None, :] < N), 0,
            )
            acc = _dot(a, b, acc, PRECISION)
        tl.store(
            OUT + (start + row[:, None]) * N + n[None, :], acc,
            (row[:, None] < count) & (n[None, :] < N),
        )


@triton.jit
def _activation_backward(
    GATE, UP, BASE_GRAD, ROUTE_WEIGHT, MAP,
    INTERMEDIATE, DGATE, DUP, DWEIGHT,
    I: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    gate = tl.load(GATE + row * I + col, col < I, 0)
    up = tl.load(UP + row * I + col, col < I, 0)
    base_grad = tl.load(BASE_GRAD + row * I + col, col < I, 0)
    route = tl.load(MAP + row)
    weight = tl.load(ROUTE_WEIGHT + route)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    gate_output = gate * sigmoid
    intermediate = gate_output * up
    grad_intermediate = base_grad * weight
    grad_gate_output = grad_intermediate * up
    silu_derivative = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    grad_gate = grad_gate_output * silu_derivative
    grad_up = grad_intermediate * gate_output
    tl.store(INTERMEDIATE + row * I + col, intermediate, col < I)
    tl.store(DGATE + row * I + col, grad_gate, col < I)
    tl.store(DUP + row * I + col, grad_up, col < I)
    tl.store(DWEIGHT + route, tl.sum(base_grad * intermediate, 0))


@triton.jit
def _weight_gradient(
    LEFT, RIGHT, ROUTE_WEIGHT, MAP, COUNTS, OFFSETS, OUT,
    H: tl.constexpr, I: tl.constexpr, TOP: tl.constexpr,
    DOWN: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    expert = tl.program_id(2)
    count = tl.load(COUNTS + expert)
    start = tl.load(OFFSETS + expert)
    M: tl.constexpr = H if DOWN else I
    N: tl.constexpr = I if DOWN else H
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tile in range(tl.cdiv(count, BK)):
        row = tile * BK + kk
        route = tl.load(MAP + start + row, row < count, 0)
        token = route // TOP
        if DOWN:
            a = tl.load(
                LEFT + token[None, :] * H + m[:, None],
                (row[None, :] < count) & (m[:, None] < H), 0,
            )
            weight = tl.load(ROUTE_WEIGHT + route, row < count, 0)
            a = a * weight[None, :]
            b = tl.load(
                RIGHT + (start + row[:, None]) * I + n[None, :],
                (row[:, None] < count) & (n[None, :] < I), 0,
            )
        else:
            a = tl.load(
                LEFT + (start + row[None, :]) * I + m[:, None],
                (row[None, :] < count) & (m[:, None] < I), 0,
            )
            b = tl.load(
                RIGHT + token[:, None] * H + n[None, :],
                (row[:, None] < count) & (n[None, :] < H), 0,
            )
        acc = _dot(a, b, acc, PRECISION)
    tl.store(
        OUT + expert.to(tl.int64) * M * N + m[:, None] * N + n[None, :],
        acc, (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _input_gradient(
    DGATE, DUP, WGATE, WUP, COUNTS, OFFSETS, OUT,
    H: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    expert = tl.program_id(1)
    count = tl.load(COUNTS + expert)
    start = tl.load(OFFSETS + expert)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    for tile in range(tl.cdiv(count, BM)):
        row = tile * BM + tl.arange(0, BM)
        acc_gate = tl.full((BM, BN), 0, tl.float32)
        acc_up = tl.full((BM, BN), 0, tl.float32)
        for block in range(triton.cdiv(I, BK)):
            col = block * BK + kk
            offsets_a = (start + row[:, None]) * I + col[None, :]
            offsets_b = expert.to(tl.int64) * I * H + col[:, None] * H + n[None, :]
            mask_a = (row[:, None] < count) & (col[None, :] < I)
            mask_b = (col[:, None] < I) & (n[None, :] < H)
            ag = tl.load(DGATE + offsets_a, mask_a, 0)
            au = tl.load(DUP + offsets_a, mask_a, 0)
            bg = tl.load(WGATE + offsets_b, mask_b, 0)
            bu = tl.load(WUP + offsets_b, mask_b, 0)
            acc_gate = _dot(ag, bg, acc_gate, PRECISION)
            acc_up = _dot(au, bu, acc_up, PRECISION)
        tl.store(
            OUT + (start + row[:, None]) * H + n[None, :],
            acc_gate + acc_up,
            (row[:, None] < count) & (n[None, :] < H),
        )


@triton.jit
def _gather_hidden(EXPERT, INVERSE, ROUTE_GRAD, OUT,
                    H: tl.constexpr, TOP: tl.constexpr,
                    BLOCK: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ranks = tl.arange(0, TOP)
    experts = tl.load(EXPERT + token * TOP + ranks)
    slots = tl.load(INVERSE + token * TOP + ranks)
    ordered = tl.sort((experts.to(tl.int64) << 32) | slots.to(tl.int64))
    slots = ordered.to(tl.int32)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for rank in tl.static_range(TOP):
        slot = tl.sum(tl.where(ranks == rank, slots, 0), 0)
        value = tl.load(ROUTE_GRAD + slot * H + col, col < H, 0)
        acc = acc + value
    tl.store(OUT + token * H + col, acc, col < H)


@torch.no_grad()
def run(
    grad_output, hidden_states, topk_indices, topk_weights,
    gate_weights, up_weights, down_weights,
    grad_hidden_states, grad_topk_weights,
    grad_gate_weights, grad_up_weights, grad_down_weights,
):
    tokens, hidden = hidden_states.shape
    experts, intermediate, _ = gate_weights.shape
    top = topk_indices.shape[1]
    routes = tokens * top
    device = hidden_states.device
    counts = torch.zeros(experts, dtype=torch.int32, device=device)
    local = torch.empty(routes, dtype=torch.int32, device=device)
    offsets = torch.empty(experts + 1, dtype=torch.int32, device=device)
    route_map = torch.empty(routes, dtype=torch.int32, device=device)
    inverse = torch.empty_like(route_map)
    buffers = [torch.empty((routes, intermediate), dtype=torch.float32, device=device) for _ in range(6)]
    gate, up, base_grad, activated, dgate, dup = buffers
    route_grad = torch.empty((routes, hidden), dtype=torch.float32, device=device)
    _assign_routes[(triton.cdiv(routes, 256),)](topk_indices, counts, local, routes, 256)
    _prefix_counts[(1,)](counts, offsets, experts, triton.next_power_of_2(experts))
    _compact_routes[(triton.cdiv(routes, 256),)](
        topk_indices, local, offsets, route_map, inverse, routes, 256,
    )
    pm, pn, pk, ps = PROJECT_TILE
    wm, wn, wk, ws = WEIGHT_TILE
    im, inn, ik, ins = INPUT_TILE
    for weights, output in ((gate_weights, gate), (up_weights, up)):
        _grouped_project[(triton.cdiv(intermediate, pn), experts)](
            hidden_states, weights, route_map, counts, offsets, output,
            hidden, intermediate, top, True, pm, pn, pk, FORWARD_PRECISION,
            num_warps=4, num_stages=ps, arch=GEMM_ARCH,
        )
    _grouped_project[(triton.cdiv(intermediate, pn), experts)](
        grad_output, down_weights, route_map, counts, offsets, base_grad,
        hidden, intermediate, top, False, pm, pn, pk, FORWARD_PRECISION,
        num_warps=4, num_stages=ps, arch=GEMM_ARCH,
    )
    _activation_backward[(routes,)](
        gate, up, base_grad, topk_weights, route_map,
        activated, dgate, dup, grad_topk_weights,
        intermediate, triton.next_power_of_2(intermediate),
        num_warps=4, enable_fp_fusion=False,
    )
    for left, output in ((dgate, grad_gate_weights), (dup, grad_up_weights)):
        _weight_gradient[(triton.cdiv(intermediate, wm), triton.cdiv(hidden, wn), experts)](
            left, hidden_states, topk_weights, route_map, counts, offsets, output,
            hidden, intermediate, top, False, wm, wn, wk, BACKWARD_PRECISION,
            num_warps=4, num_stages=ws, enable_fp_fusion=False, arch=GEMM_ARCH,
        )
    _weight_gradient[(triton.cdiv(hidden, wm), triton.cdiv(intermediate, wn), experts)](
        grad_output, activated, topk_weights, route_map, counts, offsets, grad_down_weights,
        hidden, intermediate, top, True, wm, wn, wk, BACKWARD_PRECISION,
        num_warps=4, num_stages=ws, enable_fp_fusion=False, arch=GEMM_ARCH,
    )
    _input_gradient[(triton.cdiv(hidden, inn), experts)](
        dgate, dup, gate_weights, up_weights, counts, offsets, route_grad,
        hidden, intermediate, im, inn, ik, BACKWARD_PRECISION,
        num_warps=4, num_stages=ins, arch=GEMM_ARCH,
    )
    _gather_hidden[(tokens, triton.cdiv(hidden, 1024))](
        topk_indices, inverse, route_grad, grad_hidden_states,
        hidden, top, 1024, num_warps=4,
    )
