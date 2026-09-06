# SPDX-License-Identifier: Apache-2.0
"""Single-launch Q/K RMSNorm with short contiguous rows and FP32 intermediates."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_grid(
    Q, K, WQ, WK, OQ, OK, EPS: tl.constexpr, ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, STREAM: tl.constexpr = False,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, 128)
    if tl.program_id(1) == 0:
        source, weight, output = Q, WQ, OQ
    else:
        source, weight, output = K, WK, OK
    offsets = row[:, None] * 128 + col[None, :]
    values = tl.load(
        source + offsets, row[:, None] < ROWS, 0,
        cache_modifier=".cg" if STREAM else "",
    )
    scale = tl.rsqrt(tl.sum(values * values, 1) * (1.0 / 128) + EPS)
    weights = tl.load(weight + (row[:, None] % 48) * 128 + col[None, :])
    normalized = values * scale[:, None]
    tl.store(
        output + offsets, normalized * weights, row[:, None] < ROWS,
        cache_modifier=".cs" if STREAM else "",
    )


@triton.jit
def _rmsnorm_dual(
    Q, K, WQ, WK, OQ, OK, EPS: tl.constexpr, ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, STREAM: tl.constexpr = False,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, 128)
    offsets = row[:, None] * 128 + col[None, :]
    weight_offsets = (row[:, None] % 48) * 128 + col[None, :]
    q = tl.load(Q + offsets, row[:, None] < ROWS, 0, cache_modifier=".cg" if STREAM else "")
    k = tl.load(K + offsets, row[:, None] < ROWS, 0, cache_modifier=".cg" if STREAM else "")
    qs = tl.rsqrt(tl.sum(q * q, 1) * (1.0 / 128) + EPS)
    ks = tl.rsqrt(tl.sum(k * k, 1) * (1.0 / 128) + EPS)
    wq = tl.load(WQ + weight_offsets)
    wk = tl.load(WK + weight_offsets)
    qn = q * qs[:, None]
    kn = k * ks[:, None]
    tl.store(OQ + offsets, qn * wq, row[:, None] < ROWS, cache_modifier=".cs" if STREAM else "")
    tl.store(OK + offsets, kn * wk, row[:, None] < ROWS, cache_modifier=".cs" if STREAM else "")


def run(query, key, weight_q, weight_k, eps, query_norm, key_norm):
    rows = query.numel() // 128
    block_rows = 8
    _rmsnorm_grid[(triton.cdiv(rows, block_rows), 2)](
        query, key, weight_q, weight_k, query_norm, key_norm,
        eps, rows, block_rows, True, num_warps=4, enable_fp_fusion=False,
    )
