"""Grouped Q/K/V projection with the reference's BF16 bias boundary."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _qkv(
    X, QW, QB, KW, KB, VW, VB, Q, K, V,
    M: tl.constexpr, S: tl.constexpr,
    X0: tl.constexpr, X1: tl.constexpr,
    QW0: tl.constexpr, QW1: tl.constexpr, QB0: tl.constexpr,
    KW0: tl.constexpr, KW1: tl.constexpr, KB0: tl.constexpr,
    VW0: tl.constexpr, VW1: tl.constexpr, VB0: tl.constexpr,
    Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr, Q3: tl.constexpr,
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
    V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr, V3: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    tile = tl.program_id(0)
    m_tiles: tl.constexpr = tl.cdiv(M, BM)
    n_tiles: tl.constexpr = 1536 // BN
    group = tile // (GROUP_M * n_tiles)
    first_m = group * GROUP_M
    group_size = tl.minimum(m_tiles - first_m, GROUP_M)
    tile_m = first_m + tile % group_size
    tile_n = (tile % (GROUP_M * n_tiles)) // group_size
    n_start = tile_n * BN
    if n_start < 1024:
        weight, bias, output = QW, QB, Q
        w0, w1, bs = QW0, QW1, QB0
        y0, y1, y2, y3 = Q0, Q1, Q2, Q3
        width = 1024
        col_start = n_start
    elif n_start < 1280:
        weight, bias, output = KW, KB, K
        w0, w1, bs = KW0, KW1, KB0
        y0, y1, y2, y3 = K0, K1, K2, K3
        width = 256
        col_start = n_start - 1024
    else:
        weight, bias, output = VW, VB, V
        w0, w1, bs = VW0, VW1, VB0
        y0, y1, y2, y3 = V0, V1, V2, V3
        width = 256
        col_start = n_start - 1280
    rows = tile_m * BM + tl.arange(0, BM)
    columns = col_start + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulated = tl.full((BM, BN), 0.0, tl.float32)
    for start in range(tl.cdiv(640, BK)):
        depth = start * BK + inner
        x = tl.load(X + rows[:, None] * X0 + depth[None, :] * X1,
                    (rows[:, None] < M) & (depth[None, :] < 640), other=0.0)
        w = tl.load(weight + columns[None, :] * w0 + depth[:, None] * w1,
                    (columns[None, :] < width) & (depth[:, None] < 640), other=0.0)
        accumulated = tl.dot(x, w, accumulated)
    # torch.matmul returns BF16 before the separate bias addition.
    projected = accumulated.to(tl.bfloat16).to(tl.float32)
    offset = tl.load(bias + columns * bs, columns < width, other=0.0).to(tl.float32)
    result = projected + offset[None, :]
    output_rows = (rows // S) * y0 + (rows % S) * y1
    output_columns = (columns // 256) * y2 + (columns % 256) * y3
    tl.store(output + output_rows[:, None] + output_columns[None, :], result,
             (rows[:, None] < M) & (columns[None, :] < width))


def launch(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
           query_states, key_states, value_states,
           block_m=32, block_n=64, block_k=128, warps=4, stages=3, group_m=4):
    batch, sequence, _ = hidden_states.shape
    rows = batch * sequence
    matrix = hidden_states.reshape(rows, 640)
    _qkv[(triton.cdiv(rows, block_m) * (1536 // block_n),)](
        matrix, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
        query_states, key_states, value_states, rows, sequence,
        *matrix.stride(), *q_weight.stride(), q_bias.stride(0),
        *k_weight.stride(), k_bias.stride(0), *v_weight.stride(), v_bias.stride(0),
        *query_states.stride(), *key_states.stride(), *value_states.stride(),
        block_m, block_n, block_k, group_m,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False,
    )


@triton.jit
def _qkv_tma_tile(
    tile, X, QW, KW, VW, QB, KB, VB, Q, K, V,
    M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, WARP_SPECIALIZE: tl.constexpr,
):
    m_tiles: tl.constexpr = tl.cdiv(M, BM)
    n_tiles: tl.constexpr = 1536 // BN
    group = tile // (GROUP_M * n_tiles)
    first_m = group * GROUP_M
    group_size = tl.minimum(m_tiles - first_m, GROUP_M)
    tile_m = first_m + tile % group_size
    tile_n = (tile % (GROUP_M * n_tiles)) // group_size
    n_start = tile_n * BN
    if n_start < 1024:
        weight, bias, output = QW, QB, Q
        width = 1024
        col_start = n_start
    elif n_start < 1280:
        weight, bias, output = KW, KB, K
        width = 256
        col_start = n_start - 1024
    else:
        weight, bias, output = VW, VB, V
        width = 256
        col_start = n_start - 1280
    accumulated = tl.full((BM, BN), 0.0, tl.float32)
    for start in tl.range(tl.cdiv(640, BK), warp_specialize=WARP_SPECIALIZE):
        x = X.load([tile_m * BM, start * BK])
        w = weight.load([col_start, start * BK])
        accumulated = tl.dot(x, tl.trans(w), accumulated)
    columns = col_start + tl.arange(0, BN)
    projected = accumulated.to(tl.bfloat16).to(tl.float32)
    offset = tl.load(bias + columns).to(tl.float32)
    output.store([tile_m * BM, col_start], (projected + offset[None, :]).to(tl.bfloat16))


@triton.jit
def _qkv_tma(
    X, QW, KW, VW, QB, KB, VB, Q, K, V,
    M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, PERSISTENT: tl.constexpr, PROGRAMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    if PERSISTENT:
        for tile in range(tl.program_id(0), tl.cdiv(M, BM) * (1536 // BN), PROGRAMS):
            _qkv_tma_tile(tile, X, QW, KW, VW, QB, KB, VB, Q, K, V,
                          M, BM, BN, BK, GROUP_M, WARP_SPECIALIZE)
    else:
        _qkv_tma_tile(tl.program_id(0), X, QW, KW, VW, QB, KB, VB, Q, K, V,
                      M, BM, BN, BK, GROUP_M, WARP_SPECIALIZE)


def launch_tma(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
               query_states, key_states, value_states,
               block_m=128, block_n=128, block_k=64, warps=4, stages=3, group_m=4,
               persistent=False, ctas=1, warp_specialize=False):
    matrix = hidden_states.reshape(-1, 640)
    rows = matrix.shape[0]
    a = TensorDescriptor.from_tensor(matrix, [block_m, block_k])
    qw = TensorDescriptor.from_tensor(q_weight, [block_n, block_k])
    kw = TensorDescriptor.from_tensor(k_weight, [block_n, block_k])
    vw = TensorDescriptor.from_tensor(v_weight, [block_n, block_k])
    qo = TensorDescriptor.from_tensor(query_states.view(rows, 1024), [block_m, block_n])
    ko = TensorDescriptor.from_tensor(key_states.view(rows, 256), [block_m, block_n])
    vo = TensorDescriptor.from_tensor(value_states.view(rows, 256), [block_m, block_n])
    programs = triton.cdiv(rows, block_m) * (1536 // block_n)
    if persistent:
        programs = min(programs, torch.cuda.get_device_properties(matrix.device).multi_processor_count // ctas)
    _qkv_tma[(programs,)](
        a, qw, kw, vw, q_bias, k_bias, v_bias, qo, ko, vo,
        rows, block_m, block_n, block_k, group_m, persistent, programs, warp_specialize,
        num_warps=warps, num_stages=stages, num_ctas=ctas, enable_fp_fusion=False,
    )


@torch.no_grad()
def run(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
        query_states, key_states, value_states):
    launch(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
           query_states, key_states, value_states)
