# SPDX-License-Identifier: Apache-2.0
"""One grouped projection launch returning the reference's concrete head views."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _plain(X, QW, KW, VW, Q, K, V,
           M: tl.constexpr, H: tl.constexpr, QN: tl.constexpr, KN: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           GROUP: tl.constexpr):
    tile = tl.program_id(0)
    mt: tl.constexpr = triton.cdiv(M, BM)
    nt: tl.constexpr = (QN + 2 * KN) // BN
    group = tile // (GROUP * nt)
    first_m = group * GROUP
    group_size = tl.minimum(mt - first_m, GROUP)
    tm = first_m + tile % group_size
    tn = (tile % (GROUP * nt)) // group_size
    col = tn * BN
    if col < QN:
        weight, output, width = QW, Q, QN
    elif col < QN + KN:
        weight, output, width = KW, K, KN
        col -= QN
    else:
        weight, output, width = VW, V, KN
        col -= QN + KN
    rows = tm * BM + tl.arange(0, BM)
    columns = col + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(triton.cdiv(H, BK)):
        depth = block * BK + inner
        x = tl.load(X + rows[:, None] * H + depth[None, :],
                    (rows[:, None] < M) & (depth[None, :] < H), 0)
        w = tl.load(weight + columns[None, :] * H + depth[:, None],
                    (columns[None, :] < width) & (depth[:, None] < H), 0)
        acc = tl.dot(x, w, acc)
    tl.store(output + rows[:, None] * width + columns[None, :], acc,
             (rows[:, None] < M) & (columns[None, :] < width))


@triton.jit
def _tma_tile(tile, X, QW, KW, VW, Q, K, V,
              M: tl.constexpr, H: tl.constexpr, QN: tl.constexpr, KN: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              GROUP: tl.constexpr):
    mt: tl.constexpr = triton.cdiv(M, BM)
    nt: tl.constexpr = (QN + 2 * KN) // BN
    group = tile // (GROUP * nt)
    first_m = group * GROUP
    group_size = tl.minimum(mt - first_m, GROUP)
    tm = first_m + tile % group_size
    tn = (tile % (GROUP * nt)) // group_size
    col = tn * BN
    if col < QN:
        weight, output = QW, Q
    elif col < QN + KN:
        weight, output = KW, K
        col -= QN
    else:
        weight, output = VW, V
        col -= QN + KN
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(triton.cdiv(H, BK)):
        x = X.load([tm * BM, block * BK])
        w = weight.load([col, block * BK])
        acc = tl.dot(x, w.T, acc)
    output.store([tm * BM, col], acc.to(tl.bfloat16))


@triton.jit
def _tma(X, QW, KW, VW, Q, K, V,
         M: tl.constexpr, H: tl.constexpr, QN: tl.constexpr, KN: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         GROUP: tl.constexpr, PROGRAMS: tl.constexpr, PERSISTENT: tl.constexpr):
    if PERSISTENT:
        for tile in range(tl.program_id(0), triton.cdiv(M, BM) * ((QN + 2 * KN) // BN), PROGRAMS):
            _tma_tile(tile, X, QW, KW, VW, Q, K, V, M, H, QN, KN, BM, BN, BK, GROUP)
    else:
        _tma_tile(tl.program_id(0), X, QW, KW, VW, Q, K, V, M, H, QN, KN, BM, BN, BK, GROUP)


def launch(x, qw, kw, vw, q, k, v, *, mode="tma", bm=128, bn=128,
           bk=64, warps=4, stages=3, group=4, persistent=False):
    m, hidden = x.shape
    qn, kn = qw.shape[0], kw.shape[0]
    programs = triton.cdiv(m, bm) * ((qn + 2 * kn) // bn)
    if mode == "plain":
        _plain[(programs,)](x, qw, kw, vw, q, k, v, m, hidden, qn, kn,
                            bm, bn, bk, group, num_warps=warps, num_stages=stages)
        return
    if persistent:
        programs = min(programs, 148)
    xd = TensorDescriptor.from_tensor(x, [bm, bk])
    qwd = TensorDescriptor.from_tensor(qw, [bn, bk])
    kwd = TensorDescriptor.from_tensor(kw, [bn, bk])
    vwd = TensorDescriptor.from_tensor(vw, [bn, bk])
    qd = TensorDescriptor.from_tensor(q, [bm, bn])
    kd = TensorDescriptor.from_tensor(k, [bm, bn])
    vd = TensorDescriptor.from_tensor(v, [bm, bn])
    _tma[(programs,)](xd, qwd, kwd, vwd, qd, kd, vd, m, hidden, qn, kn,
                      bm, bn, bk, group, programs, persistent,
                      num_warps=warps, num_stages=stages)


@torch.no_grad()
def run(hidden_states, q_weight, k_weight, v_weight):
    batch, sequence, hidden = hidden_states.shape
    rows = batch * sequence
    x = hidden_states.reshape(rows, hidden)
    q = torch.empty((rows, q_weight.shape[0]), dtype=x.dtype, device=x.device)
    k = torch.empty((rows, k_weight.shape[0]), dtype=x.dtype, device=x.device)
    v = torch.empty((rows, v_weight.shape[0]), dtype=x.dtype, device=x.device)
    if rows <= 512:
        launch(x, q_weight, k_weight, v_weight, q, k, v,
               mode="plain", bm=32, bn=64, bk=128)
    else:
        launch(x, q_weight, k_weight, v_weight, q, k, v)
    return (q.view(batch, sequence, 16, 128).transpose(1, 2),
            k.view(batch, sequence, 4, 128).transpose(1, 2),
            v.view(batch, sequence, 4, 128).transpose(1, 2))
