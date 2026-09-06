"""Native grouped GEMM with freshly packed pointers and a BF16 bias pass."""

import hashlib
from pathlib import Path
import sysconfig

import torch
from torch.utils.cpp_extension import load
import triton
import triton.language as tl


def load_bridge():
    source = Path(__file__).with_suffix(".cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cuda = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
    return load(
        name=f"sol_qkv_grouped_{digest[:16]}", sources=[str(source)],
        extra_include_paths=[str(cuda / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[f"-L{cuda / 'lib'}", "-l:libcublas.so.13", "-l:libcudart.so.13"],
        with_cuda=False, verbose=False,
    )


_bridge = None


@triton.jit
def _pack_pointers(X, QW, KW, VW, Q, K, V, P):
    index = tl.arange(0, 16)
    pointer = X.to(tl.uint64)
    pointer = tl.where(index == 0, QW.to(tl.uint64), pointer)
    pointer = tl.where(index == 1, KW.to(tl.uint64), pointer)
    pointer = tl.where(index == 2, VW.to(tl.uint64), pointer)
    pointer = tl.where(index == 6, Q.to(tl.uint64), pointer)
    pointer = tl.where(index == 7, K.to(tl.uint64), pointer)
    pointer = tl.where(index == 8, V.to(tl.uint64), pointer)
    tl.store(P + index, pointer, index < 9)


@triton.jit
def _bias(Q, QB, K, KB, V, VB, M: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_q = index < M * 1024
    in_k = index < M * 1280
    k_index = index - M * 1024
    v_index = index - M * 1280
    output = tl.where(in_q, Q + index, tl.where(in_k, K + k_index, V + v_index))
    bias = tl.where(in_q, QB + index % 1024,
                    tl.where(in_k, KB + k_index % 256, VB + v_index % 256))
    value = tl.load(output, index < M * 1536, other=0).to(tl.float32)
    offset = tl.load(bias, index < M * 1536, other=0).to(tl.float32)
    tl.store(output, value + offset, index < M * 1536)


@torch.no_grad()
def run(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
        query_states, key_states, value_states):
    global _bridge
    if _bridge is None:
        _bridge = load_bridge()
    pointers = torch.empty(9, device=hidden_states.device, dtype=torch.int64)
    rows = hidden_states.numel() // 640
    plan = _bridge.prepare(rows)
    _pack_pointers[(1,)](hidden_states, q_weight, k_weight, v_weight,
                         query_states, key_states, value_states, pointers, num_warps=1)
    _bridge.execute(plan, pointers)
    _bias[(triton.cdiv(rows * 1536, 4096),)](
        query_states, q_bias, key_states, k_bias, value_states, v_bias,
        rows, 4096, num_warps=4,
    )
