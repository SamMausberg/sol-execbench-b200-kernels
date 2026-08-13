# SPDX-License-Identifier: BSD-3-Clause
#
# SOL-ExecBench problem 179:
#   output = silu(blockwise_fp8_mm(x, gate_weight)) *
#            blockwise_fp8_mm(x, up_weight)
#
# The tensor-core mainloop is NVIDIA's official CUTLASS v4.4.1 SM100
# blockwise GEMM implementation from the evaluator image. This file supplies
# shape-specialized scheduling, invocation-local scratch, and the BF16 epilogue.

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
import triton
import triton.language as tl
from cutlass.cute.runtime import from_dlpack


_HIDDEN_SIZE = 3584
_INTERMEDIATE_SIZE = 18944
_BLOCK_K = 128
_BLOCK_N = 128
_EPILOGUE_BLOCK = 1024
_DEBUG = os.environ.get("SOL179_DEBUG", "0") == "1"

_COMPILED = {}
_MAX_ACTIVE_CLUSTERS = {}


def _load_official_blockwise_kernel():
    cutlass_root = Path(os.environ.get("CUTLASS_DIR", "/usr/local/cutlass"))
    source = (
        cutlass_root
        / "examples"
        / "python"
        / "CuTeDSL"
        / "blackwell"
        / "blockwise_gemm"
        / "blockwise_gemm.py"
    )
    if not source.is_file():
        raise RuntimeError(
            "The official CUTLASS blockwise GEMM source was not found at "
            f"{source}. SOL-ExecBench's pinned image installs it there."
        )

    module_name = "_sol179_cutlass_blockwise_gemm_v441"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import CUTLASS source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.BlockwiseGemmKernel


BlockwiseGemmKernel = _load_official_blockwise_kernel()


@triton.jit
def _swiglu_bf16_epilogue(
    gate_ptr,
    up_ptr,
    output_ptr,
    BLOCK: tl.constexpr,
):
    # All official workloads have M % 128 == 0 and N == 148 * 128, so the
    # flattened element count is divisible by 16,384 and this launch has no tail.
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(gate_ptr + offsets).to(tl.float32)
    up = tl.load(up_ptr + offsets).to(tl.float32)

    # Match the reference's precision boundary:
    # GEMM -> BF16, SiLU -> BF16, multiply -> BF16.
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
    result = activated.to(tl.float32) * up
    tl.store(output_ptr + offsets, result)


def _view_with_unit_batch(tensor_2d: torch.Tensor) -> torch.Tensor:
    rows, cols = tensor_2d.shape
    return torch.as_strided(
        tensor_2d,
        size=(rows, cols, 1),
        stride=(tensor_2d.stride(0), tensor_2d.stride(1), rows * cols),
    )


def _to_cute_dynamic(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(
        tensor,
        assumed_align=16,
        use_32bit_stride=True,
    ).mark_layout_dynamic(leading_dim=1)


def _to_cute_static(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16)


def _to_cute_fp8_dynamic(tensor_bits: torch.Tensor) -> cute.Tensor:
    tensor = from_dlpack(
        tensor_bits,
        assumed_align=16,
        use_32bit_stride=True,
    )
    tensor.element_type = cutlass.Float8E4M3FN
    return tensor.mark_layout_dynamic(leading_dim=1)


def _to_cute_fp8_static(tensor_bits: torch.Tensor) -> cute.Tensor:
    tensor = from_dlpack(tensor_bits, assumed_align=16)
    tensor.element_type = cutlass.Float8E4M3FN
    return tensor


def _current_stream(device: torch.device):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _max_active_clusters(device_index: int, cluster_shape: tuple[int, int]) -> int:
    key = (device_index, cluster_shape)
    value = _MAX_ACTIVE_CLUSTERS.get(key)
    if value is None:
        cluster_size = cluster_shape[0] * cluster_shape[1]
        value = cutlass_utils.HardwareInfo().get_max_active_clusters(cluster_size)
        _MAX_ACTIVE_CLUSTERS[key] = value
    return value


def _fresh_up_output(output: torch.Tensor) -> torch.Tensor:
    # Numeric scratch is invocation-local. It is completely recomputed from
    # the current inputs and can never expose data from an earlier call.
    return torch.empty_like(output)


def _select_config(m: int):
    # N/128 == 148 is exactly divisible by four. This gives four-way
    # activation TMA multicast with no cluster tail on any workload.
    return False, (128, 128), (1, 4)


def _compile_or_get(
    config,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    sfa: cute.Tensor,
    sfb: cute.Tensor,
    stream,
    device_index: int,
):
    use_2cta, mma_tiler_mn, cluster_shape_mn = config
    key = (device_index, use_2cta, mma_tiler_mn, cluster_shape_mn)
    compiled = _COMPILED.get(key)
    if compiled is not None:
        return compiled

    max_clusters = _max_active_clusters(device_index, cluster_shape_mn)
    gemm = BlockwiseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=use_2cta,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )
    compiled = cute.compile(
        gemm,
        a,
        b,
        c,
        sfa,
        sfb,
        max_clusters,
        stream,
        # CUTLASS's CUDA 13.1 path uses opt-level 2 for this kernel family.
        options="--opt-level 2",
    )
    _COMPILED[key] = compiled
    return compiled


def _debug_validate(
    x: torch.Tensor,
    scale_x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    scale_gate: torch.Tensor,
    up_proj_weight: torch.Tensor,
    scale_up: torch.Tensor,
    output: torch.Tensor,
) -> None:
    m, k = x.shape
    expected = {
        "x": ((m, _HIDDEN_SIZE), torch.float8_e4m3fn),
        "scale_x": ((m, _HIDDEN_SIZE // _BLOCK_K), torch.float32),
        "gate_proj_weight": (
            (_INTERMEDIATE_SIZE, _HIDDEN_SIZE),
            torch.float8_e4m3fn,
        ),
        "scale_gate": (
            (_INTERMEDIATE_SIZE // _BLOCK_N, _HIDDEN_SIZE // _BLOCK_K),
            torch.float32,
        ),
        "up_proj_weight": (
            (_INTERMEDIATE_SIZE, _HIDDEN_SIZE),
            torch.float8_e4m3fn,
        ),
        "scale_up": (
            (_INTERMEDIATE_SIZE // _BLOCK_N, _HIDDEN_SIZE // _BLOCK_K),
            torch.float32,
        ),
        "output": ((m, _INTERMEDIATE_SIZE), torch.bfloat16),
    }
    tensors = {
        "x": x,
        "scale_x": scale_x,
        "gate_proj_weight": gate_proj_weight,
        "scale_gate": scale_gate,
        "up_proj_weight": up_proj_weight,
        "scale_up": scale_up,
        "output": output,
    }
    if k != _HIDDEN_SIZE or m % 128 != 0:
        raise ValueError(f"Unsupported contract shape M={m}, K={k}")
    for name, tensor in tensors.items():
        shape, dtype = expected[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name}: expected shape={shape}, dtype={dtype}; "
                f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CUDA tensor")


@torch.no_grad()
def run(
    x: torch.Tensor,
    scale_x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    scale_gate: torch.Tensor,
    up_proj_weight: torch.Tensor,
    scale_up: torch.Tensor,
    output: torch.Tensor,
):
    if _DEBUG:
        _debug_validate(
            x,
            scale_x,
            gate_proj_weight,
            scale_gate,
            up_proj_weight,
            scale_up,
            output,
        )

    m = x.shape[0]
    config = _select_config(m)
    up_output = _fresh_up_output(output)

    a_bits_3d = _view_with_unit_batch(x.view(torch.int8))
    gate_bits_3d = _view_with_unit_batch(gate_proj_weight.view(torch.int8))
    up_bits_3d = _view_with_unit_batch(up_proj_weight.view(torch.int8))
    gate_output_3d = _view_with_unit_batch(output)
    up_output_3d = _view_with_unit_batch(up_output)
    scale_x_3d = _view_with_unit_batch(scale_x)
    scale_gate_3d = _view_with_unit_batch(scale_gate)
    scale_up_3d = _view_with_unit_batch(scale_up)

    a = _to_cute_fp8_dynamic(a_bits_3d)
    gate_b = _to_cute_fp8_static(gate_bits_3d)
    up_b = _to_cute_fp8_static(up_bits_3d)
    gate_c = _to_cute_dynamic(gate_output_3d)
    up_c = _to_cute_dynamic(up_output_3d)
    sfa = _to_cute_dynamic(scale_x_3d)
    gate_sfb = _to_cute_static(scale_gate_3d)
    up_sfb = _to_cute_static(scale_up_3d)

    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _current_stream(x.device)

    compiled = _compile_or_get(
        config,
        a,
        gate_b,
        gate_c,
        sfa,
        gate_sfb,
        stream,
        device_index,
    )

    # One compiled dynamic-M kernel serves both projections and all 16 workloads.
    compiled(a, gate_b, gate_c, sfa, gate_sfb, stream)
    compiled(a, up_b, up_c, sfa, up_sfb, stream)

    total = output.numel()
    if _DEBUG and total % _EPILOGUE_BLOCK != 0:
        raise ValueError("The shape-specialized epilogue requires exact 1024-element tiles")
    grid = (total // _EPILOGUE_BLOCK,)
    _swiglu_bf16_epilogue[grid](
        output,
        up_output,
        output,
        BLOCK=_EPILOGUE_BLOCK,
        num_warps=8,
    )
