# SPDX-License-Identifier: BSD-3-Clause
#
# SOL-ExecBench problem 179:
#   output = silu(blockwise_fp8_mm(x, gate_weight)) *
#            blockwise_fp8_mm(x, up_weight)
#
# The tensor-core mainloop is a source-visible BSD-3-Clause derivative of the
# CUTLASS v4.4.1 SM100 blockwise GEMM example.  The second projection loads the
# first projection's in-place BF16 result in its epilogue and writes
# SiLU(gate) * up, avoiding all scratch tensors and a third kernel launch.

from __future__ import annotations

import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.runtime import from_dlpack
from blockwise_gemm_fused import BlockwiseGemmKernel


_HIDDEN_SIZE = 3584
_INTERMEDIATE_SIZE = 18944
_BLOCK_K = 128
_BLOCK_N = 128
_DEBUG = os.environ.get("SOL179_DEBUG", "0") == "1"

_COMPILED = {}
_MAX_ACTIVE_CLUSTERS = {}


class _GateUpFusedPipeline:
    """Enqueue both projections from one compiled CuTe host entry point."""

    def __init__(
        self,
        acc_dtype,
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
    ) -> None:
        common = {
            "acc_dtype": acc_dtype,
            "use_2cta_instrs": use_2cta_instrs,
            "mma_tiler_mn": mma_tiler_mn,
            "cluster_shape_mn": cluster_shape_mn,
        }
        self.gate_gemm = BlockwiseGemmKernel(
            **common,
            fuse_source_silu=False,
            pdl_role=1,
        )
        self.up_fused_gemm = BlockwiseGemmKernel(
            **common,
            fuse_source_silu=True,
            pdl_role=2,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        gate_b: cute.Tensor,
        up_b: cute.Tensor,
        output_c: cute.Tensor,
        sfa: cute.Tensor,
        gate_sfb: cute.Tensor,
        up_sfb: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.gate_gemm(
            a,
            gate_b,
            output_c,
            output_c,  # constexpr-disabled source argument
            sfa,
            gate_sfb,
            max_active_clusters,
            stream,
        )
        self.up_fused_gemm(
            a,
            up_b,
            output_c,
            output_c,
            sfa,
            up_sfb,
            max_active_clusters,
            stream,
        )


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


def _select_config(_m: int):
    # Submission 36464 established this as the fastest schedule on B200.
    # N/128 == 148 is exactly divisible by cluster-N=4, with no tail cluster.
    return False, (128, 128), (1, 4)


def _compile_or_get(
    config,
    a: cute.Tensor,
    gate_b: cute.Tensor,
    up_b: cute.Tensor,
    output_c: cute.Tensor,
    sfa: cute.Tensor,
    gate_sfb: cute.Tensor,
    up_sfb: cute.Tensor,
    stream,
    device_index: int,
):
    use_2cta, mma_tiler_mn, cluster_shape_mn = config
    key = (
        device_index,
        use_2cta,
        mma_tiler_mn,
        cluster_shape_mn,
    )
    cached = _COMPILED.get(key)
    if cached is not None:
        return cached

    max_clusters = _max_active_clusters(device_index, cluster_shape_mn)
    pipeline = _GateUpFusedPipeline(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=use_2cta,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )
    compiled = cute.compile(
        pipeline,
        a,
        gate_b,
        up_b,
        output_c,
        sfa,
        gate_sfb,
        up_sfb,
        max_clusters,
        stream,
        # CUTLASS's CUDA 13.1 path uses opt-level 2 for this kernel family.
        options="--opt-level 2",
    )
    cached = (compiled, max_clusters)
    _COMPILED[key] = cached
    return cached


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
    a_bits_3d = _view_with_unit_batch(x.view(torch.int8))
    gate_bits_3d = _view_with_unit_batch(gate_proj_weight.view(torch.int8))
    up_bits_3d = _view_with_unit_batch(up_proj_weight.view(torch.int8))
    output_3d = _view_with_unit_batch(output)
    scale_x_3d = _view_with_unit_batch(scale_x)
    scale_gate_3d = _view_with_unit_batch(scale_gate)
    scale_up_3d = _view_with_unit_batch(scale_up)

    a = _to_cute_fp8_dynamic(a_bits_3d)
    gate_b = _to_cute_fp8_static(gate_bits_3d)
    up_b = _to_cute_fp8_static(up_bits_3d)
    output_c = _to_cute_dynamic(output_3d)
    sfa = _to_cute_dynamic(scale_x_3d)
    gate_sfb = _to_cute_static(scale_gate_3d)
    up_sfb = _to_cute_static(scale_up_3d)

    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _current_stream(x.device)

    compiled, max_clusters = _compile_or_get(
        config,
        a,
        gate_b,
        up_b,
        output_c,
        sfa,
        gate_sfb,
        up_sfb,
        stream,
        device_index,
    )
    compiled(
        a,
        gate_b,
        up_b,
        output_c,
        sfa,
        gate_sfb,
        up_sfb,
        max_clusters,
        stream,
    )
