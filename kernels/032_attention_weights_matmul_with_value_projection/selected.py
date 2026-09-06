# SPDX-License-Identifier: Apache-2.0
"""Direct final-layout attention/value products with shape-based kernel selection."""
import torch
from kernel import launch as triton_launch
from cute_kernel import launch as cute_launch


@torch.no_grad()
def run(attn_weights, value_states):
    batch, _, sequence, _ = attn_weights.shape
    if any(not tensor.is_contiguous() or tensor.data_ptr() % 16
           for tensor in (attn_weights, value_states)):
        return torch.matmul(attn_weights, value_states).transpose(1, 2).reshape(batch, sequence, 5120)
    output = torch.empty((batch, sequence, 5120), dtype=attn_weights.dtype, device=attn_weights.device)
    if sequence % 128 == 0:
        bm = 64 if sequence == 256 and batch <= 2 else 128
        cute_launch(attn_weights, value_states, output, bm, 128, reverse=sequence == 128)
    elif sequence < 192:
        triton_launch(attn_weights, value_states, output, 16, 128, 64, 4, 2, mma_v2=True)
    elif sequence < 384:
        triton_launch(attn_weights, value_states, output, 32, 128, 64, 4, 3, mma_v2=True)
    else:
        triton_launch(attn_weights, value_states, output, 64, 128, 128, 4, 2)
    return output
