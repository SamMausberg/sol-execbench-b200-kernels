# SPDX-License-Identifier: Apache-2.0
"""Guard TMA layout assumptions and support other input layouts with PyTorch."""

import torch
import torch.nn.functional as F

from cute_kernel import run as run_dense


def supports_tma(inputs):
    return all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in inputs)


@torch.no_grad()
def run(hidden_states, q_weight, k_weight, v_weight):
    inputs = (hidden_states, q_weight, k_weight, v_weight)
    if supports_tma(inputs):
        return run_dense(*inputs)
    batch, sequence, _ = hidden_states.shape
    q = F.linear(hidden_states, q_weight)
    k = F.linear(hidden_states, k_weight)
    v = F.linear(hidden_states, v_weight)
    return (q.view(batch, sequence, 16, 128).transpose(1, 2),
            k.view(batch, sequence, 4, 128).transpose(1, 2),
            v.view(batch, sequence, 4, 128).transpose(1, 2))
