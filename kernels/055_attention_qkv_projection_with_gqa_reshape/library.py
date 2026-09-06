# SPDX-License-Identifier: Apache-2.0
"""Library comparison with the same concrete output views."""

import torch


@torch.no_grad()
def run(hidden_states, q_weight, k_weight, v_weight):
    batch, sequence, hidden = hidden_states.shape
    x = hidden_states.reshape(-1, hidden)
    q, k, v = [torch.mm(x, weight.T) for weight in (q_weight, k_weight, v_weight)]
    return (q.view(batch, sequence, 16, 128).transpose(1, 2),
            k.view(batch, sequence, 4, 128).transpose(1, 2),
            v.view(batch, sequence, 4, 128).transpose(1, 2))
