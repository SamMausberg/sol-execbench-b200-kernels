"""Value projection with an ordinary tensor view in attention axis order."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states, v_proj_weight):
    batch, sequence, _ = hidden_states.shape
    projected = F.linear(hidden_states, v_proj_weight)
    return projected.view(batch, sequence, 8, 128).transpose(1, 2)
