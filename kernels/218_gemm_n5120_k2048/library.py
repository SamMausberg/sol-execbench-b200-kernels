# SPDX-License-Identifier: Apache-2.0
"""Exact torch.mm comparison for the captured Qwen FP16 GEMM."""

import torch


@torch.no_grad()
def run(A, B, C):
    torch.mm(A, B.T, out=C)
