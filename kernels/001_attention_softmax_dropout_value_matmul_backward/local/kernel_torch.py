"""Scheme A in pure PyTorch: correctness anchor for the reformulated math.

dV  = Wd_cat^T @ dO_cat          (GQA group-sum folded into GEMM K = 10*Sq)
dP  = dO_cat @ V^T               (materialized bf16)
Dlt = rowsum(dP * m * W) / (1-p)
dS  = W * (dP * m / (1-p) - Dlt)
"""

import torch

H, KV, D = 80, 8, 128
G = H // KV


def run(grad_attn_output, attn_weights, attn_weights_dropped, value_states,
        dropout_mask, attention_dropout, grad_attn_scores, grad_value_states):
    dO = grad_attn_output
    W = attn_weights
    Wd = attn_weights_dropped
    V = value_states
    m = dropout_mask
    p = float(attention_dropout)
    B, Sq = dO.shape[0], dO.shape[1]
    Skv = V.shape[2]

    dO_t = dO.transpose(1, 2).contiguous()          # [B,H,Sq,D]
    dO_cat = dO_t.view(B * KV, G * Sq, D)
    Wd_cat = Wd.view(B * KV, G * Sq, Skv)

    # dV: [B*KV, Skv, D] bf16, fp32 accumulation inside cuBLAS
    torch.bmm(Wd_cat.transpose(1, 2), dO_cat,
              out=grad_value_states.view(B * KV, Skv, D))

    # dP: [B*KV, G*Sq, Skv] bf16
    dP = torch.bmm(dO_cat, V.view(B * KV, Skv, D).transpose(1, 2))
    dP = dP.view(B, H, Sq, Skv)

    inv = 1.0 / (1.0 - p) if p > 0 else 1.0
    dW = dP.float() * m * inv
    Wf = W.float()
    delta = (dW * Wf).sum(dim=-1, keepdim=True)
    grad_attn_scores.copy_((Wf * (dW - delta)).to(torch.bfloat16))
