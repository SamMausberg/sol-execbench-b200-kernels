#!/usr/bin/env python3
"""Local mini-evaluator for SOL-ExecBench kernel 001 on the SM120 dev GPU.

Replicates the official evaluator's correctness check (atol+rtol*|ref| bound,
required_matched_ratio=0.99, nan/inf and all-zero rejection) and its
L2-cold-cache CUDA-event timing loop. The reference is computed in a
memory-frugal chunked form (identical math per chunk, fp32) so large
workloads fit in 16 GB.

Usage:
  python3 harness.py --candidate kernel_torch.py [--workloads 0,2,6] [--rounds 3]
  python3 harness.py --candidate ../variants/x/build/benchmark_kernel.so --time
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent.resolve()
PROBLEM = HERE.parent

MAX_ATOL = 1e-5
MAX_RTOL = 0.05
REQUIRED_MATCHED_RATIO = 0.99

H, KV, D = 80, 8, 128


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_workloads():
    wls = []
    for line in (PROBLEM / "workload.jsonl").read_text().splitlines():
        if line.strip():
            w = json.loads(line)
            a = w["axes"]
            wls.append((a["batch_size"], a["seq_len_q"], a["seq_len_kv"]))
    return wls


def gen_inputs(B, Sq, Skv, device):
    ref = load_module(PROBLEM / "reference.py", "reference001")
    axes = {"batch_size": B, "seq_len_q": Sq, "seq_len_kv": Skv}
    return ref.get_inputs(axes, torch.device(device))


def reference_chunked(inp, hchunk=10):
    """Reference math per (batch, head-chunk) in fp32; matches reference.py.

    grad_value_states needs the group-sum over 10 heads; process exactly one
    GQA group (10 heads) at a time so the sum is formed the same way.
    """
    dO = inp["grad_attn_output"]
    W = inp["attn_weights"]
    Wd = inp["attn_weights_dropped"]
    V = inp["value_states"]
    m = inp["dropout_mask"]
    p = inp["attention_dropout"]
    B, Sq = dO.shape[0], dO.shape[1]
    Skv = V.shape[2]
    G = H // KV  # 10
    dS = torch.empty(B, H, Sq, Skv, dtype=torch.bfloat16, device=dO.device)
    dV = torch.empty(B, KV, Skv, D, dtype=torch.bfloat16, device=dO.device)
    for b in range(B):
        for kvh in range(KV):
            hs = slice(kvh * G, (kvh + 1) * G)
            dOg = dO[b, :, hs, :].transpose(0, 1).to(torch.float32)  # [G,Sq,D]
            Vg = V[b, kvh].to(torch.float32)  # [Skv,D]
            dPd = torch.matmul(dOg, Vg.transpose(-2, -1))  # [G,Sq,Skv]
            dW = dPd * m[b, hs] / (1.0 - p) if p > 0 else dPd
            Wf = W[b, hs].to(torch.float32)
            s = (dW * Wf).sum(dim=-1, keepdim=True)
            dS[b, hs] = (Wf * (dW - s)).to(torch.bfloat16)
            dVe = torch.matmul(
                Wd[b, hs].to(torch.float32).transpose(-2, -1), dOg
            )  # [G,Skv,D]
            dV[b, kvh] = dVe.sum(dim=0).to(torch.bfloat16)
    return dS, dV


def check_output(name, out, ref):
    x = out.to(torch.float32)
    y = ref.to(torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return False, f"{name}: non-finite values"
    if y.norm() > 0 and x.norm() == 0:
        return False, f"{name}: all-zero output"
    abs_err = (x - y).abs()
    bound = MAX_ATOL + MAX_RTOL * y.abs()
    exceeds = (abs_err > bound).sum().item()
    total = abs_err.numel()
    ratio = 1.0 - exceeds / total
    max_abs = abs_err.max().item()
    rel = abs_err / y.abs().clamp(min=MAX_ATOL)
    ok = ratio >= REQUIRED_MATCHED_RATIO
    return ok, (
        f"{name}: matched={ratio:.6f} (need {REQUIRED_MATCHED_RATIO}), "
        f"max_abs={max_abs:.3e}, max_rel={rel.max().item():.3e}, "
        f"exceeds={exceeds}/{total}"
    )


def make_args(inp, device):
    """Inputs in definition order + DPS outputs."""
    B, Sq = inp["grad_attn_output"].shape[0], inp["grad_attn_output"].shape[1]
    Skv = inp["value_states"].shape[2]
    outs = [
        torch.zeros(B, H, Sq, Skv, dtype=torch.bfloat16, device=device),
        torch.zeros(B, KV, Skv, D, dtype=torch.bfloat16, device=device),
    ]
    args = [
        inp["grad_attn_output"],
        inp["attn_weights"],
        inp["attn_weights_dropped"],
        inp["value_states"],
        inp["dropout_mask"],
        inp["attention_dropout"],
    ]
    return args, outs


def bench(fn, args, outs, device, warmup=10, rep=30):
    props = torch.cuda.get_device_properties(device)
    cache = torch.empty(props.L2_cache_size * 2, dtype=torch.int8, device=device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    torch.cuda.synchronize()
    for _ in range(warmup):
        cache.zero_()
        fn(*args, *outs)
    for i in range(rep):
        cache.zero_()
        starts[i].record()
        fn(*args, *outs)
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--workloads", default=None, help="comma indices; default: all that fit")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--extra", action="store_true", help="add synthetic edge-case workloads")
    args_ns = ap.parse_args()

    device = "cuda:0"
    cand_path = Path(args_ns.candidate)
    if cand_path.suffix == ".so":
        spec = importlib.util.spec_from_file_location("benchmark_kernel", cand_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.run
    else:
        fn = load_module(cand_path, "candidate001").run

    wls = get_workloads()
    if args_ns.extra:
        wls += [(3, 373, 449), (1, 96, 160), (2, 33, 65), (1, 1, 1), (1, 130, 258), (2, 255, 511)]
    free_gb = (torch.cuda.mem_get_info()[0]) / 1e9
    sel = None
    if args_ns.workloads:
        sel = {int(x) for x in args_ns.workloads.split(",")}

    all_ok = True
    for i, (B, Sq, Skv) in enumerate(wls):
        if sel is not None and i not in sel:
            continue
        need = B * H * Sq * Skv * 9.5 / 1e9 + B * Sq * H * D * 8 / 1e9 + 0.7
        if sel is None and need > free_gb * 0.92:
            print(f"[{i:2d}] B={B} Sq={Sq} Skv={Skv}: SKIP (needs ~{need:.1f} GB)")
            continue
        for r in range(args_ns.rounds):
            torch.manual_seed(1234 + 977 * i + r)
            inp = gen_inputs(B, Sq, Skv, device)
            call_args, outs = make_args(inp, device)
            fn(*call_args, *outs)
            torch.cuda.synchronize()
            ref_dS, ref_dV = reference_chunked(inp)
            ok1, msg1 = check_output("grad_attn_scores", outs[0], ref_dS)
            ok2, msg2 = check_output("grad_value_states", outs[1], ref_dV)
            status = "PASS" if ok1 and ok2 else "FAIL"
            all_ok &= ok1 and ok2
            print(f"[{i:2d}] B={B} Sq={Sq} Skv={Skv} r{r}: {status}\n     {msg1}\n     {msg2}")
            del ref_dS, ref_dV
            if not (ok1 and ok2):
                break
        if args_ns.time:
            t = bench(fn, call_args, outs, device)
            print(f"     median latency: {t*1000:.1f} us")
        del inp, call_args, outs
        gc_free()
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


def gc_free():
    import gc

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
