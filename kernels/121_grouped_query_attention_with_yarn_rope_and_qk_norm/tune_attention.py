"""Measure attention tile choices with the official cold-L2 CUPTI helper."""

import argparse
import datetime
import fcntl
import json
from pathlib import Path
import runpy
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.timing import bench_gpu_time_with_cupti
from sol_execbench.core.data.workload import ToleranceSpec

from variants.attention_experiments import _grouped_attention, _normalize_rotate_qk, _yarn_coefficients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=int, action="append")
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--tma", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    ref = runpy.run_path(str(root / ".work/problems/121/reference.py"))
    workloads = [json.loads(line) for line in (root / ".work/problems/121/workload.jsonl").read_text().splitlines()]
    configurations = [(64, 64, 4, 3, False), (64, 64, 4, 1, True),
                      (128, 64, 4, 1, True), (128, 64, 8, 1, True),
                      (64, 128, 4, 1, True), (128, 128, 8, 1, True)]
    records = []
    with (root / ".work/gpu.lock").open("a+") as lock:
        print("Waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.tma:
            triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.uint8))
        for index in args.workload or [3, 0]:
            workload = workloads[index]
            inputs = ref["get_inputs"](workload["axes"], torch.device("cuda"))
            expected = ref["run"](**inputs)
            hidden = inputs["hidden_states"]
            batch, seq, width = hidden.shape
            rows = batch * seq
            query = F.linear(hidden, inputs["q_proj_weight"])
            key = F.linear(hidden, inputs["k_proj_weight"])
            value = F.linear(hidden, inputs["v_proj_weight"])
            coefficients = torch.empty((rows, 128), dtype=torch.bfloat16, device="cuda")
            q_rot, k_rot = torch.empty_like(query), torch.empty_like(key)
            pos, mask = inputs["position_ids"], inputs["attention_mask"]
            _yarn_coefficients[(triton.cdiv(rows * 64, 256),)](
                pos, inputs["inv_freq"], coefficients, rows, seq,
                pos.stride(0), pos.stride(1), inputs["attention_factor"], 256,
                num_warps=4, enable_fp_fusion=False,
            )
            _normalize_rotate_qk[(rows, 2)](
                query, key, inputs["q_norm_weight"], inputs["k_norm_weight"],
                coefficients, q_rot, k_rot, inputs["rms_norm_eps"],
                num_warps=8, enable_fp_fusion=False,
            )
            merged = torch.empty_like(query).view(rows, width)
            output = torch.empty_like(hidden)
            if args.tma:
                configurations = [(64, 128, 4, 1, True), (128, 128, 8, 1, True),
                                  (64, 128, 4, 2, False), (128, 128, 8, 2, False)]
            if args.layout:
                q_rot = q_rot.view(batch, seq, 40, 128).transpose(1, 2).contiguous()
                k_rot = k_rot.view(batch, seq, 8, 128).transpose(1, 2).contiguous()
                value = value.view(batch, seq, 8, 128).transpose(1, 2).contiguous()
                configurations = [(64, 128, 4, 1, True), (128, 128, 8, 1, True),
                                  (64, 128, 4, 2, False), (128, 128, 8, 2, False)]
            for bm, bn, warps, stages, skip in configurations:
                record = {"workload": index, "axes": workload["axes"],
                          "block_m": bm, "block_n": bn, "num_warps": warps,
                          "num_stages": stages, "skip_empty_mask_tiles": skip}
                record["head_major"] = args.layout
                record["tma"] = args.tma
                def attention():
                    return _grouped_attention[(triton.cdiv(seq, bm), batch * 40)](
                        q_rot, k_rot, value, mask, merged, seq,
                        mask.stride(0), mask.stride(2), mask.stride(3),
                        inputs["scaling"], bm, bn, skip, args.layout, args.tma, args.tma and not skip,
                        num_warps=warps, num_stages=stages, enable_fp_fusion=False,
                    )
                try:
                    compiled = attention()
                    torch.mm(merged, inputs["o_proj_weight"].t(), out=output.view(rows, width))
                    tolerance = ToleranceSpec(**workload["tolerance"])
                    stats, exceeds = compute_error_stats(output, expected, tolerance)
                    record.update(passed=not exceeds, errors=stats.model_dump(),
                                  registers=compiled.n_regs, spills=compiled.n_spills,
                                  shared_bytes=compiled.metadata.shared)
                    if not exceeds:
                        samples = bench_gpu_time_with_cupti(attention, warmup=3, rep=20, cold_l2_cache=True)
                        record["latency_ms"] = statistics.median(samples)
                except Exception as exc:
                    record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                records.append(record)
                print(json.dumps(record), flush=True)
            if args.layout:
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    def cudnn_attention():
                        return F.scaled_dot_product_attention(
                            q_rot, k_rot, value, attn_mask=mask, dropout_p=0.0,
                            is_causal=False, scale=inputs["scaling"], enable_gqa=True,
                        )
                    attention_output = cudnn_attention()
                    torch.mm(attention_output.transpose(1, 2).contiguous().view(rows, width),
                             inputs["o_proj_weight"].t(), out=output.view(rows, width))
                    stats, exceeds = compute_error_stats(output, expected, tolerance)
                    samples = bench_gpu_time_with_cupti(cudnn_attention, warmup=3, rep=20, cold_l2_cache=True)
                    record = {"workload": index, "axes": workload["axes"], "implementation": "cudnn_head_major",
                              "passed": not exceeds, "errors": stats.model_dump(),
                              "latency_ms": statistics.median(samples)}
                    records.append(record)
                    print(json.dumps(record), flush=True)
            del inputs, expected, query, key, value, coefficients, q_rot, k_rot, merged, output
    target = root / ".work/tuning/121"
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"attention-{stamp}.json"
    path.write_text(json.dumps(records, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
