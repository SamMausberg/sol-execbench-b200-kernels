#!/usr/bin/env python3
"""Package a kernel-001 variant into a SOL-ExecBench solution.json."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()
PROBLEM = HERE.parent

DEFINITION = "001_attention_softmax_dropout_value_matmul_backward"

VARIANTS = {
    "flash_v1": {
        "name": "attn_sdpa_bwd_b200_tcgen05_v1",
        "description": (
            "Two fused kernels: K1 streams attn_weights_dropped once, computing "
            "grad_value_states as a single GQA-folded tcgen05 GEMM (K = 10*Sq) "
            "plus per-tile softmax-backward row-sum partials; K2 recomputes "
            "dO@V^T on tensor cores with a fused dropout/softmax-backward "
            "epilogue (PDL-overlapped). TMA everywhere; predicated loader "
            "path for non-16B-aligned seq_len_kv."
        ),
    },
    "safe_v1": {
        "name": "attn_sdpa_bwd_b200_cublas_fused_v1",
        "description": (
            "cuBLAS batched GEMMs with the GQA reduction folded into the "
            "contraction (K = 10*Sq), a bf16 dP workspace, and one fused "
            "smem-staged kernel computing the softmax/dropout backward "
            "(row-sum + dS) in a single pass over dP/W/mask."
        ),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("--author", default="Sam")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    meta = VARIANTS[args.variant]
    src = (PROBLEM / "variants" / args.variant / "kernel.cu").read_text()

    solution = {
        "name": meta["name"],
        "definition": DEFINITION,
        "author": args.author,
        "description": meta["description"],
        "spec": {
            "languages": ["cuda_cpp"],
            "target_hardware": ["B200"],
            "entry_point": "kernel.cu::run",
            "destination_passing_style": True,
            "binding": "torch",
            "dependencies": [],
            "compile_options": {
                "cuda_cflags": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-DNDEBUG",
                    "-gencode=arch=compute_100a,code=sm_100a",
                ],
                "ld_flags": ["-lcuda"],
            },
        },
        "sources": [{"path": "kernel.cu", "content": src}],
    }

    out = Path(args.output) if args.output else (
        PROBLEM / "dist" / f"solution_{args.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(solution, indent=1))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
