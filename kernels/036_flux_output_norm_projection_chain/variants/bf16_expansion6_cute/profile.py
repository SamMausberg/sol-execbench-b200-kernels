# SPDX-License-Identifier: Apache-2.0
"""One bounded CUPTI timeline and compiler-resource inspection."""
import hashlib
import json
import os
from pathlib import Path
import sys
import shutil

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[3]
if os.environ.get("SOL_36_CUTE_PROFILE_LOCKED") != "1":
    lock = os.environ.get("SOL_GPU_LOCK", str(ROOT / ".work/gpu.lock"))
    dump = ROOT / ".work/tuning/36-shared-fragments-compiler"
    dump.mkdir(parents=True, exist_ok=True)
    os.execvpe("flock", ["flock", "--exclusive", lock, "timeout", "240", sys.executable, __file__],
               {**os.environ, "SOL_36_CUTE_PROFILE_LOCKED": "1", "CUTE_DSL_KEEP_PTX": "1",
                "CUTE_DSL_KEEP_CUBIN": "1", "CUTE_DSL_DUMP_DIR": str(dump)})

import torch
from sol_execbench.core.bench.cupti_utils import collect_cupti_activities
from kernel import launch


torch.manual_seed(36911)
a = torch.randn((8192, 3072), dtype=torch.float32, device="cuda")
b = torch.randn((64, 3072), dtype=torch.float32, device="cuda")
bias = torch.randn(64, dtype=torch.float32, device="cuda")
out = torch.empty((8192, 64), dtype=torch.float32, device="cuda")
partials = torch.empty((4, 6, 8192, 64), dtype=torch.float32, device="cuda")
cache = torch.empty(torch.cuda.get_device_properties(0).L2_cache_size * 2, dtype=torch.uint8, device="cuda")
for _ in range(2):
    launch(a, b, bias, out, partials)
torch.cuda.synchronize()
report = {"kernel_sha256": hashlib.sha256((DIRECTORY / "kernel.py").read_bytes()).hexdigest(),
          "scope": "three cold-cache timeline samples; fixed input addresses; exploratory only", "samples": []}
for _ in range(3):
    cache.zero_()
    torch.cuda.synchronize()
    with collect_cupti_activities() as buffers:
        launch(a, b, bias, out, partials)
        torch.cuda.synchronize()
    rows = sorted(buffers.kernels, key=lambda k: k.start)
    report["samples"].append({
        "kernels": [{"name": k.name, "duration_us": (k.end - k.start) / 1000} for k in rows],
        "span_us": (max(k.end for k in rows) - min(k.start for k in rows)) / 1000,
        "gaps_us": [(right.start - left.end) / 1000 for left, right in zip(rows, rows[1:])],
    })
for path in (ROOT / ".work/tuning/36-shared-fragments-compiler").glob("*.ptx"):
    shutil.copy2(path, DIRECTORY / "reports/compiler" / path.name)
target = DIRECTORY / "reports" / "timeline-six-products.json"
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
