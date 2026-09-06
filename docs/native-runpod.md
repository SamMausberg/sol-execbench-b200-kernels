# Native B200 work on Runpod

Keep the checkout, virtual environment, compiler and caches under `/workspace`.
Confirm the pod mounts `/workspace` as its persistent volume; persistence across
pod deletion depends on the Runpod volume type. Git pushes provide a separate
backup of committed source code.

```bash
cd /workspace/sol-execbench-b200-kernels
bash tools/bootstrap_native.sh
source tools/native_env.sh
```

The bootstrap installs the frozen evaluator dependency graph, then applies the
same `fbtriton==3.7.1` and `nvidia-cutlass-dsl-libs-cu13==4.4.2` overrides as the
pinned evaluator Dockerfile. It installs CUTLASS v4.4.1 and the CUDA 13.1 compiler
wheels, including matching NVVM. The NVIDIA driver remains a host prerequisite.
Run `make native-info` to inspect the resulting software, compiler and GPU state.

The pinned fbtriton 3.7.1 wheel contains `triton/runtime/launch.h`, but its NVIDIA
driver searches `triton/backends/nvidia/launch.h`. Bootstrap creates that missing
path as a symlink to the wheel's own header and verifies identical content. This
repairs the package layout without changing the evaluator or Triton source.

`SOL_WORKSPACE`, `SOL_VENV`, `SOL_CUDA_HOME`, `CUTLASS_DIR` and cache variables can
override the default paths before sourcing the environment. The shared GPU lock
defaults to `/workspace/sol-execbench-b200-kernels/.work/gpu.lock`. Set
`SOL_GPU_LOCK` to the same absolute path in every checkout and other GPU project.

## Extract and package a problem

Use the website problem ID. Its dataset name can have a different numeric prefix:
website #179 is `003_fp8_mlp_gate_up_projection` in the Quant collection.

```bash
python tools/campaign.py fetch 25 53 84 85 88 179
python tools/campaign.py package 88 \
  --solution kernels/088_rotary_position_embedding_application/solution.json \
  --output dist/88-b200.json
```

`fetch` verifies each parquet against the SHA-256 supplied by Hugging Face for the
revision in `benchmark.lock.json`, extracts the definition and every workload,
and matches the contract to the live website by name and workload axes. Extracted
contracts and provenance live in `.work/problems/<website_id>/`.

A source manifest uses the official solution schema. Each `sources` entry can
contain a relative `path` alone, or embedded `content`. `package` embeds every
source, validates it with the pinned evaluator, sets the target to B200 and writes
deterministic JSON plus a SHA-256 sidecar. List all local helper source files in
the manifest. The `stage` command is an alias for `package`.

## Evaluate and compare

```bash
python tools/campaign.py bench 88 \
  --solution kernels/088_rotary_position_embedding_application/solution.json \
  --workload 0 --label first-check

python tools/campaign.py bench 88 \
  --solution dist/88-b200.json --trials 3 --label candidate
```

Each run acquires the shared file lock for compilation and all evaluator trials.
Other GPU clients must cooperate with the same lock, for example:

```bash
flock "$SOL_GPU_LOCK" python other_project/benchmark.py
```

The run uses the unmodified official evaluator and preserves its complete logs
and raw JSONL traces. Artifacts under `.work/runs/<website_id>/<run>/` include:

- The exact evaluated `solution.json`, its recorded SHA-256, and source hashes.
- Full and selected workloads, the definition, and pinned contract provenance.
- The live scoring API response and retrieval time.
- Installed package versions, compiler versions, evaluator revision, environment,
  GPU state, clocks before and after each trial, exit codes, and elapsed times.
- `summary.json` with each workload's correctness, latencies, spread and SOL score.

The default is unlocked timing. This Runpod host denied GPU clock changes when
probed on September 6, 2026. Unlocked measurements are local estimates and can
differ from the official evaluation environment. On a host that permits clock
changes, `--lock-clocks` applies and verifies the official clock presets, requires
them during evaluation, and resets them on exit. A failed lock attempt stops the
run before measurement.

Scores use each workload's `baseline_latency_ms` and `sol_ms` from
`/api/kernels/<website_id>`. The problem score is the arithmetic mean of the
individual workload SOL scores. The summary applies the pinned evaluator's
formula to each workload's median latency across trials; it also reports a score
for each trial. Geometric mean latency is a separate descriptive metric.
The pinned formula is not capped at one; observations faster than the supplied
SOL estimate are flagged for inspection. Incomplete or failing runs receive no
full problem score. Selected workload runs report their local subset estimate.

```bash
python tools/campaign.py summarize .work/runs/88/<run-directory>
```

Replaying a summary checks the archived package, scoring snapshot and contract
hashes, then checks trace identity, workload UUIDs and axes. An official rank
requires submission and evaluation on NVIDIA's service.

Some pinned FlashInfer-Bench definitions encode an absent `hf_id` as an empty
string. The pinned evaluator accepts `null` for this optional provenance field
and rejects the empty string. For these records, the campaign preserves the
original `definition.json` bytes and contract hash, writes a separate
`definition.native.json` with `hf_id: null`, and selects it through the official
CLI's `--definition` option. Run and summary records explicitly label this
metadata adaptation and include both file hashes and the exact field change.
Summary validation rejects additional changes, including rehashed reference
changes. Run the focused checks with `python tools/test_campaign_metadata.py`.

The `native-fetch`, `native-package`, `native-test` and `native-bench` Make targets
accept `KERNEL_ID` and `SOLUTION`. `native-test` runs one complete trial;
`native-bench` runs three by default. Add `NATIVE_ARGS='--workload 0'` for a selected
workload or `NATIVE_ARGS='--lock-clocks'` on a host with clock permissions.
