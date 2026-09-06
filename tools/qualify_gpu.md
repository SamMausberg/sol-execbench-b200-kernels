# Qualifying local timing with shared GPU sessions

`qualify_gpu.py` runs separate full campaign trials and records the CUDA process list every 0.5 seconds. It waits for two seconds without CUDA contexts before launching each attempt. The campaign acquires the existing shared GPU lock normally.

```bash
source tools/native_env.sh
python tools/qualify_gpu.py 10 \
  --solution kernels/010_attention_value_projection_with_transpose/variants/cublaslt_v2/solution.json \
  --trials 3 --max-attempts 6 --deadline 1200
```

The helper tracks the campaign's descendants using Linux process IDs and process start times. Previously observed descendants remain identifiable after reparenting. A reused process ID with a different start time does not inherit ownership. The monitor reads process identities and the CUDA process list; it does not import Torch, acquire a CUDA context, change GPU settings, or alter another process.

Each attempt writes a campaign log and a JSONL process monitor under `.work/qualification/<problem_id>`. Its audit uses the trial start and finish times in the campaign's `run.json`, so another job seen while the campaign waits for the shared lock does not invalidate the trial. A foreign CUDA context observed within the trial window, a failed process query, or a large gap in monitoring excludes that attempt from performance qualification.

Qualified trials remain in the aggregate. Only attempts rejected for contention or incomplete monitoring are retried, within the attempt and elapsed-time limits. An official correctness or compilation failure stops the run. The helper stops starting attempts when its deadline passes and lets an already running campaign finish under the campaign's normal timeout.

`qualification.json` includes exact run, log, monitor, package, and source-tool hashes, per-attempt audits, and the aggregate from qualified trials. Package hashes, workload contracts, and scoring constants must agree across qualified trials. The aggregate uses the median latency for each workload and the pinned raw SOL formula.

The evidence establishes that no foreign CUDA context was observed during the sampled trial window. A competing kernel shorter than the sample interval can escape observation. Foreign contexts may also be idle, so rejection is conservative. All benchmark sessions should still cooperate through `SOL_GPU_LOCK`. This monitor does not resolve clock differences between Runpod and the hosted evaluator.

The CPU replay checks cover process identity reuse, reparented descendants, competition during a trial, benign lock waits, and missing monitor observations:

```bash
python -m unittest discover -s tools -p test_qualify_gpu.py -v
```

Process records store executable basenames. If an earlier monitor recorded full
executable paths, archived copies retain the original file hash and separately
identify basename redaction and the archived hash. Ownership and timestamps are
preserved.
