from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

ROOT = Path('/workspace/sol-execbench-b200-kernels')
sys.path.insert(0, str(ROOT / 'tools'))
from qualify_gpu import aggregate, now, run_attempt

stamp = now().replace('-', '').replace(':', '').replace('.', '').split('+')[0]
directory = ROOT / '.work/qualification/4' / (stamp + '-cublaslt-selected')
directory.mkdir(parents=True)
args = SimpleNamespace(problem_id=4, solution=ROOT / 'kernels/004_attention_output_projection_with_reshape_backward/variants/cublaslt/solution.json',
                       label='cublaslt-selected', timeout=600, compile_timeout=600,
                       keep_staging=False, poll_interval=0.5)
report = {'schema_version': 1, 'problem_id': 4, 'started_at': now(),
          'requested_clean_trials': 1, 'max_attempts': 1, 'sampling_seconds': 0.5,
          'scope': 'one queued full correctness trial with unchanged process-window timing audit; no idle prefilter or retries',
          'tool_sha256': hashlib.sha256((ROOT / 'tools/qualify_gpu.py').read_bytes()).hexdigest(), 'attempts': []}
(directory / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
attempt = run_attempt(args, directory, 1)
report['attempts'].append(attempt)
report['aggregate'] = aggregate(report['attempts'])
report['complete'] = attempt['status'] == 'qualified'
report['finished_at'] = now()
(directory / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'qualification': str(directory), 'status': attempt['status'], 'passed': attempt.get('passed'),
                  'raw_score': attempt.get('local_sol_score'), 'foreign_observations': len(attempt.get('audit', {}).get('foreign_context_observations', []))}), flush=True)
