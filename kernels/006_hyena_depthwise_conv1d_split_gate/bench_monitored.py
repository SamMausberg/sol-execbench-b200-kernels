"""Run the official campaign while recording GPU process ownership."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def descendant(pid, ancestor):
    for _ in range(32):
        if pid == ancestor:
            return True
        if pid < 2:
            return False
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            pid = int(stat[1])
        except (OSError, ValueError, IndexError):
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args, campaign_args = parser.parse_known_args()
    command = [sys.executable, "tools/campaign.py", "bench", "6", *campaign_args]
    active = threading.Event()
    lines = []
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)

    def read_output():
        for line in child.stdout:
            print(line, end="", flush=True)
            lines.append(line.rstrip())
            if "GPU lock acquired" in line:
                active.set()
            if line.startswith("Summary:"):
                active.clear()

    reader = threading.Thread(target=read_output)
    reader.start()
    samples = []
    while True:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        processes = []
        for row in completed.stdout.splitlines():
            pid_text, process_name, memory = row.split(",", 2)
            pid = int(pid_text)
            processes.append({"pid": pid, "process_basename": Path(process_name.strip()).name,
                              "memory_mib": memory.strip(),
                              "belongs_to_campaign": descendant(pid, child.pid),
                              })
        samples.append({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "gpu_lock_acquired": active.is_set(), "processes": processes})
        if child.poll() is not None:
            break
        time.sleep(0.5)
    reader.join()
    foreign = [sample for sample in samples if sample["gpu_lock_acquired"] and
               any(process["belongs_to_campaign"] is not True for process in sample["processes"])]
    report = {"monitor_pid": os.getpid(), "campaign_pid": child.pid,
              "return_code": child.returncode, "observed_foreign_process_samples": len(foreign),
              "samples": samples}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"GPU process report: {args.output}; foreign samples after lock: {len(foreign)}", flush=True)
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
