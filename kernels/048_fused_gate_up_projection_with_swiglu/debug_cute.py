# SPDX-License-Identifier: Apache-2.0
"""One launch for native sanitizer diagnostics; invoke under the shared flock."""
import torch
from cute_kernel import configured_run

torch.manual_seed(48)
x=torch.randn((1,128,3072),dtype=torch.bfloat16,device="cuda")
gate=torch.randn((24576,3072),dtype=torch.bfloat16,device="cuda")
up=torch.randn_like(gate)
out=torch.empty((1,128,24576),dtype=torch.bfloat16,device="cuda")
configured_run(x,gate,up,out)
torch.cuda.synchronize()
print("Native launch finished",flush=True)
