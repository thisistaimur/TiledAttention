# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "kernels",
#     "torch",
# ]
# ///

from pathlib import Path

import kernels
import torch

# Load the locally built kernel package from ./build
kernel = kernels.get_local_kernel(Path("build"), "tiledattention")

device = torch.device("cuda")

torch.manual_seed(0)
q = torch.randn(1, 2, 64, 64, device=device, dtype=torch.float16)
k = torch.randn(1, 2, 64, 64, device=device, dtype=torch.float16)
v = torch.randn(1, 2, 64, 64, device=device, dtype=torch.float16)

out = kernel.sdpa(q, k, v, causal=True)
print("Output shape:", out.shape)
print("Output dtype:", out.dtype)
