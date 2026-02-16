# TiledAttention

`TiledAttention` is the companion library for the paper in this repository:
`tiledattn.pdf` ("TiledAttention: a TileIR SDPA Kernel for PyTorch on CUDA systems").

The goal of this codebase is to make the paper's SDPA kernel and measurement workflow
usable as a practical Python library, not just a one-off artifact.

## What This Library Implements

- PyTorch-facing SDPA API:
  - `sdpa(q, k, v, causal=False, scale=None) -> o`
  - input/output shape: `[B, H, S, D]`
- cuTile Python forward kernel (FlashAttention-style):
  - tiled streaming over `K/V`
  - online softmax updates (running max + normalizer)
  - no materialization of the full `S x S` attention matrix
- Runtime checks for:
  - CUDA availability
  - Blackwell-class GPUs only (`10.x` / `12.x`)
  - cuTile + CuPy availability
- In-memory compile cache keyed by:
  - `(TM, TN, D, dtype, causal)`

## Paper-to-Code Mapping

- Paper API section (`sdpa(q,k,v,...)`):
  - `src/tiledattention/sdpa.py`
- cuTile kernel and online softmax logic:
  - `src/tiledattention/kernels/flash_fwd.py`
- Kernel compilation cache:
  - `src/tiledattention/kernels/compile_cache.py`
- Correctness smoke test against PyTorch SDPA:
  - `tests/test_sdpa_correctness_small.py`

## Current Scope

- Implemented:
  - forward pass SDPA
  - causal and non-causal masking
  - FP16 path validated on host GPU
  - BF16 code path present (not yet fully benchmarked in this repo)
- Not implemented yet:
  - backward pass
  - broad architecture portability beyond Blackwell-class targets
  - full benchmark/tuning CLI surface described in the paper draft

## Requirements

- NVIDIA Blackwell-class GPU (compute capability `10.x` or `12.x`)
- NVIDIA driver with CUDA 13.1 support
- CUDA Toolkit `13.1+` (for `nvcc` / `tileiras`, required by cuTile toolchain)
- Python `>=3.10`
- PyTorch CUDA build, CuPy, and cuTile Python package

Example stack used during development:

- `torch==2.10.0+cu130`
- `torchvision==0.25.0+cu130`
- `cupy-cuda13x==13.6.0`
- `cuda-tile==1.1.0`
- host toolkit: CUDA `13.1`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]" --no-build-isolation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install cupy-cuda13x cuda-tile
```

## Quickstart

```python
import torch
from tiledattention import sdpa

# q, k, v: [B, H, S, D] on CUDA
q = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)

o = sdpa(q, k, v, causal=True)
print(o.shape, o.dtype)
```

## Reproducibility Checks

```bash
# Lint + tests
ruff check src tests scripts
pytest -q

# GPU correctness smoke test (compares to torch SDPA)
pytest -q tests/test_sdpa_correctness_small.py
```

## Tile Defaults and Tuning Note

Current defaults in `src/tiledattention/kernels/flash_fwd.py` are tuned for a representative host (NVIDIA DGX Spark/GB10) run and set to:

- `TM = 64`
- `TN = 64`

These are workload- and platform-dependent. Re-tuning is recommended for your target
`(S, D, dtype, causal)` regimes.

## Performance Modes

The kernel exposes a focused precision/performance switch for the attention accumulator:

- `TILEDATTN_ACCUM_MODE=fp32` (default):
  - safer numerical mode
  - recommended default for correctness-sensitive runs
- `TILEDATTN_ACCUM_MODE=fp16`:
  - optional fast mode
  - can improve throughput in some long-context regimes

Example:

```bash
# safe default
export TILEDATTN_ACCUM_MODE=fp32

# optional fast mode
export TILEDATTN_ACCUM_MODE=fp16
```

Runtime synchronization behavior is controlled by `TILEDATTN_SYNC_MODE`:

- `TILEDATTN_SYNC_MODE=async` (default): do not force per-call stream sync
- `TILEDATTN_SYNC_MODE=post`: synchronize stream after each launch
- `TILEDATTN_SYNC_MODE=strict`: synchronize before and after each launch

## Companion-Paper Positioning

Use this repository as the executable companion to `tiledattn.pdf`:

- same algorithmic kernel family as described in the paper
- explicit implementation hooks for schedule exploration
- direct correctness comparison against framework SDPA

As the paper draft evolves, this README should remain the "how to run the paper artifact"
entry point.
