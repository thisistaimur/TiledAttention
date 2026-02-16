# Notebook Study Artifacts

This document is the full runbook for reproducing the paper-study artifacts from:

- `TiledAttention/benchmark-gb10/run_study.py`

The script benchmarks `tiledattention.sdpa`, runs tile tuning, and writes table/figure
artifacts used by the paper draft.

## 1) Prerequisites

- Linux host with NVIDIA GPU (Blackwell-class target for this project)
- NVIDIA driver + CUDA runtime available (`nvidia-smi` must work)
- CUDA toolkit with `nvcc`/`tileiras` available in `PATH`
- Python 3.10+

## 2) Enter Project Directory

```bash
cd TiledAttention
```

## 3) Create/Activate Virtual Environment

If `.venv` does not exist:

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

## 4) Install Dependencies

Install project + dev tools:

```bash
pip install -e ".[dev]" --no-build-isolation
```

Install CUDA PyTorch (current setup in this repo):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Install cuTile + CuPy:

```bash
pip install cupy-cuda13x cuda-tile
```

Install plotting dependency for study figures:

```bash
pip install matplotlib
```

## 5) Sanity Checks Before Study

Check GPU visibility:

```bash
nvidia-smi
```

Check Python stack:

```bash
python - <<'PY'
import torch, cupy, cuda.tile as ct
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
print("cupy", cupy.__version__)
print("cuda.tile", ct.__version__)
PY
```

Optional code health checks:

```bash
ruff check src tests notebook
pytest -q
```

## 6) Run Full Study

Default run (used for full tables/figures):

```bash
python benchmark-gb10/run_study.py
```

This executes:

- benchmark sweep over:
  - `S = [512, 1024, 2048, 4096, 8192]`
  - `D = [64, 96, 128, 160]`
  - dtypes: `float16`, `bfloat16`
  - masks: causal + non-causal
- tuning sweep for representative regimes (`S=1024,4096,8192`, `D=128`, fp16)
- figure/table generation

Approximate runtime: several minutes (depends on host/GPU load).

## 7) Optional Study Flags

Run with custom timing parameters:

```bash
python benchmark-gb10/run_study.py --warmup 3 --iters 10
```

Run non-causal only:

```bash
python benchmark-gb10/run_study.py --no-causal
```

Kernel behavior can also be overridden via environment variables:

```bash
export TILEDATTN_SYNC_MODE=async           # strict | post | async
export TILEDATTN_ACCUM_MODE=fp32           # fp32 | fp16
# Optional: pin tile sizes. If unset, runtime auto-policy is used.
export TILEDATTN_TILE_M=64
export TILEDATTN_TILE_N=64
export TILEDATTN_KERNEL_OPT_LEVEL=3
export TILEDATTN_KERNEL_OCCUPANCY=2        # optional
export TILEDATTN_KERNEL_NUM_CTAS=2         # optional
# Optional experimental path (off by default):
# export TILEDATTN_CHUNKED_HEAD_DIMS=96,160
```

## 8) Output Files

Study outputs are written to:

- `benchmark-gb10/results/benchmark_results.csv`
- `benchmark-gb10/results/tuning_results.csv`
- `benchmark-gb10/results/table3_reproducibility.md`
- `benchmark-gb10/results/table4_tiling_sensitivity.md`
- `benchmark-gb10/results/study_summary.md`
- `benchmark-gb10/figures/figure3_throughput_vs_s.png`
- `benchmark-gb10/figures/figure4_regime_map.png`
- `benchmark-gb10/figures/figure5_bw_proxy.png`
- `benchmark-gb10/figures/figure_fa_style_tflops_fp16.png`
- `benchmark-gb10/figures/figure6_explicit_baselines_tflops_fp16.png`

Quick inspection commands:

```bash
sed -n '1,120p' benchmark-gb10/results/table3_reproducibility.md
sed -n '1,120p' benchmark-gb10/results/table4_tiling_sensitivity.md
head -n 5 benchmark-gb10/results/benchmark_results.csv
head -n 5 benchmark-gb10/results/tuning_results.csv
```

## 9) Baseline Note

If `flash_attn` is not installed, the study uses PyTorch SDPA (`torch_sdpa`) as baseline.
This is recorded automatically in:

- `benchmark-gb10/results/study_summary.md`
- `benchmark-gb10/results/table3_reproducibility.md`

## 10) Focused Optimization Pass (Reduced Benchmark)

This runs a quick optimization loop:

1. Async-first runtime configs (no forced per-call sync)
2. Focused knob pass (`TM/TN`, `opt_level`, `occupancy`, `accum_mode`)

Command:

```bash
python benchmark-gb10/run_reduced_optimization.py
```

Outputs:

- `benchmark-gb10/results/reduced_optimization_results.csv`
- `benchmark-gb10/results/reduced_optimization_summary.md`

Current reduced benchmark uses:

- Shapes: `(1024,64)`, `(2048,64)`, `(4096,128)`
- `dtype=float16`, `causal=False`
- Baseline: `torch_sdpa`

## 11) Nsight Compute One-Pass Bottleneck Profiling

Use this when throughput is unexpectedly low and you want a direct signal on:

- tensor-core use vs FP32-heavy execution
- achieved occupancy
- compute vs memory saturation
- major scheduler stall classes

Command (default shape: `B=1,H=8,S=4096,D=128`, fp16, non-causal):

```bash
python benchmark-gb10/run_ncu_profile.py
```

Recommended command for current GB10 issue triage:

```bash
python benchmark-gb10/run_ncu_profile.py \
  --batch 1 \
  --heads 8 \
  --seq-len 4096 \
  --head-dim 128 \
  --dtype float16 \
  --accum-mode fp32 \
  --warmup 5 \
  --repeats 1 \
  --sync-mode async \
  --tile-m 64 \
  --tile-n 64 \
  --occupancy 2
```

Compare accumulator modes (safe default vs fast mode):

```bash
# fp32 accumulator (default/safe)
sudo -E TiledAttention/.venv/bin/python \
  benchmark-gb10/run_ncu_profile.py \
  --ncu-path /usr/local/cuda/bin/ncu \
  --output-dir /tmp/tiledattention_ncu_fp32 \
  --batch 1 --heads 8 --seq-len 4096 --head-dim 128 \
  --dtype float16 --accum-mode fp32 --warmup 5 --repeats 1 \
  --sync-mode async --tile-m 64 --tile-n 64 --occupancy 2

# fp16 accumulator (optional fast mode)
sudo -E TiledAttention/.venv/bin/python \
  benchmark-gb10/run_ncu_profile.py \
  --ncu-path /usr/local/cuda/bin/ncu \
  --output-dir /tmp/tiledattention_ncu_fp16 \
  --batch 1 --heads 8 --seq-len 4096 --head-dim 128 \
  --dtype float16 --accum-mode fp16 --warmup 5 --repeats 1 \
  --sync-mode async --tile-m 64 --tile-n 64 --occupancy 2
```

Note on kernel duration units:

- Nsight raw CSV can use mixed units (`ms` in one report, `us` in another).
- `benchmark-gb10/run_ncu_profile.py` normalizes kernel time to milliseconds in the summary table.

Outputs:

- Nsight reports:
  - `benchmark-gb10/results/ncu_tiledattention_*.ncu-rep`
  - `benchmark-gb10/results/ncu_torch_sdpa_*.ncu-rep`
- Raw CSV exports:
  - `benchmark-gb10/results/ncu_tiledattention_*_raw.csv`
  - `benchmark-gb10/results/ncu_torch_sdpa_*_raw.csv`
- Auto-summary:
  - `benchmark-gb10/results/ncu_profile_summary_*_acc*.md`

Open a report in UI (optional):

```bash
ncu-ui benchmark-gb10/results/ncu_tiledattention_*.ncu-rep
```
