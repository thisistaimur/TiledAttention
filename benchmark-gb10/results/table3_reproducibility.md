# Table 3 - Reproducibility Checklist

| Item | Value |
| --- | --- |
| System | Host GPU: NVIDIA GB10 |
| GPU driver | 580.126.09 |
| CUDA toolkit | Cuda compilation tools, release 13.1, V13.1.115 |
| tileiras | tileiras: NVIDIA (R) Cuda Tile IR optimizing assembler |
| cuTile | 1.1.0 |
| PyTorch | 2.10.0+cu130 |
| CuPy | 13.6.0 |
| FlashAttention version | Not installed (baseline used: torch SDPA) |
| Timing method | CUDA events, median + p95 after warmup |
