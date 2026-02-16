# Reduced Optimization Summary

Focused optimization sequence:
1. Run async-first configs (no forced per-call sync).
2. Compare focused kernel knobs (tile sizes, opt level, occupancy, accum mode).

- Reduced benchmark shapes: [(1024, 64), (2048, 64), (4096, 128)]
- dtype: float16, causal=False
- Baseline: torch SDPA

## Best Config

- `async_default`
- Mean speedup vs `async_default` (tokens/s): 1.000x

## Ranking (mean speedup vs async_default)
- `async_default`: 1.000x
- `async_tm64_tn64`: 0.973x
- `async_tm64_tn64_fp16acc`: 0.971x
- `async_tm32_tn64_occ2_fp16acc`: 0.737x
- `async_tm32_tn64_occ2`: 0.732x
- `async_tm32_tn128`: 0.678x
- `async_tm32_tn64_opt2`: 0.479x

Detailed CSV: `notebook/results/reduced_optimization_results.csv`
