# Reduced Optimization Summary

Focused optimization sequence:
1. Run async-first configs (no forced per-call sync).
2. Compare focused kernel knobs (tile sizes, opt level, occupancy, accum mode).

- Reduced benchmark shapes: [(1024, 64), (2048, 64), (4096, 128)]
- dtype: float16, causal=False
- Baseline: torch SDPA

## Best Config

- `async_tm64_tn64_fp16acc`
- Mean speedup vs `async_default` (tokens/s): 1.008x

## Ranking (mean speedup vs async_default)
- `async_tm64_tn64_fp16acc`: 1.008x
- `async_tm64_tn64`: 1.004x
- `async_default`: 1.000x
- `async_tm32_tn64_occ2_fp16acc`: 0.769x
- `async_tm32_tn64_occ2`: 0.757x
- `async_tm32_tn128`: 0.700x
- `async_tm32_tn64_opt2`: 0.495x

Detailed CSV: `notebook/results/reduced_optimization_results.csv`
