# Study Summary

This folder contains the benchmark/tuning study outputs to populate the paper tables and figures.

## Baseline

- Baseline used: `torch_sdpa`
- FlashAttention package availability: not installed or unsupported in this environment.

## Outputs

- Benchmark CSV: `benchmark-gb10/results/benchmark_results.csv`
- Benchmark CSV includes latency, tokens/s, TFLOPs/s, and bandwidth proxy columns.
- Tuning CSV: `benchmark-gb10/results/tuning_results.csv`
- Table 3 markdown: `benchmark-gb10/results/table3_reproducibility.md`
- Table 4 markdown: `benchmark-gb10/results/table4_tiling_sensitivity.md`
- Figure 3: `benchmark-gb10/figures/figure3_throughput_vs_s.png`
- Figure 4: `benchmark-gb10/figures/figure4_regime_map.png`
- Figure 5: `benchmark-gb10/figures/figure5_bw_proxy.png`
- FlashAttention-style TFLOPs figure: `benchmark-gb10/figures/figure_fa_style_tflops_fp16.png`
- Explicit baselines TFLOPs figure: `benchmark-gb10/figures/figure6_explicit_baselines_tflops_fp16.png`
