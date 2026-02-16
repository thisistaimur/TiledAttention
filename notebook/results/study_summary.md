# Study Summary

This folder contains the benchmark/tuning study outputs to populate the paper tables and figures.

## Baseline

- Baseline used: `torch_sdpa`
- FlashAttention package availability: not installed or unsupported in this environment.

## Outputs

- Benchmark CSV: `notebook/results/benchmark_results.csv`
- Benchmark CSV includes latency, tokens/s, TFLOPs/s, and bandwidth proxy columns.
- Tuning CSV: `notebook/results/tuning_results.csv`
- Table 3 markdown: `notebook/results/table3_reproducibility.md`
- Table 4 markdown: `notebook/results/table4_tiling_sensitivity.md`
- Figure 3: `notebook/figures/figure3_throughput_vs_s.png`
- Figure 4: `notebook/figures/figure4_regime_map.png`
- Figure 5: `notebook/figures/figure5_bw_proxy.png`
- FlashAttention-style TFLOPs figure: `notebook/figures/figure_fa_style_tflops_fp16.png`
- Explicit baselines TFLOPs figure: `notebook/figures/figure6_explicit_baselines_tflops_fp16.png`
