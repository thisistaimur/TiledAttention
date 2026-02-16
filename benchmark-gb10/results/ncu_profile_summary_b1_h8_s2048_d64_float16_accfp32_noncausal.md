# Nsight Compute One-Pass Summary

## Run Configuration

- Shape: B=1, H=8, S=2048, D=64
- dtype: float16
- tiledattention accum_mode: fp32
- causal: False
- warmup: 5, repeats: 1

## Key Metrics (primary kernel row)

| Method | Primary Kernel | Kernel Time (gpu__time_duration.sum) | Tensor Pipe %peak | FP32 FMA Pipe %peak | Achieved Warps Active % | SM Throughput %peak | DRAM/Memory Throughput %peak | L2 Throughput %peak | Long Scoreboard Stall | Math Pipe Stall | Barrier Stall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiledattention | `flash_fwd_kernel_fp32acc_aligned` | 0.2103 | 39.5 | 15.86 | 12.28 | 39.5 | 33.68 | 33.68 | 0.2845 | 0.2078 | 0.01744 |
| torch_sdpa | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...` | 0.2095 | 40.03 | 7.181 | 14.25 | 40.03 | 20.54 | 20.54 | 2.312 | 0.9281 | 0.6808 |

## Quick Diagnosis

### tiledattention

- Primary kernel: `flash_fwd_kernel_fp32acc_aligned`
- Kernel time (gpu__time_duration.sum, normalized to ms): 0.2103
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `notebook/results/ncu_tiledattention_b1_h8_s2048_d64_float16_accfp32_noncausal.ncu-rep`
- Raw CSV: `notebook/results/ncu_tiledattention_b1_h8_s2048_d64_float16_accfp32_noncausal_raw.csv`

### torch_sdpa

- Primary kernel: `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, 0, 0, cutlass::half_t, Flash_kernel_traits<64, 128, 128, 4, cutlass::half_t>>, 0, 0, 0, 0, 1, 1, 0, 0>(pytorch_flash::Flash_fwd_params)`
- Kernel time (gpu__time_duration.sum, normalized to ms): 0.2095
- Low achieved occupancy (<30% of peak active warps).
- Long-scoreboard stalls are high; likely memory-latency / dependency bottleneck.
- Nsight report: `notebook/results/ncu_torch_sdpa_b1_h8_s2048_d64_float16_accfp32_noncausal.ncu-rep`
- Raw CSV: `notebook/results/ncu_torch_sdpa_b1_h8_s2048_d64_float16_accfp32_noncausal_raw.csv`
