# Nsight Compute One-Pass Summary

## Run Configuration

- Shape: B=1, H=8, S=4096, D=128
- dtype: float16
- tiledattention accum_mode: fp32
- causal: False
- warmup: 5, repeats: 1

## Key Metrics (primary kernel row)

| Method | Primary Kernel | Kernel Time (gpu__time_duration.sum) | Tensor Pipe %peak | FP32 FMA Pipe %peak | Achieved Warps Active % | SM Throughput %peak | DRAM/Memory Throughput %peak | L2 Throughput %peak | Long Scoreboard Stall | Math Pipe Stall | Barrier Stall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiledattention | `flash_fwd_kernel_fp32acc_aligned` | 1.203 | 54.63 | 12.57 | 12.97 | 54.63 | 45.97 | 45.97 | 0.2713 | 0.5606 | 0.00378 |
| torch_sdpa | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...` | 1.132 | 58.26 | 6.735 | 8.43 | 58.26 | 29.19 | 29.19 | 0.4278 | 0.000816 | 0.1837 |
| torch_sdpa_flash_forced | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...` | 1.131 | 58.29 | 6.738 | 8.364 | 58.29 | 29.19 | 29.19 | 1.402 | 0.000816 | 0.1107 |

## Quick Diagnosis

### tiledattention

- Primary kernel: `flash_fwd_kernel_fp32acc_aligned`
- Kernel time (gpu__time_duration.sum, normalized to ms): 1.203
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `benchmark-gb10/results/ncu_tiledattention_b1_h8_s4096_d128_float16_accfp32_noncausal.ncu-rep`
- Raw CSV: `benchmark-gb10/results/ncu_tiledattention_b1_h8_s4096_d128_float16_accfp32_noncausal_raw.csv`

### torch_sdpa

- Primary kernel: `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, cutlass::half_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::half_t>>, 0, 0, 0, 0, 1, 1, 0, 0>(pytorch_flash::Flash_fwd_params)`
- Kernel time (gpu__time_duration.sum, normalized to ms): 1.132
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `benchmark-gb10/results/ncu_torch_sdpa_b1_h8_s4096_d128_float16_accfp32_noncausal.ncu-rep`
- Raw CSV: `benchmark-gb10/results/ncu_torch_sdpa_b1_h8_s4096_d128_float16_accfp32_noncausal_raw.csv`

### torch_sdpa_flash_forced

- Primary kernel: `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, cutlass::half_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::half_t>>, 0, 0, 0, 0, 1, 1, 0, 0>(pytorch_flash::Flash_fwd_params)`
- Kernel time (gpu__time_duration.sum, normalized to ms): 1.131
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `benchmark-gb10/results/ncu_torch_sdpa_flash_forced_b1_h8_s4096_d128_float16_accfp32_noncausal.ncu-rep`
- Raw CSV: `benchmark-gb10/results/ncu_torch_sdpa_flash_forced_b1_h8_s4096_d128_float16_accfp32_noncausal_raw.csv`
