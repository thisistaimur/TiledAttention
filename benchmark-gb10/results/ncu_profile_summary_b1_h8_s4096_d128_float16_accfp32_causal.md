# Nsight Compute One-Pass Summary

## Run Configuration

- Shape: B=1, H=8, S=4096, D=128
- dtype: float16
- tiledattention accum_mode: fp32
- causal: True
- warmup: 5, repeats: 1

## Key Metrics (primary kernel row)

| Method | Primary Kernel | Kernel Time (gpu__time_duration.sum) | Tensor Pipe %peak | FP32 FMA Pipe %peak | Achieved Warps Active % | SM Throughput %peak | DRAM/Memory Throughput %peak | L2 Throughput %peak | Long Scoreboard Stall | Math Pipe Stall | Barrier Stall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiledattention | `flash_fwd_kernel_fp32acc` | 0.6855 | 48.72 | 12.08 | 12.44 | 48.72 | 41.61 | 41.61 | 0.2364 | 0.4822 | 0.00624 |
| torch_sdpa | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...` | 0.6124 | 55.23 | 6.391 | 8.226 | 55.23 | 24.56 | 24.56 | 0.4289 | 0.000859 | 0.08622 |
| torch_sdpa_flash_forced | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...` | 0.5912 | 57.28 | 6.628 | 8.021 | 57.28 | 25.45 | 25.45 | 0.4122 | 0.000866 | 0.05943 |

## Quick Diagnosis

### tiledattention

- Primary kernel: `flash_fwd_kernel_fp32acc`
- Kernel time (gpu__time_duration.sum, normalized to ms): 0.6855
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `benchmark-gb10/results/ncu_tiledattention_b1_h8_s4096_d128_float16_accfp32_causal.ncu-rep`
- Raw CSV: `benchmark-gb10/results/ncu_tiledattention_b1_h8_s4096_d128_float16_accfp32_causal_raw.csv`

### torch_sdpa

- Primary kernel: `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, cutlass::half_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::half_t>>, 0, 1, 0, 0, 1, 1, 0, 0>(pytorch_flash::Flash_fwd_params)`
- Kernel time (gpu__time_duration.sum, normalized to ms): 0.6124
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `benchmark-gb10/results/ncu_torch_sdpa_b1_h8_s4096_d128_float16_accfp32_causal.ncu-rep`
- Raw CSV: `benchmark-gb10/results/ncu_torch_sdpa_b1_h8_s4096_d128_float16_accfp32_causal_raw.csv`

### torch_sdpa_flash_forced

- Primary kernel: `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, cutlass::half_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::half_t>>, 0, 1, 0, 0, 1, 1, 0, 0>(pytorch_flash::Flash_fwd_params)`
- Kernel time (gpu__time_duration.sum, normalized to ms): 0.5912
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `benchmark-gb10/results/ncu_torch_sdpa_flash_forced_b1_h8_s4096_d128_float16_accfp32_causal.ncu-rep`
- Raw CSV: `benchmark-gb10/results/ncu_torch_sdpa_flash_forced_b1_h8_s4096_d128_float16_accfp32_causal_raw.csv`
