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
| tiledattention | `flash_fwd_kernel_fp32acc` | 0.6696 | 49.84 | 12.36 | 11.79 | 49.84 | 42.57 | 42.57 | 0.2362 | 0.482 | 0.006238 |
| torch_sdpa | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...` | 0.6302 | 53.66 | 6.209 | 8.283 | 53.66 | 23.84 | 23.84 | 0.581 | 0.000863 | 0.06226 |

## Quick Diagnosis

### tiledattention

- Primary kernel: `flash_fwd_kernel_fp32acc`
- Kernel time (gpu__time_duration.sum, normalized to ms): 0.6696
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `benchmark-gb10/results/ncu_tiledattention_b1_h8_s4096_d128_float16_accfp32_causal.ncu-rep`
- Raw CSV: `benchmark-gb10/results/ncu_tiledattention_b1_h8_s4096_d128_float16_accfp32_causal_raw.csv`

### torch_sdpa

- Primary kernel: `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, cutlass::half_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::half_t>>, 0, 1, 0, 0, 1, 1, 0, 0>(pytorch_flash::Flash_fwd_params)`
- Kernel time (gpu__time_duration.sum, normalized to ms): 0.6302
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `benchmark-gb10/results/ncu_torch_sdpa_b1_h8_s4096_d128_float16_accfp32_causal.ncu-rep`
- Raw CSV: `benchmark-gb10/results/ncu_torch_sdpa_b1_h8_s4096_d128_float16_accfp32_causal_raw.csv`
