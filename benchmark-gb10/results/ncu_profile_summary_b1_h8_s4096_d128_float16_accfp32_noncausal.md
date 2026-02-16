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
| tiledattention | `flash_fwd_kernel_fp32acc_aligned` | 1.176 | 55.99 | 12.88 | 12.22 | 55.99 | 47.05 | 47.05 | 0.2657 | 0.5606 | 0.003779 |
| torch_sdpa | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...` | 1.103 | 59.7 | 6.901 | 7.88 | 59.7 | 29.9 | 29.9 | 0.6268 | 0.000816 | 0.3002 |

## Quick Diagnosis

### tiledattention

- Primary kernel: `flash_fwd_kernel_fp32acc_aligned`
- Kernel time (gpu__time_duration.sum, normalized to ms): 1.176
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `benchmark-gb10/results/ncu_tiledattention_b1_h8_s4096_d128_float16_accfp32_noncausal.ncu-rep`
- Raw CSV: `benchmark-gb10/results/ncu_tiledattention_b1_h8_s4096_d128_float16_accfp32_noncausal_raw.csv`

### torch_sdpa

- Primary kernel: `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, cutlass::half_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::half_t>>, 0, 0, 0, 0, 1, 1, 0, 0>(pytorch_flash::Flash_fwd_params)`
- Kernel time (gpu__time_duration.sum, normalized to ms): 1.103
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `benchmark-gb10/results/ncu_torch_sdpa_b1_h8_s4096_d128_float16_accfp32_noncausal.ncu-rep`
- Raw CSV: `benchmark-gb10/results/ncu_torch_sdpa_b1_h8_s4096_d128_float16_accfp32_noncausal_raw.csv`
