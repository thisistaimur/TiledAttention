# Nsight Compute One-Pass Summary

## Run Configuration

- Shape: B=1, H=8, S=4096, D=128
- dtype: float16
- causal: False
- warmup: 5, repeats: 1

## Key Metrics (primary kernel row)

| Method | Primary Kernel | Kernel Time (gpu__time_duration.sum) | Tensor Pipe %peak | FP32 FMA Pipe %peak | Achieved Warps Active % | SM Throughput %peak | DRAM/Memory Throughput %peak | L2 Throughput %peak | Long Scoreboard Stall | Math Pipe Stall | Barrier Stall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiledattention | `flash_fwd_kernel_fp16acc` | 0.977 | 67.58 | 13.49 | 22.39 | 67.58 | 56.73 | 56.73 | 0.5625 | 2.069 | 0.004287 |
| torch_sdpa | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...` | 1.098 | 59.85 | 6.919 | 8.511 | 59.85 | 30.03 | 30.03 | 1.364 | 0.000816 | 0.09814 |

## Quick Diagnosis

### tiledattention

- Primary kernel: `flash_fwd_kernel_fp16acc`
- Kernel time (gpu__time_duration.sum, normalized to ms): 0.977
- Low achieved occupancy (<30% of peak active warps).
- Math-pipe throttle is high; check instruction mix and issue balance.
- Nsight report: `notebook/results/ncu_tiledattention_b1_h8_s4096_d128_float16_noncausal.ncu-rep`
- Raw CSV: `notebook/results/ncu_tiledattention_b1_h8_s4096_d128_float16_noncausal_raw.csv`

### torch_sdpa

- Primary kernel: `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, cutlass::half_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::half_t>>, 0, 0, 0, 0, 1, 1, 0, 0>(pytorch_flash::Flash_fwd_params)`
- Kernel time (gpu__time_duration.sum, normalized to ms): 1.098
- Low achieved occupancy (<30% of peak active warps).
- Nsight report: `notebook/results/ncu_torch_sdpa_b1_h8_s4096_d128_float16_noncausal.ncu-rep`
- Raw CSV: `notebook/results/ncu_torch_sdpa_b1_h8_s4096_d128_float16_noncausal_raw.csv`
