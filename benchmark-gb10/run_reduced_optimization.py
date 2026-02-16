from __future__ import annotations

import csv
import os
import statistics
from pathlib import Path

import torch

from tiledattention import sdpa
from tiledattention.kernels.compile_cache import reset_cache_for_tests


def time_cuda_callable(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    vals: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        vals.append(float(start.elapsed_time(end)))
    if len(vals) == 1:
        return vals[0], vals[0]
    return statistics.median(vals), statistics.quantiles(vals, n=20)[18]


def throughput_tokens_per_s(*, b: int, h: int, s: int, median_ms: float) -> float:
    return (b * h * s) / (median_ms / 1000.0)


def attention_forward_flops(*, b: int, h: int, s: int, d: int, causal: bool) -> float:
    if causal:
        return 2.0 * b * h * s * (s + 1) * d
    return 4.0 * b * h * s * s * d


def tflops_per_s(*, flops: float, median_ms: float) -> float:
    return flops / (median_ms / 1000.0) / 1e12


def apply_env(overrides: dict[str, str | None], keys: list[str]) -> dict[str, str | None]:
    prev = {k: os.environ.get(k) for k in keys}
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return prev


def restore_env(snapshot: dict[str, str | None]) -> None:
    for k, v in snapshot.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for reduced optimization benchmark.")

    device = torch.device("cuda")
    torch.manual_seed(2026)

    results_dir = Path("benchmark-gb10/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    b, h = 1, 8
    shapes = [(1024, 64), (2048, 64), (4096, 128)]
    dtype = torch.float16
    causal = False
    warmup = 4
    iters = 12

    managed_keys = [
        "TILEDATTN_SYNC_MODE",
        "TILEDATTN_ACCUM_MODE",
        "TILEDATTN_TILE_M",
        "TILEDATTN_TILE_N",
        "TILEDATTN_KERNEL_OPT_LEVEL",
        "TILEDATTN_KERNEL_OCCUPANCY",
        "TILEDATTN_KERNEL_NUM_CTAS",
    ]

    baseline_config = "async_auto"
    configs = [
        {
            "name": "async_auto",
            "env": {
                "TILEDATTN_SYNC_MODE": "async",
                "TILEDATTN_ACCUM_MODE": None,
                "TILEDATTN_TILE_M": None,
                "TILEDATTN_TILE_N": None,
                "TILEDATTN_KERNEL_OPT_LEVEL": None,
                "TILEDATTN_KERNEL_OCCUPANCY": None,
                "TILEDATTN_KERNEL_NUM_CTAS": None,
            },
        },
        {
            "name": "async_default_fp32",
            "env": {
                "TILEDATTN_SYNC_MODE": "async",
                "TILEDATTN_ACCUM_MODE": "fp32",
                "TILEDATTN_TILE_M": None,
                "TILEDATTN_TILE_N": None,
                "TILEDATTN_KERNEL_OPT_LEVEL": None,
                "TILEDATTN_KERNEL_OCCUPANCY": None,
                "TILEDATTN_KERNEL_NUM_CTAS": None,
            },
        },
        {
            "name": "async_tm64_tn64",
            "env": {
                "TILEDATTN_SYNC_MODE": "async",
                "TILEDATTN_ACCUM_MODE": "fp32",
                "TILEDATTN_TILE_M": "64",
                "TILEDATTN_TILE_N": "64",
                "TILEDATTN_KERNEL_OPT_LEVEL": "3",
                "TILEDATTN_KERNEL_OCCUPANCY": None,
                "TILEDATTN_KERNEL_NUM_CTAS": None,
            },
        },
        {
            "name": "async_tm32_tn64_occ2",
            "env": {
                "TILEDATTN_SYNC_MODE": "async",
                "TILEDATTN_ACCUM_MODE": "fp32",
                "TILEDATTN_TILE_M": "32",
                "TILEDATTN_TILE_N": "64",
                "TILEDATTN_KERNEL_OPT_LEVEL": "3",
                "TILEDATTN_KERNEL_OCCUPANCY": "2",
                "TILEDATTN_KERNEL_NUM_CTAS": None,
            },
        },
        {
            "name": "async_tm32_tn64_opt2",
            "env": {
                "TILEDATTN_SYNC_MODE": "async",
                "TILEDATTN_ACCUM_MODE": "fp32",
                "TILEDATTN_TILE_M": "32",
                "TILEDATTN_TILE_N": "64",
                "TILEDATTN_KERNEL_OPT_LEVEL": "2",
                "TILEDATTN_KERNEL_OCCUPANCY": None,
                "TILEDATTN_KERNEL_NUM_CTAS": None,
            },
        },
        {
            "name": "async_tm32_tn128",
            "env": {
                "TILEDATTN_SYNC_MODE": "async",
                "TILEDATTN_ACCUM_MODE": "fp32",
                "TILEDATTN_TILE_M": "32",
                "TILEDATTN_TILE_N": "128",
                "TILEDATTN_KERNEL_OPT_LEVEL": "3",
                "TILEDATTN_KERNEL_OCCUPANCY": None,
                "TILEDATTN_KERNEL_NUM_CTAS": None,
            },
        },
        {
            "name": "async_tm64_tn64_fp16acc",
            "env": {
                "TILEDATTN_SYNC_MODE": "async",
                "TILEDATTN_ACCUM_MODE": "fp16",
                "TILEDATTN_TILE_M": "64",
                "TILEDATTN_TILE_N": "64",
                "TILEDATTN_KERNEL_OPT_LEVEL": "3",
                "TILEDATTN_KERNEL_OCCUPANCY": None,
                "TILEDATTN_KERNEL_NUM_CTAS": None,
            },
        },
        {
            "name": "async_tm32_tn64_occ2_fp16acc",
            "env": {
                "TILEDATTN_SYNC_MODE": "async",
                "TILEDATTN_ACCUM_MODE": "fp16",
                "TILEDATTN_TILE_M": "32",
                "TILEDATTN_TILE_N": "64",
                "TILEDATTN_KERNEL_OPT_LEVEL": "3",
                "TILEDATTN_KERNEL_OCCUPANCY": "2",
                "TILEDATTN_KERNEL_NUM_CTAS": None,
            },
        },
    ]

    # Baseline torch timings by shape.
    baseline_by_shape: dict[tuple[int, int], dict[str, float]] = {}
    tensors: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for s, d in shapes:
        q = torch.randn((b, h, s, d), device=device, dtype=dtype)
        k = torch.randn((b, h, s, d), device=device, dtype=dtype)
        v = torch.randn((b, h, s, d), device=device, dtype=dtype)
        tensors[(s, d)] = (q, k, v)

        median_ms, p95_ms = time_cuda_callable(
            lambda q=q, k=k, v=v: torch_sdpa(q, k, v, causal=causal),
            warmup=warmup,
            iters=iters,
        )
        flops = attention_forward_flops(b=b, h=h, s=s, d=d, causal=causal)
        baseline_by_shape[(s, d)] = {
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "tokens_per_s": throughput_tokens_per_s(b=b, h=h, s=s, median_ms=median_ms),
            "tflops_per_s": tflops_per_s(flops=flops, median_ms=median_ms),
        }

    rows: list[dict[str, str | int | float]] = []

    for cfg in configs:
        snapshot = apply_env(cfg["env"], managed_keys)
        reset_cache_for_tests()

        # Quick correctness guard on smallest shape.
        s0, d0 = shapes[0]
        q0, k0, v0 = tensors[(s0, d0)]
        out = sdpa(q0, k0, v0, causal=causal)
        ref = torch_sdpa(q0, k0, v0, causal=causal)
        torch.cuda.synchronize()
        if not torch.allclose(out, ref, atol=3e-2, rtol=3e-2):
            restore_env(snapshot)
            raise RuntimeError(f"Config {cfg['name']} failed correctness check.")

        for s, d in shapes:
            q, k, v = tensors[(s, d)]
            median_ms, p95_ms = time_cuda_callable(
                lambda q=q, k=k, v=v: sdpa(q, k, v, causal=causal),
                warmup=warmup,
                iters=iters,
            )
            flops = attention_forward_flops(b=b, h=h, s=s, d=d, causal=causal)
            tokens_per_s = throughput_tokens_per_s(b=b, h=h, s=s, median_ms=median_ms)
            tflops = tflops_per_s(flops=flops, median_ms=median_ms)

            base = baseline_by_shape[(s, d)]
            ratio_to_torch = tokens_per_s / float(base["tokens_per_s"])

            rows.append(
                {
                    "config": cfg["name"],
                    "B": b,
                    "H": h,
                    "S": s,
                    "D": d,
                    "dtype": "float16",
                    "causal": causal,
                    "median_ms": median_ms,
                    "p95_ms": p95_ms,
                    "tokens_per_s": tokens_per_s,
                    "tflops_per_s": tflops,
                    "baseline_torch_median_ms": float(base["median_ms"]),
                    "baseline_torch_tokens_per_s": float(base["tokens_per_s"]),
                    "ratio_to_torch_tokens": ratio_to_torch,
                }
            )
            print(
                f"[opt] {cfg['name']} S={s} D={d} median={median_ms:.3f}ms "
                f"ratio_to_torch={ratio_to_torch:.4f}"
            )

        restore_env(snapshot)

    baseline_rows = [r for r in rows if r["config"] == baseline_config]
    baseline_by_shape_cfg = {(int(r["S"]), int(r["D"])): float(r["tokens_per_s"]) for r in baseline_rows}
    for row in rows:
        baseline_tps = baseline_by_shape_cfg[(int(row["S"]), int(row["D"]))]
        row["speedup_vs_baseline_tokens"] = float(row["tokens_per_s"]) / baseline_tps

    out_csv = results_dir / "reduced_optimization_results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_cfg: dict[str, list[float]] = {}
    for row in rows:
        by_cfg.setdefault(str(row["config"]), []).append(float(row["speedup_vs_baseline_tokens"]))
    sorted_cfg = sorted(by_cfg.items(), key=lambda kv: statistics.mean(kv[1]), reverse=True)
    best_cfg, best_vals = sorted_cfg[0]

    summary_path = results_dir / "reduced_optimization_summary.md"
    lines = [
        "# Reduced Optimization Summary",
        "",
        "Focused optimization sequence:",
        "1. Run async-first configs (no forced per-call sync).",
        "2. Compare focused kernel knobs (tile sizes, opt level, occupancy, accum mode).",
        "",
        f"- Reduced benchmark shapes: {shapes}",
        "- dtype: float16, causal=False",
        "- Baseline: torch SDPA",
        "",
        "## Best Config",
        "",
        f"- `{best_cfg}`",
        f"- Mean speedup vs `{baseline_config}` (tokens/s): {statistics.mean(best_vals):.3f}x",
        "",
        f"## Ranking (mean speedup vs {baseline_config})",
    ]
    for cfg_name, vals in sorted_cfg:
        lines.append(f"- `{cfg_name}`: {statistics.mean(vals):.3f}x")
    lines.append("")
    lines.append(f"Detailed CSV: `{out_csv}`")
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"[opt] wrote {out_csv}")
    print(f"[opt] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
