from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from tiledattention import sdpa
from tiledattention._runtime import get_cupy_module, get_cutile_module, get_torch_module
from tiledattention.kernels.flash_fwd import _ct_dtype_for_torch_dtype, make_flashattn_fwd_kernel

AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 12
ANNOTATION_FONTSIZE = 10
FIG_DPI = 300


def p95(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20)[18]


def time_cuda_callable(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))

    return statistics.median(timings), p95(timings)


def throughput_tokens_per_s(*, b: int, h: int, s: int, median_ms: float) -> float:
    return (b * h * s) / (median_ms / 1000.0)


def attention_forward_flops(*, b: int, h: int, s: int, d: int, causal: bool) -> float:
    if causal:
        return 2.0 * b * h * s * (s + 1) * d
    return 4.0 * b * h * s * s * d


def tflops_per_s(*, flops: float, median_ms: float) -> float:
    return flops / (median_ms / 1000.0) / 1e12


def approximate_bw_gbps(*, b: int, h: int, s: int, d: int, bytes_per_elem: int, median_ms: float) -> float:
    bytes_moved = 4.0 * b * h * s * d * bytes_per_elem
    seconds = median_ms / 1000.0
    return bytes_moved / seconds / 1e9


def _style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)


def _save_figure(fig, path: Path) -> None:
    fig.tight_layout(pad=0.25)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.02)


def maybe_flash_baseline():
    try:
        from flash_attn import flash_attn_func as fa_func

        return fa_func
    except Exception:
        pass

    try:
        from flash_attn.flash_attn_interface import flash_attn_func as fa_func

        return fa_func
    except Exception:
        return None


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    flash_attn_func: Callable[..., torch.Tensor],
) -> torch.Tensor:
    # flash-attn expects [B, S, H, D], whereas this harness uses [B, H, S, D].
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()
    out_fa = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=0.0, causal=causal)
    return out_fa.transpose(1, 2)


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal
    )


def torch_sdpa_math(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
        enable_cudnn=False,
    ):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal
        )


def standard_eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    causal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        if causal_mask is None:
            s = scores.shape[-1]
            causal_mask = torch.ones((1, 1, s, s), device=scores.device, dtype=torch.bool).triu(1)
        neg = torch.tensor(torch.finfo(scores.dtype).min, dtype=scores.dtype, device=scores.device)
        scores = torch.where(causal_mask, neg, scores)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


def time_cuda_callable_safe(fn, *, warmup: int, iters: int, method: str) -> tuple[float, float]:
    try:
        return time_cuda_callable(fn, warmup=warmup, iters=iters)
    except (TypeError, ValueError) as exc:
        if method == "flashattention":
            print(f"[bench] warning: flashattention unsupported for this shape/dtype ({exc}); writing NaN metrics")
            return math.nan, math.nan
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            print(f"[bench] warning: OOM while benchmarking {method}; writing NaN metrics")
            return math.nan, math.nan
        if method == "flashattention":
            print(f"[bench] warning: flashattention runtime failure ({exc}); writing NaN metrics")
            return math.nan, math.nan
        raise


def collect_repro_info() -> dict[str, str]:
    info: dict[str, str] = {}

    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()[0]
        gpu_name, driver = [x.strip() for x in line.split(",", maxsplit=1)]
        info["system"] = f"Host GPU: {gpu_name}"
        info["driver"] = driver
    except Exception as exc:  # pragma: no cover
        info["system"] = f"Unknown ({exc})"
        info["driver"] = "Unknown"

    try:
        nvcc_out = subprocess.check_output(["nvcc", "--version"], text=True)
        nvcc_line = [ln for ln in nvcc_out.splitlines() if "release" in ln][-1]
        info["cuda_toolkit"] = nvcc_line.strip()
    except Exception as exc:  # pragma: no cover
        info["cuda_toolkit"] = f"Unknown ({exc})"

    try:
        tileiras_out = subprocess.check_output(["tileiras", "--version"], text=True)
        info["tileiras"] = tileiras_out.splitlines()[0].strip()
    except Exception:
        info["tileiras"] = "Unavailable in PATH"

    import cuda.tile as ct
    import cupy

    info["cutile"] = ct.__version__
    info["torch"] = torch.__version__
    info["cupy"] = cupy.__version__

    try:
        import flash_attn

        info["flashattention"] = getattr(flash_attn, "__version__", "installed")
    except Exception:
        info["flashattention"] = "Not installed (baseline used: torch SDPA)"

    info["timing"] = "CUDA events, median + p95 after warmup"
    return info


def write_table3(path: Path, info: dict[str, str]) -> None:
    lines = [
        "# Table 3 - Reproducibility Checklist",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| System | {info['system']} |",
        f"| GPU driver | {info['driver']} |",
        f"| CUDA toolkit | {info['cuda_toolkit']} |",
        f"| tileiras | {info['tileiras']} |",
        f"| cuTile | {info['cutile']} |",
        f"| PyTorch | {info['torch']} |",
        f"| CuPy | {info['cupy']} |",
        f"| FlashAttention version | {info['flashattention']} |",
        f"| Timing method | {info['timing']} |",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_benchmark(
    *,
    b: int,
    h: int,
    s_values: list[int],
    d_values: list[int],
    dtypes: list[torch.dtype],
    causal_flags: list[bool],
    warmup: int,
    iters: int,
    enable_flashattention: bool,
) -> list[dict[str, str | int | float | bool]]:
    records: list[dict[str, str | int | float | bool]] = []

    baseline_name = "torch_sdpa"
    flash_attn_func = maybe_flash_baseline() if enable_flashattention else None

    for causal in causal_flags:
        for dtype in dtypes:
            bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
            dtype_name = str(dtype).replace("torch.", "")

            for s in s_values:
                for d in d_values:
                    torch.manual_seed(1234)
                    q = torch.randn((b, h, s, d), device="cuda", dtype=dtype)
                    k = torch.randn((b, h, s, d), device="cuda", dtype=dtype)
                    v = torch.randn((b, h, s, d), device="cuda", dtype=dtype)
                    flops = attention_forward_flops(b=b, h=h, s=s, d=d, causal=causal)

                    causal_mask = None
                    if causal:
                        causal_mask = torch.ones((1, 1, s, s), device="cuda", dtype=torch.bool).triu(1)

                    method_fns = [
                        ("tiledattention", lambda q=q, k=k, v=v, causal=causal: sdpa(q, k, v, causal=causal)),
                        (
                            baseline_name,
                            lambda q=q, k=k, v=v, causal=causal: torch_sdpa(q, k, v, causal=causal),
                        ),
                        (
                            "torch_sdpa_math",
                            lambda q=q, k=k, v=v, causal=causal: torch_sdpa_math(q, k, v, causal=causal),
                        ),
                        (
                            "standard_eager",
                            lambda q=q, k=k, v=v, causal=causal, causal_mask=causal_mask: standard_eager_attention(
                                q, k, v, causal=causal, causal_mask=causal_mask
                            ),
                        ),
                    ]
                    if flash_attn_func is not None:
                        method_fns.append(
                            (
                                "flashattention",
                                lambda q=q, k=k, v=v, causal=causal, flash_attn_func=flash_attn_func: flash_attention_forward(
                                    q,
                                    k,
                                    v,
                                    causal=causal,
                                    flash_attn_func=flash_attn_func,
                                ),
                            )
                        )

                    stats: dict[str, tuple[float, float]] = {}
                    for method, fn in method_fns:
                        stats[method] = time_cuda_callable_safe(
                            fn,
                            warmup=warmup,
                            iters=iters,
                            method=method,
                        )

                    for method, (median_ms, p95_ms) in stats.items():
                        tput = throughput_tokens_per_s(b=b, h=h, s=s, median_ms=median_ms)
                        tflops = tflops_per_s(flops=flops, median_ms=median_ms)
                        bw = approximate_bw_gbps(
                            b=b,
                            h=h,
                            s=s,
                            d=d,
                            bytes_per_elem=bytes_per_elem,
                            median_ms=median_ms,
                        )
                        records.append(
                            {
                                "method": method,
                                "baseline_name": baseline_name,
                                "dtype": dtype_name,
                                "causal": causal,
                                "B": b,
                                "H": h,
                                "S": s,
                                "D": d,
                                "flops_forward": flops,
                                "median_ms": median_ms,
                                "p95_ms": p95_ms,
                                "throughput_tokens_per_s": tput,
                                "tflops_per_s": tflops,
                                "approx_bw_gbps": bw,
                            }
                        )

                    msg = (
                        f"[bench] causal={causal} dtype={dtype_name} S={s} D={d} "
                        f"tiled={stats['tiledattention'][0]:.3f}ms "
                        f"fused={stats['torch_sdpa'][0]:.3f}ms "
                        f"math={stats['torch_sdpa_math'][0]:.3f}ms "
                        f"eager={stats['standard_eager'][0]:.3f}ms"
                    )
                    if "flashattention" in stats:
                        msg += f" flashattn={stats['flashattention'][0]:.3f}ms"
                    print(msg)

    return records


def write_csv(path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(
    rows: list[dict[str, str | int | float | bool]],
    *,
    method: str,
    dtype: str,
    causal: bool,
    d: int,
) -> dict[int, dict[str, str | int | float | bool]]:
    out: dict[int, dict[str, str | int | float | bool]] = {}
    for row in rows:
        if (
            row["method"] == method
            and row["dtype"] == dtype
            and row["causal"] == causal
            and row["D"] == d
        ):
            out[int(row["S"])] = row
    return out


def build_figures(
    *,
    rows: list[dict[str, str | int | float | bool]],
    fig_dir: Path,
) -> None:
    baseline_name = "torch_sdpa"
    s_values = [512, 1024, 2048, 4096, 8192]
    d_values = [64, 96, 128, 160]

    fig3_dtype = 128
    tiled_fp16 = filter_rows(rows, method="tiledattention", dtype="float16", causal=False, d=fig3_dtype)
    base_fp16 = filter_rows(rows, method=baseline_name, dtype="float16", causal=False, d=fig3_dtype)
    tiled_bf16 = filter_rows(rows, method="tiledattention", dtype="bfloat16", causal=False, d=fig3_dtype)
    base_bf16 = filter_rows(rows, method=baseline_name, dtype="bfloat16", causal=False, d=fig3_dtype)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        s_values,
        [float(tiled_fp16[s]["throughput_tokens_per_s"]) for s in s_values],
        marker="o",
        label="TiledAttention FP16",
    )
    ax.plot(
        s_values,
        [float(base_fp16[s]["throughput_tokens_per_s"]) for s in s_values],
        marker="o",
        label="Baseline FP16",
    )
    ax.plot(
        s_values,
        [float(tiled_bf16[s]["throughput_tokens_per_s"]) for s in s_values],
        marker="s",
        label="TiledAttention BF16",
    )
    ax.plot(
        s_values,
        [float(base_bf16[s]["throughput_tokens_per_s"]) for s in s_values],
        marker="s",
        label="Baseline BF16",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(s_values, [str(s) for s in s_values])
    ax.set_xlabel("Sequence length, S")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    _style_axis(ax)
    _save_figure(fig, fig_dir / "figure3_throughput_vs_s.png")
    plt.close(fig)

    heat = np.zeros((len(s_values), len(d_values)), dtype=np.float64)
    for i, s in enumerate(s_values):
        for j, d in enumerate(d_values):
            tiled_row = next(
                row
                for row in rows
                if row["method"] == "tiledattention"
                and row["dtype"] == "float16"
                and row["causal"] is False
                and row["S"] == s
                and row["D"] == d
            )
            base_row = next(
                row
                for row in rows
                if row["method"] == baseline_name
                and row["dtype"] == "float16"
                and row["causal"] is False
                and row["S"] == s
                and row["D"] == d
            )
            heat[i, j] = 100.0 * float(tiled_row["throughput_tokens_per_s"]) / float(
                base_row["throughput_tokens_per_s"]
            )

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    im = ax.imshow(heat, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("TiledAttention / Baseline (%)", fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xticks(range(len(d_values)), [str(d) for d in d_values])
    ax.set_yticks(range(len(s_values)), [str(s) for s in s_values])
    ax.set_xlabel("Head dimension, D")
    ax.set_ylabel("Sequence length, S")
    for i in range(len(s_values)):
        for j in range(len(d_values)):
            ax.text(
                j,
                i,
                f"{heat[i, j]:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=ANNOTATION_FONTSIZE,
            )
    _style_axis(ax)
    _save_figure(fig, fig_dir / "figure4_regime_map.png")
    plt.close(fig)

    tiled_bw = np.array([float(tiled_fp16[s]["approx_bw_gbps"]) for s in s_values], dtype=np.float64)
    base_bw = np.array([float(base_fp16[s]["approx_bw_gbps"]) for s in s_values], dtype=np.float64)
    norm = max(float(tiled_bw.max()), float(base_bw.max()))
    tiled_bw_n = tiled_bw / norm
    base_bw_n = base_bw / norm

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(s_values, tiled_bw_n, marker="o", label="TiledAttention")
    ax.plot(s_values, base_bw_n, marker="o", label="Baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks(s_values, [str(s) for s in s_values])
    ax.set_xlabel("Sequence length, S")
    ax.set_ylabel("Normalized bandwidth proxy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    _style_axis(ax)
    _save_figure(fig, fig_dir / "figure5_bw_proxy.png")
    plt.close(fig)


def build_flashattention_style_figure(
    *,
    rows: list[dict[str, str | int | float | bool]],
    fig_dir: Path,
) -> None:
    baseline_name = "torch_sdpa"
    s_values = [512, 1024, 2048, 4096, 8192]
    d_values = [64, 96, 128, 160]
    causal_values = sorted({bool(row["causal"]) for row in rows if row["dtype"] == "float16"})
    if not causal_values:
        return

    fig, axes = plt.subplots(
        len(d_values),
        len(causal_values),
        figsize=(7 * len(causal_values), 4 * len(d_values)),
        sharex=True,
        squeeze=False,
    )

    for i, d in enumerate(d_values):
        for j, causal in enumerate(causal_values):
            ax = axes[i, j]
            x = np.arange(len(s_values), dtype=np.float64)
            width = 0.35

            tiled_vals: list[float] = []
            base_vals: list[float] = []
            for s in s_values:
                tiled_row = next(
                    row
                    for row in rows
                    if row["method"] == "tiledattention"
                    and row["dtype"] == "float16"
                    and row["causal"] is causal
                    and row["S"] == s
                    and row["D"] == d
                )
                base_row = next(
                    row
                    for row in rows
                    if row["method"] == baseline_name
                    and row["dtype"] == "float16"
                    and row["causal"] is causal
                    and row["S"] == s
                    and row["D"] == d
                )
                tiled_vals.append(float(tiled_row["tflops_per_s"]))
                base_vals.append(float(base_row["tflops_per_s"]))

            ax.bar(x - width / 2.0, tiled_vals, width=width, label="TiledAttention", color="#1f77b4")
            ax.bar(x + width / 2.0, base_vals, width=width, label="Baseline (torch_sdpa)", color="#ff7f0e")
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in s_values], rotation=0)
            ax.grid(axis="y", alpha=0.25)
            if j == 0:
                ax.set_ylabel("TFLOPs/s")
            if i == len(d_values) - 1:
                mode = "causal" if causal else "non-causal"
                ax.set_xlabel(f"Sequence length, S ({mode})")
            if i == 0 and j == 0:
                ax.legend(fontsize=LEGEND_FONTSIZE)
            _style_axis(ax)

    _save_figure(fig, fig_dir / "figure_fa_style_tflops_fp16.png")
    plt.close(fig)


def build_explicit_baseline_figure(
    *,
    rows: list[dict[str, str | int | float | bool]],
    fig_dir: Path,
) -> None:
    s_values = [512, 1024, 2048, 4096, 8192]
    d = 128
    dtype = "float16"
    methods = [
        ("standard_eager", "Standard attention", "#4e79a7"),
        ("torch_sdpa_math", "PyTorch SDPA (math)", "#59a14f"),
        ("torch_sdpa", "PyTorch SDPA (fused)", "#f28e2b"),
        ("tiledattention", "TiledAttention", "#e15759"),
    ]
    if any(row["method"] == "flashattention" for row in rows):
        methods.insert(3, ("flashattention", "FlashAttention", "#b07aa1"))
    causal_values = sorted({bool(row["causal"]) for row in rows if row["dtype"] == dtype})
    if not causal_values:
        return

    fig, axes = plt.subplots(
        1,
        len(causal_values),
        figsize=(8 * len(causal_values), 4.8),
        sharey=True,
        squeeze=False,
    )
    x = np.arange(len(s_values), dtype=np.float64)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))

    for j, causal in enumerate(causal_values):
        ax = axes[0, j]
        for offset, (method, label, color) in zip(offsets, methods, strict=False):
            vals: list[float] = []
            for s in s_values:
                row = next(
                    (
                        row
                        for row in rows
                        if row["method"] == method
                        and row["dtype"] == dtype
                        and row["causal"] is causal
                        and row["S"] == s
                        and row["D"] == d
                    ),
                    None,
                )
                vals.append(float(row["tflops_per_s"]) if row is not None else math.nan)
            ax.bar(x + offset, vals, width=width, label=label, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in s_values])
        mode = "causal" if causal else "non-causal"
        ax.set_xlabel(f"Sequence length, S ({mode})")
        ax.grid(axis="y", alpha=0.25)
        if j == 0:
            ax.set_ylabel("TFLOPs/s")
            ax.legend(fontsize=LEGEND_FONTSIZE)
        _style_axis(ax)

    _save_figure(fig, fig_dir / "figure6_explicit_baselines_tflops_fp16.png")
    plt.close(fig)


def run_tune(
    *,
    b: int,
    h: int,
    warmup: int,
    iters: int,
) -> list[dict[str, str | int | float]]:
    ct = get_cutile_module()
    cupy = get_cupy_module()
    torch_mod = get_torch_module()

    candidates = [(32, 64), (64, 64), (64, 128), (64, 256), (128, 64), (128, 128), (128, 256)]
    regimes = [("Short S", 1024), ("Mid S", 4096), ("Long S", 8192)]
    d = 128
    dtype = torch.float16
    causal = False
    accum_mode = "fp32"

    kernel_cache: dict[tuple[int, int, int, str, bool, str], object] = {}

    def run_custom_tile(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tm: int, tn: int) -> None:
        batch, heads, seq_len, head_dim = map(int, q.shape)
        bh = batch * heads
        q_bh = q.contiguous().reshape(bh, seq_len, head_dim)
        k_bh = k.contiguous().reshape(bh, seq_len, head_dim)
        v_bh = v.contiguous().reshape(bh, seq_len, head_dim)
        k_t = k_bh.transpose(1, 2).contiguous()
        out_bh = torch_mod.empty_like(q_bh)

        key = (tm, tn, head_dim, str(q.dtype), causal, accum_mode)
        kernel = kernel_cache.get(key)
        if kernel is None:
            out_ct_dtype = _ct_dtype_for_torch_dtype(ct, q.dtype)
            kernel = make_flashattn_fwd_kernel(
                tm,
                tn,
                head_dim,
                dtype=out_ct_dtype,
                causal=causal,
                accum_mode=accum_mode,
                opt_level=3,
                occupancy=None,
                num_ctas=None,
            )
            kernel_cache[key] = kernel

        scale = 1.0 / math.sqrt(float(head_dim))
        grid = (bh, (seq_len + tm - 1) // tm, 1)
        torch_mod.cuda.synchronize()
        stream = cupy.cuda.get_current_stream()
        ct.launch(stream, grid, kernel, (q_bh, k_t, v_bh, out_bh, float(scale)))
        stream.synchronize()

    records: list[dict[str, str | int | float]] = []

    for regime_name, s in regimes:
        torch.manual_seed(2026 + s)
        q = torch.randn((b, h, s, d), device="cuda", dtype=dtype)
        k = torch.randn((b, h, s, d), device="cuda", dtype=dtype)
        v = torch.randn((b, h, s, d), device="cuda", dtype=dtype)

        measured: list[tuple[float, float, int, int]] = []
        for tm, tn in candidates:
            median_ms, p95_ms = time_cuda_callable(
                lambda q=q, k=k, v=v, tm=tm, tn=tn: run_custom_tile(q, k, v, tm, tn),
                warmup=warmup,
                iters=iters,
            )
            measured.append((median_ms, p95_ms, tm, tn))
            print(f"[tune] {regime_name} S={s} TM={tm} TN={tn} median={median_ms:.3f}ms")

        measured.sort(key=lambda x: x[0])
        best = measured[0]
        runner = measured[1]
        sensitivity_drop = 100.0 * (runner[0] - best[0]) / best[0]

        records.append(
            {
                "regime": regime_name,
                "S": s,
                "D": d,
                "best_TM": best[2],
                "best_TN": best[3],
                "best_median_ms": best[0],
                "runner_TM": runner[2],
                "runner_TN": runner[3],
                "runner_median_ms": runner[0],
                "sensitivity_drop_percent": sensitivity_drop,
            }
        )

    return records


def write_table4(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    lines = [
        "# Table 4 - Best Tile Settings by Regime",
        "",
        "| Regime | Shapes | Best (TM, TN) | Runner-up | Sensitivity drop (%) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {regime} | S={S}, D={D} | ({best_TM}, {best_TN}) | ({runner_TM}, {runner_TN}) | "
            "{sensitivity_drop_percent:.2f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def write_summary(path: Path, bench_csv: Path, tune_csv: Path, baseline_name: str) -> None:
    has_flashattention = False
    try:
        with bench_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            has_flashattention = any(row.get("method") == "flashattention" for row in reader)
    except Exception:
        has_flashattention = False

    lines = [
        "# Study Summary",
        "",
        "This folder contains the benchmark/tuning study outputs to populate the paper tables and figures.",
        "",
        "## Baseline",
        "",
        f"- Baseline used: `{baseline_name}`",
        (
            "- FlashAttention package availability: installed and benchmarked."
            if has_flashattention
            else "- FlashAttention package availability: not installed or unsupported in this environment."
        ),
        "",
        "## Outputs",
        "",
        f"- Benchmark CSV: `{bench_csv}`",
        "- Benchmark CSV includes latency, tokens/s, TFLOPs/s, and bandwidth proxy columns.",
        f"- Tuning CSV: `{tune_csv}`",
        "- Table 3 markdown: `benchmark-gb10/results/table3_reproducibility.md`",
        "- Table 4 markdown: `benchmark-gb10/results/table4_tiling_sensitivity.md`",
        "- Figure 3: `benchmark-gb10/figures/figure3_throughput_vs_s.png`",
        "- Figure 4: `benchmark-gb10/figures/figure4_regime_map.png`",
        "- Figure 5: `benchmark-gb10/figures/figure5_bw_proxy.png`",
        "- FlashAttention-style TFLOPs figure: `benchmark-gb10/figures/figure_fa_style_tflops_fp16.png`",
        "- Explicit baselines TFLOPs figure: `benchmark-gb10/figures/figure6_explicit_baselines_tflops_fp16.png`",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TiledAttention paper study.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--include-causal", action="store_true", default=True)
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--disable-flashattention", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Run this study on the host GPU.", file=sys.stderr)
        return 2

    out_root = Path("benchmark-gb10")
    results_dir = out_root / "results"
    figs_dir = out_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    causal_flags = [False, True]
    if args.no_causal:
        causal_flags = [False]
    elif not args.include_causal:
        causal_flags = [False]

    s_values = [512, 1024, 2048, 4096, 8192]
    d_values = [64, 96, 128, 160]
    dtypes = [torch.float16, torch.bfloat16]

    print(
        f"[study] warmup={args.warmup} iters={args.iters} B={args.batch} H={args.heads} "
        f"causal_flags={causal_flags}"
    )

    bench_rows = run_benchmark(
        b=args.batch,
        h=args.heads,
        s_values=s_values,
        d_values=d_values,
        dtypes=dtypes,
        causal_flags=causal_flags,
        warmup=args.warmup,
        iters=args.iters,
        enable_flashattention=not args.disable_flashattention,
    )

    bench_csv = results_dir / "benchmark_results.csv"
    write_csv(bench_csv, bench_rows)
    build_figures(rows=bench_rows, fig_dir=figs_dir)
    build_flashattention_style_figure(rows=bench_rows, fig_dir=figs_dir)
    build_explicit_baseline_figure(rows=bench_rows, fig_dir=figs_dir)

    tune_rows = run_tune(b=args.batch, h=args.heads, warmup=args.warmup, iters=args.iters)
    tune_csv = results_dir / "tuning_results.csv"
    write_csv(tune_csv, tune_rows)

    repro_info = collect_repro_info()
    write_table3(results_dir / "table3_reproducibility.md", repro_info)
    write_table4(results_dir / "table4_tiling_sensitivity.md", tune_rows)
    write_summary(
        results_dir / "study_summary.md",
        bench_csv=bench_csv,
        tune_csv=tune_csv,
        baseline_name="torch_sdpa",
    )

    print("[study] completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
