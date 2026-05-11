from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


METRIC_COLUMNS: dict[str, tuple[str, ...]] = {
    # For NCU raw CSV, these are percentages, not raw instruction counts.
    "tensor_core_insts": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__inst_executed_pipe_tensor_subpipe_hmma.avg.pct_of_peak_sustained_elapsed",
        "pmsampling:sm__inst_executed_pipe_tensor_subpipe_hmma_realtime.avg.pct_of_peak_sustained_elapsed",
    ),
    "fp32_fma_insts": (
        "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
        "sm__inst_executed_pipe_fma.sum.pct_of_peak_sustained_elapsed",
    ),
    "sm_throughput_pct_peak": (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "TriageCompute.sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "dram_throughput_pct_peak": (
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "l2_throughput_pct_peak": (
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "TriageCompute.lts__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "achieved_warps_active_pct_peak": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_active.avg.pct_of_peak_sustained_active",
    ),
    "stall_long_scoreboard": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    ),
    "stall_math_pipe_throttle": (
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    ),
    "stall_barrier": (
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    ),
}


@dataclass(frozen=True)
class ProfileResult:
    method: str
    rep_path: Path
    raw_csv_path: Path
    primary_kernel_time_ms: float | None
    primary_kernel_name: str
    metrics: dict[str, float | None]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    This helper is part of the benchmark and profiling pipeline.

    Returns:
        argparse.Namespace: Function result value.
    """
    parser = argparse.ArgumentParser(
        description="Profile TiledAttention vs torch SDPA with Nsight Compute in one pass."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--accum-mode", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sync-mode", type=str, default="async", choices=["strict", "post", "async"])
    parser.add_argument("--tile-m", type=int, default=None)
    parser.add_argument("--tile-n", type=int, default=None)
    parser.add_argument("--opt-level", type=int, default=None)
    parser.add_argument("--occupancy", type=int, default=None)
    parser.add_argument("--num-ctas", type=int, default=None)
    parser.add_argument("--disable-forced-flash", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-gb10/results"))
    parser.add_argument("--ncu-path", type=Path, default=None)
    return parser.parse_args()


def _parse_float(value: str | None) -> float | None:
    """
    Parse float.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        value: Scalar value to parse or format.

    Returns:
        float | None: Function result value.
    """
    if value is None:
        return None
    cleaned = value.strip().replace(",", "")
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_ncu_csv_rows(text: str) -> list[dict[str, str]]:
    """
    Parse ncu csv rows.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        text: Raw text content to parse.

    Returns:
        list[dict[str, str]]: Function result value.
    """
    lines = [
        line
        for line in text.splitlines()
        if line and not line.startswith("==PROF==") and not line.startswith("==ERROR==")
    ]
    header_idx = None
    for idx, line in enumerate(lines):
        if "Kernel Name" in line and '"ID"' in line:
            header_idx = idx
            break
    if header_idx is None:
        return []

    reader = csv.DictReader(lines[header_idx:])
    return [row for row in reader if row]


def _kernel_duration(row: dict[str, str]) -> float:
    """
    Internal helper for kernel duration.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        row: Single parsed row record.

    Returns:
        float: Function result value.
    """
    return _parse_float(row.get("gpu__time_duration.sum")) or _parse_float(
        row.get("gpu__time_duration.avg")
    ) or 0.0


def _time_unit_scale_to_ms(unit_text: str | None) -> float:
    """
    Measure unit scale to ms.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        unit_text: Function argument.

    Returns:
        float: Function result value.
    """
    unit = (unit_text or "").strip().lower()
    if unit == "ms":
        return 1.0
    if unit == "us":
        return 1.0 / 1000.0
    if unit == "ns":
        return 1.0 / 1_000_000.0
    if unit == "s":
        return 1000.0
    return 1.0


def _extract_time_scale_to_ms(rows: list[dict[str, str]]) -> float:
    """
    Internal helper for extract time scale to ms.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        rows: Parsed row records from CSV or benchmark output.

    Returns:
        float: Function result value.
    """
    if not rows:
        return 1.0
    header_row = rows[0]
    if (
        (header_row.get("Kernel Name", "").strip() == "")
        and (header_row.get("ID", "").strip() == "")
        and header_row.get("gpu__time_duration.sum", "").strip() != ""
    ):
        return _time_unit_scale_to_ms(header_row.get("gpu__time_duration.sum"))
    return 1.0


def _select_primary_kernel_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """
    Internal helper for select primary kernel row.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        rows: Parsed row records from CSV or benchmark output.

    Returns:
        dict[str, str] | None: Function result value.
    """
    kernel_rows = [row for row in rows if row.get("Kernel Name", "").strip() != ""]
    if not kernel_rows:
        return None
    flash_rows = [row for row in kernel_rows if "flash_fwd_kernel" in row.get("Kernel Name", "")]
    candidates = flash_rows if flash_rows else kernel_rows
    return max(candidates, key=_kernel_duration)


def _extract_metric_summary(row: dict[str, str] | None) -> dict[str, float | None]:
    """
    Internal helper for extract metric summary.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        row: Single parsed row record.

    Returns:
        dict[str, float | None]: Function result value.
    """
    summary: dict[str, float | None] = {key: None for key in METRIC_COLUMNS}
    if row is None:
        return summary

    for key, columns in METRIC_COLUMNS.items():
        for column in columns:
            value = _parse_float(row.get(column))
            if value is not None:
                summary[key] = value
                break
    return summary


def _run_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Run subprocess.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        cmd: Subprocess command to execute.
        env: Environment variables for subprocess execution.
        capture_output: Function argument.

    Returns:
        subprocess.CompletedProcess[str]: Function result value.
    """
    return subprocess.run(
        cmd,
        check=True,
        env=env,
        text=True,
        capture_output=capture_output,
    )


def _profile_method(
    *,
    method: str,
    args: argparse.Namespace,
    workload_script: Path,
    ncu_path: str,
    output_dir: Path,
) -> ProfileResult:
    """
    Internal helper for profile method.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        method: Method name to execute or profile.
        args: Parsed command-line arguments namespace.
        workload_script: Workload script path executed under profiler.
        ncu_path: Path to the Nsight Compute CLI executable.
        output_dir: Directory for generated profiling artifacts.

    Returns:
        ProfileResult: Function result value.
    """
    tag = (
        f"ncu_{method}_b{args.batch}_h{args.heads}_s{args.seq_len}_d{args.head_dim}_"
        f"{args.dtype}_acc{args.accum_mode}_{'causal' if args.causal else 'noncausal'}"
    )
    rep_base = output_dir / tag
    raw_csv_path = output_dir / f"{tag}_raw.csv"

    cmd = [
        ncu_path,
        "-f",
        "--target-processes",
        "all",
        "--set",
        "full",
        "--section",
        "SpeedOfLight",
        "--section",
        "Occupancy",
        "--section",
        "LaunchStats",
        "--section",
        "SchedulerStats",
        "--section",
        "MemoryWorkloadAnalysis",
        "--section",
        "ComputeWorkloadAnalysis",
        "-o",
        str(rep_base),
        sys.executable,
        str(workload_script),
        "--method",
        method,
        "--batch",
        str(args.batch),
        "--heads",
        str(args.heads),
        "--seq-len",
        str(args.seq_len),
        "--head-dim",
        str(args.head_dim),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]
    if args.causal:
        cmd.append("--causal")

    env = os.environ.copy()
    if method == "tiledattention":
        env["TILEDATTN_SYNC_MODE"] = args.sync_mode
        env["TILEDATTN_ACCUM_MODE"] = args.accum_mode
        if args.tile_m is not None:
            env["TILEDATTN_TILE_M"] = str(args.tile_m)
        if args.tile_n is not None:
            env["TILEDATTN_TILE_N"] = str(args.tile_n)
        if args.opt_level is not None:
            env["TILEDATTN_KERNEL_OPT_LEVEL"] = str(args.opt_level)
        if args.occupancy is not None:
            env["TILEDATTN_KERNEL_OCCUPANCY"] = str(args.occupancy)
        if args.num_ctas is not None:
            env["TILEDATTN_KERNEL_NUM_CTAS"] = str(args.num_ctas)

    print(f"[ncu] profiling method={method}")
    _run_subprocess(cmd, env=env, capture_output=False)

    rep_path = rep_base.with_suffix(".ncu-rep")
    if not rep_path.exists():
        raise FileNotFoundError(f"Nsight report was not created: {rep_path}")

    import_cmd = [ncu_path, "--import", str(rep_path), "--csv", "--page", "raw"]
    imported = _run_subprocess(import_cmd, capture_output=True)
    raw_csv_path.write_text(imported.stdout)

    rows = _parse_ncu_csv_rows(imported.stdout)
    time_scale_to_ms = _extract_time_scale_to_ms(rows)
    primary_row = _select_primary_kernel_row(rows)
    primary_kernel_name = (
        primary_row.get("Kernel Name", "n/a").strip() if primary_row is not None else "n/a"
    )
    primary_kernel_time = _parse_float(primary_row.get("gpu__time_duration.sum")) if primary_row is not None else None
    primary_kernel_time_ms = (
        primary_kernel_time * time_scale_to_ms if primary_kernel_time is not None else None
    )
    metrics = _extract_metric_summary(primary_row)
    return ProfileResult(
        method=method,
        rep_path=rep_path,
        raw_csv_path=raw_csv_path,
        primary_kernel_time_ms=primary_kernel_time_ms,
        primary_kernel_name=primary_kernel_name,
        metrics=metrics,
    )


def _fmt_metric(value: float | None) -> str:
    """
    Format metric.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        value: Scalar value to parse or format.

    Returns:
        str: Function result value.
    """
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def _short_kernel_name(name: str, *, max_len: int = 64) -> str:
    """
    Internal helper for short kernel name.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        name: Identifier or metric name.
        max_len: Function argument.

    Returns:
        str: Function result value.
    """
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def _diagnose(result: ProfileResult) -> list[str]:
    """
    Internal helper for diagnose.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        result: Single profiling result record.

    Returns:
        list[str]: Function result value.
    """
    m = result.metrics
    lines: list[str] = []
    tensor = m["tensor_core_insts"]
    occ = m["achieved_warps_active_pct_peak"]
    sm = m["sm_throughput_pct_peak"]
    dram = m["dram_throughput_pct_peak"]
    l2 = m["l2_throughput_pct_peak"]
    long_sb = m["stall_long_scoreboard"]
    math_sb = m["stall_math_pipe_throttle"]

    if tensor is not None and tensor < 1.0:
        lines.append("Near-zero tensor-core activity on primary kernel.")
    if occ is not None and occ < 30.0:
        lines.append("Low achieved occupancy (<30% of peak active warps).")
    if sm is not None and sm < 20.0 and (l2 is None or l2 < 25.0):
        lines.append("Neither compute nor DRAM are saturated; likely scheduling/pipeline overhead.")
    if long_sb is not None and long_sb > 2.0:
        lines.append("Long-scoreboard stalls are high; likely memory-latency / dependency bottleneck.")
    if math_sb is not None and math_sb > 1.0:
        lines.append("Math-pipe throttle is high; check instruction mix and issue balance.")
    if dram is not None and dram > 70.0 and long_sb is not None and long_sb > 2.0:
        lines.append("High DRAM pressure with long scoreboard stalls; likely memory-latency bound.")
    if not lines:
        lines.append("No single dominant bottleneck from coarse metrics; inspect kernel-level details in Nsight UI.")
    return lines


def _write_summary(path: Path, results: list[ProfileResult], args: argparse.Namespace) -> None:
    """
    Write summary.
    This helper is part of the benchmark and profiling pipeline.

    Args:
        path: Output path for generated artifact.
        results: Collected profiling results.
        args: Parsed command-line arguments namespace.
    """
    lines = [
        "# Nsight Compute One-Pass Summary",
        "",
        "## Run Configuration",
        "",
        f"- Shape: B={args.batch}, H={args.heads}, S={args.seq_len}, D={args.head_dim}",
        f"- dtype: {args.dtype}",
        f"- tiledattention accum_mode: {args.accum_mode}",
        f"- causal: {args.causal}",
        f"- warmup: {args.warmup}, repeats: {args.repeats}",
        "",
        "## Key Metrics (primary kernel row)",
        "",
        "| Method | Primary Kernel | Kernel Time (gpu__time_duration.sum) | Tensor Pipe %peak | FP32 FMA Pipe %peak | Achieved Warps Active % | SM Throughput %peak | DRAM/Memory Throughput %peak | L2 Throughput %peak | Long Scoreboard Stall | Math Pipe Stall | Barrier Stall |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"| {r.method} | `{_short_kernel_name(r.primary_kernel_name)}` | {_fmt_metric(r.primary_kernel_time_ms)} | "
            f"{_fmt_metric(m['tensor_core_insts'])} | {_fmt_metric(m['fp32_fma_insts'])} | "
            f"{_fmt_metric(m['achieved_warps_active_pct_peak'])} | {_fmt_metric(m['sm_throughput_pct_peak'])} | "
            f"{_fmt_metric(m['dram_throughput_pct_peak'])} | {_fmt_metric(m['l2_throughput_pct_peak'])} | "
            f"{_fmt_metric(m['stall_long_scoreboard'])} | {_fmt_metric(m['stall_math_pipe_throttle'])} | "
            f"{_fmt_metric(m['stall_barrier'])} |"
        )

    lines.append("")
    lines.append("## Quick Diagnosis")
    lines.append("")
    for r in results:
        lines.append(f"### {r.method}")
        lines.append("")
        lines.append(f"- Primary kernel: `{r.primary_kernel_name}`")
        lines.append(f"- Kernel time (gpu__time_duration.sum, normalized to ms): {_fmt_metric(r.primary_kernel_time_ms)}")
        for bullet in _diagnose(r):
            lines.append(f"- {bullet}")
        lines.append(f"- Nsight report: `{r.rep_path}`")
        lines.append(f"- Raw CSV: `{r.raw_csv_path}`")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    """
    Run the script entrypoint.
    This helper is part of the benchmark and profiling pipeline.

    Returns:
        int: Function result value.
    """
    args = parse_args()
    if args.batch <= 0 or args.heads <= 0 or args.seq_len <= 0 or args.head_dim <= 0:
        raise ValueError("batch/heads/seq-len/head-dim must be positive.")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be > 0.")

    ncu_path: str | None = None
    if args.ncu_path is not None:
        candidate = args.ncu_path.expanduser()
        if candidate.exists():
            ncu_path = str(candidate)
        else:
            raise RuntimeError(f"Provided --ncu-path does not exist: {candidate}")
    else:
        ncu_path = shutil.which("ncu")
        if ncu_path is None:
            for candidate in (
                Path("/usr/local/cuda/bin/ncu"),
                Path("/usr/local/cuda-13.1/bin/ncu"),
            ):
                if candidate.exists():
                    ncu_path = str(candidate)
                    break
    if ncu_path is None:
        raise RuntimeError(
            "Nsight Compute CLI not found. Pass --ncu-path /path/to/ncu "
            "or add ncu to PATH."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    workload_script = Path(__file__).with_name("profile_attention_workload.py")
    if not workload_script.exists():
        raise FileNotFoundError(f"Missing workload script: {workload_script}")

    methods = ["tiledattention", "torch_sdpa"]
    if not args.disable_forced_flash:
        methods.append("torch_sdpa_flash_forced")

    results: list[ProfileResult] = []
    for method in methods:
        try:
            results.append(
                _profile_method(
                    method=method,
                    args=args,
                    workload_script=workload_script,
                    ncu_path=ncu_path,
                    output_dir=output_dir,
                )
            )
        except subprocess.CalledProcessError as exc:
            if method == "torch_sdpa_flash_forced":
                print(
                    "[ncu] warning: forced FLASH_ATTENTION profiling failed; "
                    f"continuing without it (exit={exc.returncode})."
                )
                continue
            raise

    summary_path = output_dir / (
        f"ncu_profile_summary_b{args.batch}_h{args.heads}_s{args.seq_len}_d{args.head_dim}_"
        f"{args.dtype}_acc{args.accum_mode}_{'causal' if args.causal else 'noncausal'}.md"
    )
    _write_summary(summary_path, results, args)
    print(f"[ncu] summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
