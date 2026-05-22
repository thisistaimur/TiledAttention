#!/usr/bin/env python3
"""Sync canonical package sources into torch-ext package for kernel-builder.

Source of truth: src/tiledattention
Mirror target:   torch-ext/tiledattention
"""

from __future__ import annotations

from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "tiledattention"
DST_ROOT = REPO_ROOT / "torch-ext" / "tiledattention"


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def sync() -> None:
    if not SRC_ROOT.exists():
        raise SystemExit(f"Missing source root: {SRC_ROOT}")
    if not DST_ROOT.exists():
        raise SystemExit(f"Missing destination root: {DST_ROOT}")

    src_files = _iter_python_files(SRC_ROOT)

    # Remove managed files that no longer exist in src.
    managed_dst_files = _iter_python_files(DST_ROOT)
    src_rel = {p.relative_to(SRC_ROOT) for p in src_files}
    for dst_file in managed_dst_files:
        rel = dst_file.relative_to(DST_ROOT)
        # Keep builder-generated helper modules (e.g., _ops.py).
        if rel.name.startswith("_ops"):
            continue
        if rel not in src_rel:
            dst_file.unlink()

    # Copy canonical files from src to torch-ext mirror.
    copied = 0
    for src_file in src_files:
        rel = src_file.relative_to(SRC_ROOT)
        dst_file = DST_ROOT / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1

    print(f"[sync] copied {copied} files from {SRC_ROOT} -> {DST_ROOT}")


if __name__ == "__main__":
    sync()
