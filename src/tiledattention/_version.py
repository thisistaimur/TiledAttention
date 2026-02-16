"""Version helpers for tiledattention."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
    try:
        return version("tiledattention")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()
