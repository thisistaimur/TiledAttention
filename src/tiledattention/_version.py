"""Version helpers for tiledattention."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
    """
    Resolve version.
    It is used by the tiledattention runtime and tooling.

    Returns:
        str: Function result value.
    """
    try:
        return version("tiledattention")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()
