"""Runtime checks and environment introspection for tiledattention."""

from __future__ import annotations

import importlib
import os
from threading import Lock
from types import ModuleType

from ._errors import DependencyError, UnsupportedPlatformError

_SUPPORTED_CC_MAJORS = {10, 12}
_runtime_ready = False
_runtime_lock = Lock()
_cached_torch: ModuleType | None = None
_cached_cupy: ModuleType | None = None
_cached_cutile: ModuleType | None = None


def _get_torch_module() -> ModuleType:
    global _cached_torch
    if _cached_torch is not None:
        return _cached_torch

    try:
        _cached_torch = importlib.import_module("torch")
        return _cached_torch
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "PyTorch is required for tiledattention. Install a CUDA-enabled torch build first."
        ) from exc


def get_torch_module() -> ModuleType:
    """Returns the imported torch module, raising DependencyError if unavailable."""
    return _get_torch_module()


def _get_cupy_module() -> ModuleType:
    global _cached_cupy
    if _cached_cupy is not None:
        return _cached_cupy

    try:
        _cached_cupy = importlib.import_module("cupy")
        return _cached_cupy
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "CuPy is required at runtime. Install a CUDA-compatible cupy build."
        ) from exc


def get_cupy_module() -> ModuleType:
    """Returns the imported cupy module, raising DependencyError if unavailable."""
    return _get_cupy_module()


def _cutile_candidates() -> tuple[str, ...]:
    raw = os.getenv("TILEDATTENTION_CUTILE_MODULE", "").strip()
    if raw:
        parsed = tuple(part.strip() for part in raw.split(",") if part.strip())
        if parsed:
            return parsed
    return ("cutile", "cuda.tile", "cuda_tile")


def _get_cutile_module() -> ModuleType:
    global _cached_cutile
    if _cached_cutile is not None:
        return _cached_cutile

    candidates = _cutile_candidates()
    for module_name in candidates:
        try:
            _cached_cutile = importlib.import_module(module_name)
            return _cached_cutile
        except ModuleNotFoundError:
            continue

    candidate_text = ", ".join(candidates)
    raise DependencyError(
        "cuTile Python module could not be imported. "
        f"Tried: {candidate_text}. "
        "If your install uses a different module path, set "
        "TILEDATTENTION_CUTILE_MODULE=<module_name>."
    )


def get_cutile_module() -> ModuleType:
    """Returns the imported cuTile module, raising DependencyError if unavailable."""
    return _get_cutile_module()


def _query_compute_capability(torch_mod: ModuleType) -> tuple[int, int]:
    try:
        device_index = torch_mod.cuda.current_device()
        major, minor = torch_mod.cuda.get_device_capability(device_index)
        return int(major), int(minor)
    except Exception as exc:  # pragma: no cover - defensive path
        raise UnsupportedPlatformError(
            "Unable to query GPU compute capability from torch.cuda."
        ) from exc


def require_supported_runtime() -> None:
    """Checks CUDA + Blackwell runtime support.

    This is intentionally lazy and should be called from operator entrypoints.
    """

    global _runtime_ready

    with _runtime_lock:
        if _runtime_ready:
            return

        torch_mod = _get_torch_module()

        if not bool(torch_mod.cuda.is_available()):
            raise UnsupportedPlatformError(
                "tiledattention requires CUDA + Blackwell GPU. "
                "torch.cuda.is_available() is False."
            )

        major, minor = _query_compute_capability(torch_mod)
        if major not in _SUPPORTED_CC_MAJORS:
            raise UnsupportedPlatformError(
                f"Unsupported GPU: compute capability {major}.{minor}; "
                "requires Blackwell-class GPU (10.x or 12.x)."
            )

        _get_cupy_module()
        _get_cutile_module()

        _runtime_ready = True


def _reset_runtime_cache_for_tests() -> None:
    """Testing hook to clear cached runtime validation state."""

    global _runtime_ready, _cached_torch, _cached_cupy, _cached_cutile
    with _runtime_lock:
        _runtime_ready = False
        _cached_torch = None
        _cached_cupy = None
        _cached_cutile = None
