"""In-memory kernel cache for cuTile kernel factories."""

from __future__ import annotations

from collections.abc import Callable

KernelCallable = Callable[..., object]
KernelKey = tuple[object, ...]

_KERNEL_CACHE: dict[KernelKey, KernelCallable] = {}


def get_kernel(key: KernelKey, factory: Callable[[], KernelCallable]) -> KernelCallable:
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = factory()
        _KERNEL_CACHE[key] = kernel
    return kernel


def reset_cache_for_tests() -> None:
    _KERNEL_CACHE.clear()
