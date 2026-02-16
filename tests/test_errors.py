from __future__ import annotations

from types import ModuleType

import pytest

from tiledattention._errors import DependencyError, UnsupportedPlatformError
from tiledattention._runtime import _reset_runtime_cache_for_tests, require_supported_runtime


def _make_torch_module(*, cuda_available: bool, capability: tuple[int, int]) -> ModuleType:
    torch_mod = ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def get_device_capability(_device_index: int) -> tuple[int, int]:
            return capability

    torch_mod.cuda = FakeCuda()
    return torch_mod


def _raise_dependency_error(message: str) -> None:
    raise DependencyError(message)


def test_runtime_raises_on_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiledattention._runtime as runtime

    monkeypatch.setattr(
        runtime,
        "_get_torch_module",
        lambda: _make_torch_module(cuda_available=False, capability=(10, 0)),
    )

    _reset_runtime_cache_for_tests()
    with pytest.raises(UnsupportedPlatformError, match=r"requires CUDA \+ Blackwell GPU"):
        require_supported_runtime()


def test_runtime_raises_on_non_blackwell_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiledattention._runtime as runtime

    monkeypatch.setattr(
        runtime,
        "_get_torch_module",
        lambda: _make_torch_module(cuda_available=True, capability=(9, 0)),
    )

    _reset_runtime_cache_for_tests()
    with pytest.raises(UnsupportedPlatformError, match="compute capability 9.0"):
        require_supported_runtime()


def test_runtime_raises_on_missing_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiledattention._runtime as runtime

    monkeypatch.setattr(
        runtime,
        "_get_torch_module",
        lambda: _make_torch_module(cuda_available=True, capability=(10, 0)),
    )
    monkeypatch.setattr(
        runtime,
        "_get_cupy_module",
        lambda: _raise_dependency_error("CuPy is required at runtime."),
    )
    monkeypatch.setattr(runtime, "_get_cutile_module", lambda: ModuleType("cutile"))

    _reset_runtime_cache_for_tests()
    with pytest.raises(DependencyError, match="CuPy is required"):
        require_supported_runtime()


def test_runtime_raises_on_missing_cutile(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiledattention._runtime as runtime

    monkeypatch.setattr(
        runtime,
        "_get_torch_module",
        lambda: _make_torch_module(cuda_available=True, capability=(10, 0)),
    )
    monkeypatch.setattr(runtime, "_get_cupy_module", lambda: ModuleType("cupy"))
    monkeypatch.setattr(
        runtime,
        "_get_cutile_module",
        lambda: _raise_dependency_error("cuTile Python module could not be imported."),
    )

    _reset_runtime_cache_for_tests()
    with pytest.raises(DependencyError, match="cuTile Python module could not be imported"):
        require_supported_runtime()
