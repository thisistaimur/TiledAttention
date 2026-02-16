from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import pytest

from tiledattention._errors import (
    DTypeNotSupportedError,
    InvalidShapeError,
    UnsupportedPlatformError,
)
from tiledattention._runtime import _reset_runtime_cache_for_tests
from tiledattention.sdpa import sdpa


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    dtype: object
    device: str
    is_cuda: bool

    @property
    def ndim(self) -> int:
        return len(self.shape)


def _make_torch_module(*, cuda_available: bool, capability: tuple[int, int]) -> ModuleType:
    torch_mod = ModuleType("torch")
    torch_mod.Tensor = FakeTensor
    torch_mod.float16 = object()
    torch_mod.bfloat16 = object()

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


def _patch_runtime_dependency_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiledattention._runtime as runtime

    monkeypatch.setattr(runtime, "_get_cupy_module", lambda: ModuleType("cupy"))
    monkeypatch.setattr(runtime, "_get_cutile_module", lambda: ModuleType("cutile"))


def test_sdpa_prefers_runtime_error_on_cpu_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiledattention._runtime as runtime

    monkeypatch.setattr(
        runtime,
        "_get_torch_module",
        lambda: _make_torch_module(cuda_available=False, capability=(10, 0)),
    )

    _reset_runtime_cache_for_tests()
    with pytest.raises(UnsupportedPlatformError):
        sdpa(None, None, None)


def test_sdpa_validates_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiledattention._runtime as runtime

    torch_mod = _make_torch_module(cuda_available=True, capability=(10, 0))
    monkeypatch.setattr(runtime, "_get_torch_module", lambda: torch_mod)
    _patch_runtime_dependency_modules(monkeypatch)

    q = FakeTensor(shape=(1, 2, 64), dtype=torch_mod.float16, device="cuda:0", is_cuda=True)
    k = FakeTensor(shape=(1, 2, 64), dtype=torch_mod.float16, device="cuda:0", is_cuda=True)
    v = FakeTensor(shape=(1, 2, 64), dtype=torch_mod.float16, device="cuda:0", is_cuda=True)

    _reset_runtime_cache_for_tests()
    with pytest.raises(InvalidShapeError, match=r"\[B, H, S, D\]"):
        sdpa(q, k, v)


def test_sdpa_validates_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiledattention._runtime as runtime

    torch_mod = _make_torch_module(cuda_available=True, capability=(10, 0))
    monkeypatch.setattr(runtime, "_get_torch_module", lambda: torch_mod)
    _patch_runtime_dependency_modules(monkeypatch)

    q = FakeTensor(shape=(1, 2, 64, 64), dtype=object(), device="cuda:0", is_cuda=True)
    k = FakeTensor(shape=(1, 2, 64, 64), dtype=object(), device="cuda:0", is_cuda=True)
    v = FakeTensor(shape=(1, 2, 64, 64), dtype=object(), device="cuda:0", is_cuda=True)

    _reset_runtime_cache_for_tests()
    with pytest.raises(DTypeNotSupportedError, match="float16"):
        sdpa(q, k, v)


def test_sdpa_valid_input_reaches_kernel_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiledattention._runtime as runtime
    import tiledattention.sdpa as sdpa_mod

    torch_mod = _make_torch_module(cuda_available=True, capability=(10, 0))
    monkeypatch.setattr(runtime, "_get_torch_module", lambda: torch_mod)
    _patch_runtime_dependency_modules(monkeypatch)
    monkeypatch.setattr(sdpa_mod, "run_flash_fwd", lambda *_args, **_kwargs: "ok")

    q = FakeTensor(shape=(1, 2, 64, 64), dtype=torch_mod.float16, device="cuda:0", is_cuda=True)
    k = FakeTensor(shape=(1, 2, 64, 64), dtype=torch_mod.float16, device="cuda:0", is_cuda=True)
    v = FakeTensor(shape=(1, 2, 64, 64), dtype=torch_mod.float16, device="cuda:0", is_cuda=True)

    _reset_runtime_cache_for_tests()
    assert sdpa(q, k, v) == "ok"
