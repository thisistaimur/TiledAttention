"""Input checks shared by tiledattention APIs."""

from __future__ import annotations

from types import ModuleType

from .._errors import DTypeNotSupportedError, InvalidShapeError


def _ensure_tensor(torch_mod: ModuleType, name: str, value: object) -> None:
    if not isinstance(value, torch_mod.Tensor):
        raise InvalidShapeError(f"{name} must be a torch.Tensor.")


def validate_sdpa_inputs(torch_mod: ModuleType, q, k, v) -> None:
    _ensure_tensor(torch_mod, "q", q)
    _ensure_tensor(torch_mod, "k", k)
    _ensure_tensor(torch_mod, "v", v)

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.ndim != 4:
            raise InvalidShapeError(f"{name} must have shape [B, H, S, D].")
        if not bool(tensor.is_cuda):
            raise InvalidShapeError(f"{name} must be a CUDA tensor.")

    if q.shape != k.shape or q.shape != v.shape:
        raise InvalidShapeError("q, k, and v must have matching shape [B, H, S, D].")

    if q.device != k.device or q.device != v.device:
        raise InvalidShapeError("q, k, and v must be on the same CUDA device.")

    allowed_dtypes = {torch_mod.float16, torch_mod.bfloat16}
    if q.dtype not in allowed_dtypes:
        raise DTypeNotSupportedError("q dtype must be torch.float16 or torch.bfloat16.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise DTypeNotSupportedError("q, k, and v must have identical dtype.")
