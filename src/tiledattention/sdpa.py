"""PyTorch-facing SDPA API."""

from __future__ import annotations

import math
from numbers import Real

from . import _runtime
from ._errors import InvalidShapeError
from .kernels import run_flash_fwd
from .utils.checks import validate_sdpa_inputs


def sdpa(
    q,
    k,
    v,
    *,
    causal: bool = False,
    scale: float | None = None,
):
    """Runs scaled dot-product attention."""

    if not isinstance(causal, bool):
        raise InvalidShapeError("causal must be a bool.")
    if scale is not None and (not isinstance(scale, Real) or float(scale) <= 0):
        raise InvalidShapeError("scale must be a positive number when provided.")

    _runtime.require_supported_runtime()
    torch_mod = _runtime.get_torch_module()
    validate_sdpa_inputs(torch_mod, q, k, v)

    head_dim = int(q.shape[-1])
    resolved_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    return run_flash_fwd(q, k, v, causal=causal, scale=resolved_scale)
