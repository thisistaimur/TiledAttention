"""Public package interface for tiledattention."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import (
    DependencyError,
    DTypeNotSupportedError,
    InvalidShapeError,
    TiledAttentionError,
    UnsupportedPlatformError,
)
from ._version import __version__

if TYPE_CHECKING:
    import torch


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    It is part of the public SDPA execution path.

    Args:
        q: Query tensor in attention layout.
        k: Key tensor in attention layout.
        v: Value tensor in attention layout.
        causal: Whether causal masking is enabled.
        scale: Attention scaling factor.

    Returns:
        torch.Tensor: Function result value.
    """
    from .sdpa import sdpa as _sdpa

    return _sdpa(q, k, v, causal=causal, scale=scale)


__all__ = [
    "__version__",
    "sdpa",
    "TiledAttentionError",
    "UnsupportedPlatformError",
    "InvalidShapeError",
    "DTypeNotSupportedError",
    "DependencyError",
]
