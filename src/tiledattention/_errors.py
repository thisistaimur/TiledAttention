"""Custom exception types for tiledattention."""

from __future__ import annotations


class TiledAttentionError(Exception):
    """Base exception for tiledattention."""


class UnsupportedPlatformError(TiledAttentionError):
    """Raised when runtime platform requirements are not satisfied."""


class InvalidShapeError(TiledAttentionError):
    """Raised when q/k/v tensors do not match expected SDPA input contracts."""


class DTypeNotSupportedError(TiledAttentionError):
    """Raised when input tensor dtypes are unsupported."""


class DependencyError(TiledAttentionError):
    """Raised when a required dependency is missing."""
