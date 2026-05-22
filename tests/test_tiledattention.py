from __future__ import annotations


def test_tiledattention_exports_sdpa() -> None:
    import tiledattention

    assert hasattr(tiledattention, "sdpa")
    assert callable(tiledattention.sdpa)
