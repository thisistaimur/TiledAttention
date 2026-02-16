from __future__ import annotations


def test_import_tiledattention() -> None:
    import tiledattention

    assert isinstance(tiledattention.__version__, str)
    assert tiledattention.__version__
    assert hasattr(tiledattention, "sdpa")
