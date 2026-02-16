from __future__ import annotations

import pytest


def _require_gpu_runtime() -> None:
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available in this environment.")

    major, _minor = torch.cuda.get_device_capability()
    if major not in (10, 12):
        pytest.skip("This test targets Blackwell-class GPUs only.")

    pytest.importorskip("cupy")
    pytest.importorskip("cuda.tile")


@pytest.mark.gpu
def test_sdpa_matches_torch_reference_small_fp16_noncausal() -> None:
    torch = pytest.importorskip("torch")
    from tiledattention import sdpa

    _require_gpu_runtime()

    dtype = torch.float16
    device = torch.device("cuda")
    b, h, s, d = 1, 2, 64, 64

    torch.manual_seed(0)
    q = torch.randn((b, h, s, d), device=device, dtype=dtype)
    k = torch.randn((b, h, s, d), device=device, dtype=dtype)
    v = torch.randn((b, h, s, d), device=device, dtype=dtype)

    out = sdpa(q, k, v, causal=False)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )

    torch.cuda.synchronize()
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    assert torch.allclose(out, ref, atol=2.5e-2, rtol=2.5e-2)
