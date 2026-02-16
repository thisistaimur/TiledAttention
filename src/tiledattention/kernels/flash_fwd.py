"""FlashAttention-style forward kernel launcher."""

from __future__ import annotations

import os
from typing import Any

from .. import _runtime
from .._errors import DTypeNotSupportedError
from .compile_cache import get_kernel

_DEFAULT_TM = 64
_DEFAULT_TN = 64
_NEG_LARGE = -1.0e30


def _ct_dtype_for_torch_dtype(ct: Any, torch_dtype: Any) -> Any:
    torch_mod = _runtime.get_torch_module()
    if torch_dtype == torch_mod.float16:
        return ct.float16
    if torch_dtype == torch_mod.bfloat16:
        return ct.bfloat16
    raise DTypeNotSupportedError("q dtype must be torch.float16 or torch.bfloat16.")


def _get_env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _resolve_tile_config() -> tuple[int, int]:
    tile_m = _get_env_int("TILEDATTN_TILE_M", _DEFAULT_TM)
    tile_n = _get_env_int("TILEDATTN_TILE_N", _DEFAULT_TN)
    assert tile_m is not None and tile_n is not None
    if tile_m <= 0 or tile_n <= 0:
        raise ValueError("TILEDATTN_TILE_M and TILEDATTN_TILE_N must be positive.")
    return tile_m, tile_n


def _resolve_kernel_options() -> tuple[int, int | None, int | None]:
    opt_level = _get_env_int("TILEDATTN_KERNEL_OPT_LEVEL", 3)
    occupancy = _get_env_int("TILEDATTN_KERNEL_OCCUPANCY", None)
    num_ctas = _get_env_int("TILEDATTN_KERNEL_NUM_CTAS", None)
    assert opt_level is not None
    return opt_level, occupancy, num_ctas


def _resolve_accum_mode() -> str:
    mode = os.getenv("TILEDATTN_ACCUM_MODE", "fp32").strip().lower()
    if mode not in {"fp32", "fp16"}:
        raise ValueError(
            f"Unsupported TILEDATTN_ACCUM_MODE={mode!r}. Use one of: fp32, fp16."
        )
    return mode


def make_flashattn_fwd_kernel(
    tile_m: int,
    tile_n: int,
    head_dim: int,
    *,
    dtype: Any,
    causal: bool,
    accum_mode: str,
    opt_level: int,
    occupancy: int | None,
    num_ctas: int | None,
):
    ct = _runtime.get_cutile_module()
    kernel_kwargs: dict[str, Any] = {"opt_level": opt_level}
    if occupancy is not None:
        kernel_kwargs["occupancy"] = occupancy
    if num_ctas is not None:
        kernel_kwargs["num_ctas"] = num_ctas

    if accum_mode == "fp16":

        @ct.kernel(**kernel_kwargs)
        def flash_fwd_kernel_fp16acc(q, k_t, v, out, scale):
            bh_idx = ct.bid(0)
            q_tile_idx = ct.bid(1)

            # Keep GEMM inputs in low precision to allow tensor-core codegen.
            q_tile = ct.load(q, index=(bh_idx, q_tile_idx, 0), shape=(1, tile_m, head_dim))

            seq_len = q.shape[1]
            num_k_tiles = ct.cdiv(seq_len, tile_n)

            row = q_tile_idx * tile_m + ct.arange(tile_m, dtype=ct.int32)
            row = ct.expand_dims(row, 1)
            row = ct.expand_dims(row, 0)
            row_in_bounds = row < seq_len

            m_i = ct.full((1, tile_m, 1), _NEG_LARGE, ct.float32)
            l_i = ct.zeros((1, tile_m, 1), ct.float32)
            acc = ct.zeros((1, tile_m, head_dim), dtype)

            for k_tile_idx in range(num_k_tiles):
                k_tile_t = ct.load(k_t, index=(bh_idx, 0, k_tile_idx), shape=(1, head_dim, tile_n))

                score = ct.matmul(q_tile, k_tile_t)
                score = ct.astype(score, ct.float32) * scale

                col = k_tile_idx * tile_n + ct.arange(tile_n, dtype=ct.int32)
                col = ct.expand_dims(col, 0)
                col = ct.expand_dims(col, 0)
                key_in_bounds = col < seq_len
                valid = row_in_bounds & key_in_bounds
                if causal:
                    valid = valid & (row >= col)

                score = ct.where(valid, score, _NEG_LARGE)

                tile_max = ct.max(score, axis=2, keepdims=True)
                m_next = ct.maximum(m_i, tile_max)
                alpha = ct.exp(m_i - m_next)

                p = ct.exp(score - m_next)
                l_i = l_i * alpha + ct.sum(p, axis=2, keepdims=True)

                v_tile = ct.load(v, index=(bh_idx, k_tile_idx, 0), shape=(1, tile_n, head_dim))
                p_lowp = ct.astype(p, dtype)
                alpha_lowp = ct.astype(alpha, dtype)
                acc = acc * alpha_lowp + ct.matmul(p_lowp, v_tile)
                m_i = m_next

            safe_l = ct.where(row_in_bounds, l_i, 1.0)
            # Row-wise reciprocal avoids a full per-element divide over head_dim.
            inv_l = 1.0 / safe_l
            out_tile = acc * ct.astype(inv_l, dtype)
            out_tile = ct.where(row_in_bounds, out_tile, 0.0)
            ct.store(out, index=(bh_idx, q_tile_idx, 0), tile=out_tile)

        return flash_fwd_kernel_fp16acc

    @ct.kernel(**kernel_kwargs)
    def flash_fwd_kernel_fp32acc(q, k_t, v, out, scale):
        bh_idx = ct.bid(0)
        q_tile_idx = ct.bid(1)

        # Keep GEMM inputs in low precision to allow tensor-core codegen.
        q_tile = ct.load(q, index=(bh_idx, q_tile_idx, 0), shape=(1, tile_m, head_dim))

        seq_len = q.shape[1]
        num_k_tiles = ct.cdiv(seq_len, tile_n)

        row = q_tile_idx * tile_m + ct.arange(tile_m, dtype=ct.int32)
        row = ct.expand_dims(row, 1)
        row = ct.expand_dims(row, 0)
        row_in_bounds = row < seq_len

        m_i = ct.full((1, tile_m, 1), _NEG_LARGE, ct.float32)
        l_i = ct.zeros((1, tile_m, 1), ct.float32)
        acc = ct.zeros((1, tile_m, head_dim), ct.float32)

        for k_tile_idx in range(num_k_tiles):
            k_tile_t = ct.load(k_t, index=(bh_idx, 0, k_tile_idx), shape=(1, head_dim, tile_n))

            score = ct.matmul(q_tile, k_tile_t)
            score = ct.astype(score, ct.float32) * scale

            col = k_tile_idx * tile_n + ct.arange(tile_n, dtype=ct.int32)
            col = ct.expand_dims(col, 0)
            col = ct.expand_dims(col, 0)
            key_in_bounds = col < seq_len
            valid = row_in_bounds & key_in_bounds
            if causal:
                valid = valid & (row >= col)

            score = ct.where(valid, score, _NEG_LARGE)

            tile_max = ct.max(score, axis=2, keepdims=True)
            m_next = ct.maximum(m_i, tile_max)
            alpha = ct.exp(m_i - m_next)

            p = ct.exp(score - m_next)
            l_i = l_i * alpha + ct.sum(p, axis=2, keepdims=True)

            v_tile = ct.load(v, index=(bh_idx, k_tile_idx, 0), shape=(1, tile_n, head_dim))
            p_lowp = ct.astype(p, dtype)
            acc_update = ct.matmul(p_lowp, v_tile)
            acc = acc * alpha + ct.astype(acc_update, ct.float32)
            m_i = m_next

        safe_l = ct.where(row_in_bounds, l_i, 1.0)
        # Row-wise reciprocal avoids a full per-element divide over head_dim.
        inv_l = 1.0 / safe_l
        out_tile = acc * inv_l
        out_tile = ct.where(row_in_bounds, out_tile, 0.0)
        out_tile = ct.astype(out_tile, dtype)
        ct.store(out, index=(bh_idx, q_tile_idx, 0), tile=out_tile)

    return flash_fwd_kernel_fp32acc


def _launch_cutile_kernel(
    ct: Any,
    cupy_mod: Any,
    kernel: Any,
    grid: tuple[int, int, int],
    args: tuple[Any, ...],
) -> None:
    torch_mod = _runtime.get_torch_module()
    sync_mode = os.getenv("TILEDATTN_SYNC_MODE", "async").strip().lower()
    stream = cupy_mod.cuda.get_current_stream()
    if sync_mode == "strict":
        torch_mod.cuda.synchronize()
    ct.launch(stream, grid, kernel, args)
    if sync_mode in {"strict", "post"}:
        stream.synchronize()
    elif sync_mode == "async":
        pass
    else:
        raise ValueError(
            f"Unsupported TILEDATTN_SYNC_MODE={sync_mode!r}. "
            "Use one of: strict, post, async."
        )


def run_flash_fwd(
    q,
    k,
    v,
    *,
    causal: bool,
    scale: float,
):
    torch_mod = _runtime.get_torch_module()
    cupy_mod = _runtime.get_cupy_module()
    ct = _runtime.get_cutile_module()
    tile_m, tile_n = _resolve_tile_config()
    accum_mode = _resolve_accum_mode()
    opt_level, occupancy, num_ctas = _resolve_kernel_options()

    batch, heads, seq_len, head_dim = map(int, q.shape)
    bh = batch * heads

    kernel_head_dim = 1 << (head_dim - 1).bit_length()
    pad_dim = kernel_head_dim - head_dim

    q_bh = q.contiguous().reshape(bh, seq_len, head_dim)
    k_bh = k.contiguous().reshape(bh, seq_len, head_dim)
    v_bh = v.contiguous().reshape(bh, seq_len, head_dim)
    if pad_dim > 0:
        pad = (0, pad_dim)
        q_bh = torch_mod.nn.functional.pad(q_bh, pad)
        k_bh = torch_mod.nn.functional.pad(k_bh, pad)
        v_bh = torch_mod.nn.functional.pad(v_bh, pad)

    # Keep K as a transpose view to avoid a per-call materialization copy.
    k_t = k_bh.transpose(1, 2)

    out_bh = torch_mod.empty_like(q_bh)

    out_ct_dtype = _ct_dtype_for_torch_dtype(ct, q.dtype)
    kernel_key = (
        tile_m,
        tile_n,
        kernel_head_dim,
        str(q.dtype),
        causal,
        accum_mode,
        opt_level,
        occupancy if occupancy is not None else -1,
        num_ctas if num_ctas is not None else -1,
    )
    kernel = get_kernel(
        kernel_key,
        lambda: make_flashattn_fwd_kernel(
            tile_m,
            tile_n,
            kernel_head_dim,
            dtype=out_ct_dtype,
            causal=causal,
            accum_mode=accum_mode,
            opt_level=opt_level,
            occupancy=occupancy,
            num_ctas=num_ctas,
        ),
    )

    grid = (bh, (seq_len + tile_m - 1) // tile_m, 1)
    _launch_cutile_kernel(ct, cupy_mod, kernel, grid, (q_bh, k_t, v_bh, out_bh, float(scale)))

    out = out_bh[:, :, :head_dim] if pad_dim > 0 else out_bh
    return out.reshape(batch, heads, seq_len, head_dim)
