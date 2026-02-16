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
_DIRECT_HEAD_DIMS = frozenset({64, 128})
_CHUNKED_HEAD_DIM_PARTS = {
    96: (64, 32),
    160: (128, 32),
}


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
    tile_m = _get_env_int("TILEDATTN_TILE_M", None)
    tile_n = _get_env_int("TILEDATTN_TILE_N", None)
    if tile_m is None:
        tile_m = _DEFAULT_TM
    if tile_n is None:
        tile_n = _DEFAULT_TN
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


def _resolve_kernel_head_dim(head_dim: int) -> tuple[int, int]:
    """Return (kernel_head_dim, pad_dim) for direct-load kernels."""
    if head_dim in _DIRECT_HEAD_DIMS:
        return head_dim, 0
    kernel_head_dim = 1 << (head_dim - 1).bit_length()
    return kernel_head_dim, kernel_head_dim - head_dim


def _resolve_chunk_plan(head_dim: int) -> tuple[tuple[int, int], ...] | None:
    enabled = os.getenv("TILEDATTN_CHUNKED_HEAD_DIMS", "").strip()
    if enabled == "":
        return None
    try:
        enabled_dims = {int(x.strip()) for x in enabled.split(",") if x.strip() != ""}
    except ValueError as exc:
        raise ValueError(
            "Environment variable TILEDATTN_CHUNKED_HEAD_DIMS must be a comma-separated list of integers."
        ) from exc
    if head_dim not in enabled_dims:
        return None
    parts = _CHUNKED_HEAD_DIM_PARTS.get(head_dim)
    if parts is None:
        return None
    offset = 0
    plan: list[tuple[int, int]] = []
    for width in parts:
        plan.append((offset, width))
        offset += width
    return tuple(plan)


def _default_tile_config_for_shape(*, seq_len: int, head_dim: int, causal: bool) -> tuple[int, int]:
    """Heuristic defaults used only when tile env overrides are not provided."""
    tile_m = _DEFAULT_TM
    tile_n = _DEFAULT_TN
    # Mid/long non-causal D=128 favors wider N tiles in current tuning results.
    if (not causal) and head_dim == 128 and seq_len >= 2048:
        tile_n = 128
    return tile_m, tile_n


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
            if causal:
                # For causal mode, this query tile never attends beyond its last row index.
                causal_cols = ct.minimum((q_tile_idx + 1) * tile_m, seq_len)
                num_k_tiles = ct.cdiv(causal_cols, tile_n)

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
        if causal:
            # For causal mode, this query tile never attends beyond its last row index.
            causal_cols = ct.minimum((q_tile_idx + 1) * tile_m, seq_len)
            num_k_tiles = ct.cdiv(causal_cols, tile_n)

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


def make_flashattn_fwd_kernel_chunked(
    tile_m: int,
    tile_n: int,
    head_dim: int,
    *,
    chunk_plan: tuple[tuple[int, int], ...],
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
    if len(chunk_plan) != 2:
        raise ValueError(f"Chunked kernel currently expects exactly 2 chunks, got {chunk_plan!r}.")
    (off0, w0), (off1, w1) = chunk_plan

    if accum_mode == "fp16":

        @ct.kernel(**kernel_kwargs)
        def flash_fwd_kernel_fp16acc_chunked(q0, k0_t, v0, q1, k1_t, v1, out0, out1, scale):
            bh_idx = ct.bid(0)
            q_tile_idx = ct.bid(1)

            q0_tile = ct.load(q0, index=(bh_idx, q_tile_idx, 0), shape=(1, tile_m, w0))
            q1_tile = ct.load(q1, index=(bh_idx, q_tile_idx, 0), shape=(1, tile_m, w1))

            seq_len = q0.shape[1]
            num_k_tiles = ct.cdiv(seq_len, tile_n)
            if causal:
                causal_cols = ct.minimum((q_tile_idx + 1) * tile_m, seq_len)
                num_k_tiles = ct.cdiv(causal_cols, tile_n)

            row = q_tile_idx * tile_m + ct.arange(tile_m, dtype=ct.int32)
            row = ct.expand_dims(row, 1)
            row = ct.expand_dims(row, 0)
            row_in_bounds = row < seq_len

            m_i = ct.full((1, tile_m, 1), _NEG_LARGE, ct.float32)
            l_i = ct.zeros((1, tile_m, 1), ct.float32)
            acc0 = ct.zeros((1, tile_m, w0), dtype)
            acc1 = ct.zeros((1, tile_m, w1), dtype)

            for k_tile_idx in range(num_k_tiles):
                k0_tile_t = ct.load(k0_t, index=(bh_idx, 0, k_tile_idx), shape=(1, w0, tile_n))
                k1_tile_t = ct.load(k1_t, index=(bh_idx, 0, k_tile_idx), shape=(1, w1, tile_n))
                score = ct.astype(ct.matmul(q0_tile, k0_tile_t), ct.float32) + ct.astype(
                    ct.matmul(q1_tile, k1_tile_t), ct.float32
                )
                score = score * scale

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

                p_lowp = ct.astype(p, dtype)
                alpha_lowp = ct.astype(alpha, dtype)
                v0_tile = ct.load(v0, index=(bh_idx, k_tile_idx, 0), shape=(1, tile_n, w0))
                v1_tile = ct.load(v1, index=(bh_idx, k_tile_idx, 0), shape=(1, tile_n, w1))
                acc0 = acc0 * alpha_lowp + ct.matmul(p_lowp, v0_tile)
                acc1 = acc1 * alpha_lowp + ct.matmul(p_lowp, v1_tile)
                m_i = m_next

            safe_l = ct.where(row_in_bounds, l_i, 1.0)
            inv_l = 1.0 / safe_l
            inv_l_lowp = ct.astype(inv_l, dtype)
            out0_tile = ct.where(row_in_bounds, acc0 * inv_l_lowp, 0.0)
            out1_tile = ct.where(row_in_bounds, acc1 * inv_l_lowp, 0.0)
            ct.store(out0, index=(bh_idx, q_tile_idx, 0), tile=out0_tile)
            ct.store(out1, index=(bh_idx, q_tile_idx, 0), tile=out1_tile)

        return flash_fwd_kernel_fp16acc_chunked

    @ct.kernel(**kernel_kwargs)
    def flash_fwd_kernel_fp32acc_chunked(q0, k0_t, v0, q1, k1_t, v1, out0, out1, scale):
        bh_idx = ct.bid(0)
        q_tile_idx = ct.bid(1)

        q0_tile = ct.load(q0, index=(bh_idx, q_tile_idx, 0), shape=(1, tile_m, w0))
        q1_tile = ct.load(q1, index=(bh_idx, q_tile_idx, 0), shape=(1, tile_m, w1))

        seq_len = q0.shape[1]
        num_k_tiles = ct.cdiv(seq_len, tile_n)
        if causal:
            causal_cols = ct.minimum((q_tile_idx + 1) * tile_m, seq_len)
            num_k_tiles = ct.cdiv(causal_cols, tile_n)

        row = q_tile_idx * tile_m + ct.arange(tile_m, dtype=ct.int32)
        row = ct.expand_dims(row, 1)
        row = ct.expand_dims(row, 0)
        row_in_bounds = row < seq_len

        m_i = ct.full((1, tile_m, 1), _NEG_LARGE, ct.float32)
        l_i = ct.zeros((1, tile_m, 1), ct.float32)
        acc0 = ct.zeros((1, tile_m, w0), ct.float32)
        acc1 = ct.zeros((1, tile_m, w1), ct.float32)

        for k_tile_idx in range(num_k_tiles):
            k0_tile_t = ct.load(k0_t, index=(bh_idx, 0, k_tile_idx), shape=(1, w0, tile_n))
            k1_tile_t = ct.load(k1_t, index=(bh_idx, 0, k_tile_idx), shape=(1, w1, tile_n))
            score = ct.astype(ct.matmul(q0_tile, k0_tile_t), ct.float32) + ct.astype(
                ct.matmul(q1_tile, k1_tile_t), ct.float32
            )
            score = score * scale

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

            p_lowp = ct.astype(p, dtype)
            v0_tile = ct.load(v0, index=(bh_idx, k_tile_idx, 0), shape=(1, tile_n, w0))
            v1_tile = ct.load(v1, index=(bh_idx, k_tile_idx, 0), shape=(1, tile_n, w1))
            acc0 = acc0 * alpha + ct.astype(ct.matmul(p_lowp, v0_tile), ct.float32)
            acc1 = acc1 * alpha + ct.astype(ct.matmul(p_lowp, v1_tile), ct.float32)
            m_i = m_next

        safe_l = ct.where(row_in_bounds, l_i, 1.0)
        inv_l = 1.0 / safe_l
        out0_tile = ct.astype(ct.where(row_in_bounds, acc0 * inv_l, 0.0), dtype)
        out1_tile = ct.astype(ct.where(row_in_bounds, acc1 * inv_l, 0.0), dtype)
        ct.store(out0, index=(bh_idx, q_tile_idx, 0), tile=out0_tile)
        ct.store(out1, index=(bh_idx, q_tile_idx, 0), tile=out1_tile)

    return flash_fwd_kernel_fp32acc_chunked


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
    accum_mode = _resolve_accum_mode()
    opt_level, occupancy, num_ctas = _resolve_kernel_options()

    batch, heads, seq_len, head_dim = map(int, q.shape)
    bh = batch * heads
    if os.getenv("TILEDATTN_TILE_M") or os.getenv("TILEDATTN_TILE_N"):
        tile_m, tile_n = _resolve_tile_config()
    else:
        tile_m, tile_n = _default_tile_config_for_shape(
            seq_len=seq_len, head_dim=head_dim, causal=causal
        )

    chunk_plan = _resolve_chunk_plan(head_dim)
    if chunk_plan is not None:
        kernel_head_dim = head_dim
        pad_dim = 0
    else:
        kernel_head_dim, pad_dim = _resolve_kernel_head_dim(head_dim)

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
        "chunked" if chunk_plan is not None else "direct",
        str(q.dtype),
        causal,
        accum_mode,
        opt_level,
        occupancy if occupancy is not None else -1,
        num_ctas if num_ctas is not None else -1,
    )
    if chunk_plan is None:
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
    else:
        kernel = get_kernel(
            kernel_key,
            lambda: make_flashattn_fwd_kernel_chunked(
                tile_m,
                tile_n,
                kernel_head_dim,
                chunk_plan=chunk_plan,
                dtype=out_ct_dtype,
                causal=causal,
                accum_mode=accum_mode,
                opt_level=opt_level,
                occupancy=occupancy,
                num_ctas=num_ctas,
            ),
        )

    grid = (bh, (seq_len + tile_m - 1) // tile_m, 1)
    if chunk_plan is None:
        args = (q_bh, k_t, v_bh, out_bh, float(scale))
    else:
        (off0, w0), (off1, w1) = chunk_plan
        if off0 != 0 or off1 != w0 or (w0 + w1) != head_dim:
            raise ValueError(f"Unexpected chunk plan for head_dim={head_dim}: {chunk_plan!r}")
        q0 = q_bh[:, :, :w0]
        q1 = q_bh[:, :, w0:]
        k0_t = k_bh[:, :, :w0].transpose(1, 2)
        k1_t = k_bh[:, :, w0:].transpose(1, 2)
        v0 = v_bh[:, :, :w0]
        v1 = v_bh[:, :, w0:]
        out0 = out_bh[:, :, :w0]
        out1 = out_bh[:, :, w0:]
        args = (q0, k0_t, v0, q1, k1_t, v1, out0, out1, float(scale))
    _launch_cutile_kernel(ct, cupy_mod, kernel, grid, args)

    out = out_bh[:, :, :head_dim] if pad_dim > 0 else out_bh
    return out.reshape(batch, heads, seq_len, head_dim)
