from __future__ import annotations

import argparse
from typing import Any

def parse_dtype(name: str, *, torch_mod: Any) -> Any:
    normalized = name.strip().lower()
    if normalized == "float16":
        return torch_mod.float16
    if normalized == "bfloat16":
        return torch_mod.bfloat16
    raise ValueError(f"Unsupported dtype={name!r}. Use float16 or bfloat16.")


def run_attention(
    *,
    method: str,
    q,
    k,
    v,
    causal: bool,
) -> Any:
    import torch

    from tiledattention.sdpa import sdpa as tiled_sdpa

    if method == "tiledattention":
        return tiled_sdpa(q, k, v, causal=causal)
    if method == "torch_sdpa":
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
        )
    raise ValueError(f"Unsupported method={method!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one attention workload for Nsight Compute profiling."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["tiledattention", "torch_sdpa"],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this workload.")

    if args.batch <= 0 or args.heads <= 0 or args.seq_len <= 0 or args.head_dim <= 0:
        raise ValueError("batch/heads/seq-len/head-dim must be positive.")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be > 0.")

    dtype = parse_dtype(args.dtype, torch_mod=torch)
    torch.manual_seed(args.seed)

    q = torch.randn(
        (args.batch, args.heads, args.seq_len, args.head_dim),
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        (args.batch, args.heads, args.seq_len, args.head_dim),
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn(
        (args.batch, args.heads, args.seq_len, args.head_dim),
        device="cuda",
        dtype=dtype,
    )

    for _ in range(args.warmup):
        _ = run_attention(method=args.method, q=q, k=k, v=v, causal=args.causal)
    torch.cuda.synchronize()

    for _ in range(args.repeats):
        _ = run_attention(method=args.method, q=q, k=k, v=v, causal=args.causal)
    torch.cuda.synchronize()

    print(
        f"[workload] method={args.method} B={args.batch} H={args.heads} "
        f"S={args.seq_len} D={args.head_dim} dtype={args.dtype} causal={args.causal}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
