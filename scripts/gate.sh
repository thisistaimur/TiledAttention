#!/usr/bin/env bash
set -euo pipefail

# Gate script for local verification.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> Python:"
python --version

echo "==> Installing (editable + dev)..."
python -m pip install -e ".[dev]" >/dev/null

echo "==> Ruff (lint)..."
python -m ruff check .

echo "==> Black (format check)..."
python -m black --check .

echo "==> Pytest..."
python -m pytest -q

echo "Gate passed."
