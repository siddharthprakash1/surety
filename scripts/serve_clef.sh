#!/usr/bin/env bash
# Serve Clef-flash (MLX 4-bit) as a System One endpoint. Usage: scripts/serve_clef.sh [port]
set -euo pipefail
PORT="${1:-8010}"
SNAP="$(uv run --no-project --with huggingface_hub python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("mlx-community/clef-flash-4bit"))')"
cd "$SNAP"
exec uv run --no-project --with "mlx-vlm>=0.7.4,<0.8" --with huggingface_hub --with fastapi --with uvicorn python clef_mlx.py serve --port "$PORT"
