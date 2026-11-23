#!/usr/bin/env bash
# LoRA fine-tune of Kev-4B on runs/kev_ft/train.jsonl (scripts/make_kev_train.py), then Kev's benchmark on heldout.jsonl.
# Extra args go to kev.train and override the flags below (argparse keeps the last), e.g. --max_steps 5 for a smoke run.
# On Apple MPS a 5-step trial logged no step in 20 min (reference Qwen3.5 kernels): use scripts/modal_train_kev.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEV="$ROOT/third_party/kev"
OUT="$ROOT/runs/kev_ft"
PORT="${KEV_PORT:-8011}"
[[ -f "$OUT/train.jsonl" && -f "$OUT/heldout.jsonl" ]] || { echo "run scripts/make_kev_train.py first" >&2; exit 1; }
cd "$KEV"

# --dtype bf16 is CUDA-only in kev.train; --weights_dtype bf16 halves backbone memory on MPS.
# --max_state 7552: compact WebJev states run ~1k tokens (p99 ~4k); the 384 default admits ~6% of records.
uv run python -m kev.train --data "$OUT/train.jsonl" --base Qwen/Qwen3.5-4B-Base \
    --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b --init_from jaredpalmer/kev-4b \
    --epochs 2 --lr 2e-5 --batch 1 --accum 8 --weights_dtype bf16 --checkpointing 1 --max_state 7552 \
    --device mps --out "$OUT/run" "$@"

# kev.benchmark --run scores --data within the 384-token training context and skips the rest, so score through the server.
SERVER=""
trap '[[ -n "$SERVER" ]] && kill -- -"$SERVER" 2>/dev/null || true' EXIT
set -m   # own process group, so the trap also stops python under uv
uv run --extra serve python -m kev.serve --run "$OUT/run" --port "$PORT" > "$OUT/serve-eval.log" 2>&1 &
SERVER=$!
set +m
for _ in $(seq 120); do
    curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
    kill -0 "$SERVER" 2>/dev/null || { echo "server exited; see $OUT/serve-eval.log" >&2; exit 1; }
    sleep 5
done
curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "server not ready after 10 min" >&2; exit 1; }
uv run python -m kev.benchmark --remote "http://127.0.0.1:$PORT" --data "$OUT/heldout.jsonl" --out "$OUT/eval"

echo "serve the fine-tuned checkpoint:"
echo "  cd $KEV && uv run --extra serve python -m kev.serve --run $OUT/run --port $PORT"
echo "zero-shot baseline on the same heldout (Kev-4B server on 8009):"
echo "  cd $KEV && uv run python -m kev.benchmark --remote http://127.0.0.1:8009 --data $OUT/heldout.jsonl --out $OUT/eval-base"
