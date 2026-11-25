"""Kev-4B LoRA fine-tune on WebJev on a Modal H100, with Kev's own image (locked deps + fla/causal-conv1d kernels).

Kev's kev-finetune skill keeps kev.train's 384-token state default, which drops ~94% of WebJev records; this passes
--max_state 7552 (the released Kev-4B training length) with the recipe of scripts/train_kev.sh.

    uv run python scripts/make_kev_train.py                     # runs/kev_ft/{train,heldout}.jsonl from M2's training sites
    uvx modal setup                                             # once; H100s need a payment method on the workspace
    uvx --from modal==1.5.5 modal run scripts/modal_train_kev.py --max-steps 5          # smoke: a few minutes
    uvx --from modal==1.5.5 modal run --detach scripts/modal_train_kev.py --name webjev-v1   # full: 2 epochs, ~1-2 GPU h
Names must be new. --detach keeps the run going if this laptop sleeps or disconnects (a LoRA run cannot resume).
The launcher refuses training data not built from artifacts/m2/plan.json. Then score the run where it lies, on the
surety-kev-ft volume: scripts/modal_score_kev.py --run webjev-v1 --plan artifacts/m2/plan.json.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "kev"))
import modal_app as kev  # noqa: E402  (git clone https://github.com/jaredpalmer/kev third_party/kev)

DATA = ROOT / "runs" / "kev_ft"
app = modal.App("surety-kev-ft")
out = modal.Volume.from_name("surety-kev-ft", create_if_missing=True)
# modal_app itself must ship too: the container imports this file, which imports it (Modal 1.x adds no imports)
image = kev.image.add_local_python_source("modal_app").add_local_file(DATA / "train.jsonl", "/data/train.jsonl")
# init = the hub main M1 served (unchanged since 2026-10-01: Kev 1.0 weights, round 10). Architecture flags (lora 16,
# head_dim 256, lora_targets all, no option isolation) are kev.train's defaults and match its training_config.json,
# which warm_start checks before loading.
INIT = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
TRAIN = ["--data", "/data/train.jsonl", "--base", "Qwen/Qwen3.5-4B-Base",
         "--base_revision", "1001bb4d826a52d1f399e183466143f4da7b741b", "--init_from", INIT,
         "--epochs", "2", "--lr", "2e-5", "--batch", "1", "--accum", "8", "--checkpointing", "1",
         "--max_state", "7552", "--device", "cuda", "--dtype", "bf16"]


@app.function(image=image, gpu="H100", cpu=4, memory=65536, timeout=4 * 3600,
              volumes={"/out": out, kev.HF_MOUNT: kev.hf_cache})
def train(name: str, extra: list[str]) -> None:
    try:
        subprocess.run([sys.executable, "-m", "kev.train", *TRAIN, "--out", f"/out/{name}", *extra], check=True,
                       cwd="/root", env={**os.environ, "PYTHONPATH": "/root"})
    finally:
        out.commit()
        kev.hf_cache.commit()


@app.local_entrypoint()
def main(name: str = "", max_steps: int = 0) -> None:
    if not (DATA / "train.jsonl").exists():
        sys.exit("run scripts/make_kev_train.py first")
    stats = json.loads((DATA / "stats.json").read_text()) if (DATA / "stats.json").exists() else {}
    m2 = json.loads((ROOT / "artifacts" / "m2" / "plan.json").read_text())
    if stats.get("plan_sha256") != hashlib.sha256(json.dumps(m2, sort_keys=True).encode()).hexdigest():  # webjev.plan_digest
        sys.exit("runs/kev_ft was not built from artifacts/m2/plan.json (a fine-tune on M1's plan would train on its "
                 "calibration sites): run scripts/make_kev_train.py --plan artifacts/m2/plan.json")
    name = name or f"{'smoke' if max_steps else 'webjev'}-{time.strftime('%Y%m%d-%H%M%S')}"  # kev.train refuses an existing --out
    train.remote(name, ["--max_steps", str(max_steps)] if max_steps else [])
    print(f"done: /out/{name} on the surety-kev-ft volume. Score it there: uvx --from modal==1.5.5 modal run --detach "
          f"scripts/modal_score_kev.py --run {name} --plan artifacts/m2/plan.json --name m2-{name}")
