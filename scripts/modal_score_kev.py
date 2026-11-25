"""Score a plan's WebJev steps with a Kev checkpoint on a Modal GPU. kev.serve and scripts/offline_collect.py run in one
container (this laptop cannot serve Kev), so a fine-tuned checkpoint never leaves the surety-kev-ft volume.

    # parity first: the checkpoint M1 scored, on 100 of M1's steps, against M1's rows (scripts/score_parity.py)
    uvx --from modal==1.5.5 modal run scripts/modal_score_kev.py --run zero-shot --plan artifacts/m1/plan.json --name m1-parity --limit 100
    uvx --from modal==1.5.5 modal run scripts/modal_score_kev.py --run zero-shot --plan artifacts/m2/plan.json --name m2-zero-shot
    uvx --from modal==1.5.5 modal run scripts/modal_score_kev.py --run webjev-v1 --plan artifacts/m2/plan.json --name m2-webjev-v1
    uvx --from modal==1.5.5 modal volume get surety-kev-ft scores/m2-webjev-v1 runs/scores/m2-webjev-v1

--run: zero-shot (the pinned checkpoint M1 scored), a Hub id (repo@revision), or the --name of a modal_train_kev.py run.
Re-running with the same --name resumes: offline_collect skips the rows already on the volume.
"""
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "kev"))
import modal_app as kev  # noqa: E402  (git clone https://github.com/jaredpalmer/kev third_party/kev)

ZERO_SHOT = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"   # modal_train_kev.py's INIT
PORT = 8009
SRC = "/root/surety_src"
app = modal.App("surety-kev-score")
vol = modal.Volume.from_name("surety-kev-ft", create_if_missing=True)   # modal_train_kev.py writes its runs here
image = (kev.image.add_local_python_source("modal_app")
         .add_local_dir(ROOT / "src" / "surety", f"{SRC}/surety", ignore=["**/__pycache__/**"])
         .add_local_file(ROOT / "scripts" / "offline_collect.py", f"{SRC}/offline_collect.py")
         .add_local_dir(ROOT / "data" / "webjev" / "data", "/data/webjev"))


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=4 * 3600,
              volumes={"/out": vol, kev.HF_MOUNT: kev.hf_cache})
def score(run: str, plan: str, name: str, extra: list[str]) -> dict:
    import os
    sys.path.insert(0, SRC)
    from surety.eval.serve_collect import serve_and_collect

    served = run if "/" in run else f"/out/{run}"
    if served.startswith("/out/") and not Path(served, "head.pt").exists():
        raise FileNotFoundError(f"no checkpoint at {served} on the surety-kev-ft volume (modal_train_kev.py --name ...)")
    out = Path("/out/scores") / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan)
    rows = out / "rows.jsonl"
    try:
        card = serve_and_collect(
            [sys.executable, "-m", "kev.serve", "--run", served, "--port", str(PORT), "--device", "cuda"], PORT,
            [sys.executable, f"{SRC}/offline_collect.py", "--plan", str(out / "plan.json"), "--root", "/data/webjev",
             "--base-url", f"http://127.0.0.1:{PORT}", "--out", str(rows), *extra],
            expect_run=served, log=out / "serve.log", env={**os.environ, "PYTHONPATH": f"/root:{SRC}"}, cwd="/root")
        (out / "server.json").write_text(json.dumps(card, indent=1))
        return {"run": served, "rows": len(rows.read_text().splitlines()), "server": card}
    finally:
        vol.commit()
        kev.hf_cache.commit()


@app.local_entrypoint()
def main(run: str, plan: str, name: str, limit: int = 0, splits: str = "cal,test_in,test_shift", gpu: str = "H100") -> None:
    text = Path(plan).read_text()
    planned = sum(len(json.loads(text)[s]) for s in splits.split(","))
    extra = ["--splits", splits] + (["--limit", str(limit)] if limit else [])
    res = score.with_options(gpu=gpu).remote(ZERO_SHOT if run == "zero-shot" else run, text, name, extra)
    print(json.dumps(res, indent=1))
    print(f"{res['rows']} rows of {min(planned, limit) if limit else planned} planned (missing rows were refused or failed;"
          f" re-run to retry). Fetch: uvx --from modal==1.5.5 modal volume get surety-kev-ft scores/{name} runs/scores/{name}")
