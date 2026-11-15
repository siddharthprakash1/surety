# Task 12: Kev fine-tune data prep

**Status:** DONE. Branch `task-12`, commit `062f4aa` (not merged or pushed).
**Tests:** `uv run pytest -q`: 89 passed (3 new in `tests/test_kev_train.py`).

## Files
- `src/surety/eval/kev_train.py`: `build_records`, `write_jsonl`
- `scripts/make_kev_train.py`: writes train/heldout/stats
- `scripts/train_kev.sh`: train, then benchmark (not run)
- `tests/test_kev_train.py`

## Kev format: what I checked
- **Record shape.** One line per record: `{"state", "questions": {qid: {type, instructions, criteria, label, src}}}`. Source: `kev/data.py` docstring and `load_records`, plus the README "By Hand" section.
- **Choice labels.** For `choice`, `label` is the option name (README: "For `choice` the label is the option name"). `materialize` turns it into `keys.index(label)`. I ran `load_records` and `materialize` on all 8,500 real records: every label resolves.
- **`src` is optional.** `load_records` defaults it to `custom_choice`. The benchmark groups its per-task metrics by `src`, so `"webjev"` puts both task types in one group. `rows.json` still keeps `question` (= `action` or `element`) for a per-type split.
- **`--data` takes the labelled-request shape**, not the materialized record. Train help text: "your own labelled requests ... (see kev.data.load_records)".
- **One addition: `_meta: {"id": <webjev id>}`.** `load_records` merges a caller's `_meta` over its defaults. `api_request` and `materialize` drop it, so the model never sees it. With it, benchmark `predictions.jsonl` and `rows.json` carry WebJev ids that join with surety results. It is also how `stats.json` counts sites.
- **qid does not reach the model.** `to_record` encodes only `instructions` and `criteria`. So the training qids (`action`/`element`) and surety's serving qid (`q`) feed identical inputs. Criteria `{o.id: o.label}` match `SystemOneClient.choice`.

## Data (`runs/kev_ft/stats.json`)
| | n | action / element | sites | mean state chars | options p50 / p90 / max |
|---|---|---|---|---|---|
| train | 8000 | 4000 / 4000 | 784 | 3538 | 9 / 64 / 64 |
| heldout | 500 | 250 / 250 | 175 | 3491 | 8.5 / 64 / 64 |

Leak check on the real files:
- 0 ids overlap with `cal ∪ test_in ∪ test_shift`.
- 0 rows come from `held_sites`.
- All 8,500 ids are unique.

## Train flags (`scripts/train_kev.sh`)
| Flag | Why |
|---|---|
| `--data --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b --epochs 2 --lr 2e-5 --batch 1 --accum 8 --checkpointing 1` | README "By Hand" command: "`--epochs 2 --lr 2e-5 --batch 1 --accum 8 --dtype bf16 --checkpointing 1`"; "`2e-5` is a good start" |
| `--base_revision 1001bb4d…` | The pin in the README's Kev-4B recipe and in kev-4b's `training_config.json`. It is also the only cached snapshot (no `refs/main`), so this avoids a download. |
| `--weights_dtype bf16`, not `--dtype bf16` | `train.py` rejects off-CUDA bf16 with "`--dtype bf16 requires --device cuda`". `--weights_dtype` help says "bf16 halves memory". `warm_start` does not compare `weights_dtype` (not in `COMPAT_FIELDS`), and `checkpoint.py` loads bf16-backbone adapters the same way. |
| `--device mps` | README: an explicit accelerator "fails closed", where `auto` may fall back to CPU |
| `--max_state 7552` | Kev's default 384-token state admits only 5.6% of records. Measured with Kev's own `fits` on 1,000 records: 384 → 56, 4096 → 944, 7552 → 982 fit. 7552 is the limit kev-4b's own skill fine-tune used (`training_config.json`; README "fine-tunes on states of up to 7,552 tokens"). Trailing args override any flag. |

**Eval runs through the server.** The script starts `kev.serve` on port 8011, runs `kev.benchmark --remote … --data heldout.jsonl --out runs/kev_ft/eval`, then stops the server. Why not `--run`: with `--data`, the local benchmark uses the 384-token training context and skips longer records (`skip_overlong=True`), so it would score only ~6% of the heldout set. Serving accepts states up to 64k tokens. The script also prints the serve command and the zero-shot baseline benchmark command (against port 8009).

## Concerns
- **Mac training time is untested and likely long.** The kev-finetune skill notes Qwen3.5 has no DeltaNet kernels for MPS, and the job is 16k forward/backward passes at about 1k tokens each. Peak memory at 7.5k-token states is also unmeasured. Smoke test first with `scripts/train_kev.sh --max_steps 5`. If memory runs out, append `--max_state 4096` (drops about 6% of records).
- **No `--replay`.** Kev's own skill fine-tune mixed in 4,000 suite records to avoid forgetting. That doesn't matter if Kev is only used for WebJev decisions.
- **`test_in` comes from the same sites as training** (by design). Only ids are disjoint, so the same page can appear as an action question in training and an element question in `test_in`. In-distribution gains will look larger than the site-shift gains.
- `_meta` is the one addition to the specified record shape (reason above).
- `kev_train.py` reuses the private `webjev._draw` for round-robin balancing.

## Fix round 1 (commit `b85672e`)

**Changes**
- `build_records`: heldout is now whole sites. The eligible sites are sorted, then shuffled with `Random(seed)`. Sites are taken in that order until their instance count reaches `n_heldout`. Heldout is drawn only from those sites and train only from the rest. Task-type balance and the plan/held-site exclusions are unchanged.
- `webjev._draw` is renamed to the public `draw_balanced`. Nothing else used the old name, so there is no alias.
- `make_kev_train.py`: wrapped in `main()`. Stats gain top-level `train_sites`, `heldout_sites` and `shared_sites`, plus a `shortfall` per split.
- `train_kev.sh`:
  - The port can be set with `KEV_PORT` (default 8011).
  - The `trap` is set before the server starts. The server runs in its own process group (`set -m`) and the trap kills the whole group (`kill -- -PID`), so the python child under `uv` stops too. I checked this with a stand-in child process: it was killed.
  - Readiness polling uses `curl -m 5`, at most 120 tries 5 s apart, then exits with an error.
- Tests: added `test_build_records_heldout_sites_disjoint_from_train_sites`. Two existing tests changed for whole-site heldout: the exclusion test now checks the full eligible set with `n_heldout=0`, and the shape test now uses 2 sites.

**Tests:** `uv run pytest -q`: 90 passed.

**New stats** (real root/plan, `runs/kev_ft/` overwritten)
| | n | shortfall | action / element | mean state chars | options p50 / p90 / max |
|---|---|---|---|---|---|
| train | 8000 | 0 | 4000 / 4000 | 3480 | 9 / 64 / 64 |
| heldout | 500 | 0 | 250 / 250 | 3288 | 10 / 64 / 64 |

- Sites: 777 in train, 5 in heldout, 0 shared.
- Leak check: 0 overlap with the plan's `cal`, `test_in` and `test_shift` ids, and 0 rows from held-out sites. All 8,500 ids are unique.

**Concern:** taking the smallest set of sites makes heldout narrow. It has 5 sites: instructure.com 299, dell.com 163, cabinetmagazine.org 18, printerdirectory.usps.com 11, megayachtnews.com 9. Heldout accuracy will mostly reflect two sites. The fix would be a per-site cap or a minimum number of heldout sites.

**Fix round 2 (commit `2fd2f84`):** `heldout_site_cap=60` limits each heldout site to 60 instances. Tests: 90 passed. Regenerated: train 8000 (4000/4000) from 771 sites; heldout 500 (256 action / 244 element) from 27 sites; 0 sites shared; 0 plan overlap.
Heldout per site: the largest are instructure.com 59, dell.com 58, wolframalpha.com 57, open.spotify.com 46 and healthline.com 44. The other 22 sites have 1–31 each.
