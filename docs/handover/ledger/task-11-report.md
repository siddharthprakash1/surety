# Task 11 report: live harness skeleton

**Status:** DONE_WITH_CONCERNS
**Branch:** `task-11` (worktree `.claude/worktrees/task-11`, based on `m1` + merge of `worktree-agent-a542998465fe6ada9`)
**Commit:** `11308f8` feat: live harness skeleton
**Tests:** `uv run pytest -q` → 94 passed, 1 skipped (live Playwright test; needs `SURETY_LIVE=1` + playwright)

## Files
- `src/surety/env/{page,base,playwright_env,browsergym_env,__init__}.py`
- `src/surety/compile/{page_compiler,__init__}.py` (`NONE_OF_THESE` constant exported)
- `src/surety/trace/{jsonl,__init__}.py`
- `src/surety/loop/{runner,__init__}.py`
- `src/surety/supervise/{client,prompts,__init__}.py`: `Supervisor.slot_value`, `MockSupervisor(answer, slots=)`, `build_slot_prompt`; `choose` now shares `_complete` with `slot_value`
- `scripts/run_miniwob.py`
- Tests: `test_page_compiler.py`, `test_trace.py`, `test_runner.py` (7), `test_envs.py` (new, offline), `test_supervisor.py` (+3), `integration/test_playwright_static.py`
- `pyproject.toml` `live` extra: + `browsergym-miniwob>=0.13`, `gymnasium>=0.29`; `uv.lock` updated (resolved only; default sync installs none of them, verified)

## Decisions (coordinator rulings)
- Q1: a confident `act` on `none_of_these` is trusted: no re-ask, it scrolls, and it counts toward the done streak. Re-ask once with the full view only when the escalate path's supervisor returns `none_of_these` or `None`. If the re-ask is `none_of_these`/`None` again, scroll.
- Q2: count consecutive identical executed actions `(kind, element_id, text)`. The 3rd occurrence escalates to `supervisor.choose(state, None)` and acts on that answer (a different answer resets the streak). The 4th occurrence stops with `stuck`.
- Q3: after `env.act`, if `getattr(env, "terminated", False)` is true, stop with `terminated`.
- A `None` from the escalate path's first answer is handled like `none_of_these`.

## Decisions (mine, small)
- Two consecutive `none_of_these` → `env.act(Action("done"))` then `stop_reason="done"` (BrowserGym sends `send_msg_to_user('done')`; Playwright no-op).
- Done streak counts only explicit `none_of_these`; a final `None` scrolls and resets the streak.
- `compile_page` meta also carries `"task"` (used by the slot prompt; falls back to `instructions`).
- `slot_value` calls are not counted in `supervisor_calls` and have no usage recorded, because the specified signature returns only `str | None`.
- Trace/step rows also include executor `probs`, which the live calibration-gap measurement needs. Typed text is never written to the trace. It does appear in `history`, which is prompt-only.
- PlaywrightEnv: the `role` attr wins over the tag. Non-text `<input type=checkbox|submit|radio|…>` gets the type as its role, so the runner clicks it instead of typing into it. Date and time inputs count as typeable. Visibility uses one `evaluate_all` call; handles are `locator.nth(i)`.
- BrowserGymEnv imports `browsergym.<prefix>` to register the env ids. Page text joins the `StaticText` names. The `checked` property becomes the value for checkbox/radio. Action args use `repr()` quoting. `type` skips `press` if `fill` already terminated. `reset(seed=None)`.
- `run_miniwob.py`: wraps the policy in `CriticalOverride`, adds optional `--supervisor-url/--supervisor-model` (default `MockSupervisor(None)`), defaults `--base-url` to `http://127.0.0.1:8009`, and writes the trace to `runs/miniwob/<task>-<ts>.jsonl`.

## Concerns
- No browser was run (as instructed). The PlaywrightEnv and BrowserGymEnv paths are covered only by offline tests: the pure helpers, a fake gym env, and a JS syntax check with node. The live test is unexecuted.
- `conformal:<eps>` in the script is unfitted (q=inf), so it always escalates with the full set until calibration data is wired in.
- The design spec's "BrowserGym pinned to the commit with `webarena_verified_hard`" is not done. The `live` extra uses version lower bounds; the pin needs the commit hash.
- Port clash: the MiniWoB setup line serves on 8080, which is also the `mlx_lm.server` default for the supervisor.

---

## Fix round 1: commit `07beda3` fix(live): payment check on supervisor picks, viewport, combobox, enter

**Status:** DONE

### Changes
1. **Payment check on supervisor picks** (`runner.py`): every id the supervisor chooses (escalation, re-ask, stuck re-ask) is classified before acting. The detector is `policy.detector` (already a public attribute on `CriticalOverride`, so no gate change), else a module-level `CriticalDetector()`. Payment → `stop:payment`, no action. Other categories are acted on, and the row gets `critical:<cat>`.
2. **No duplicate full-view re-ask**: the re-ask happens only when `candidate_set` is a proper subset of the option ids.
3. **Combobox**:
   - Playwright: the tag decides for input/textarea/select. `input[type=search]` or `role=searchbox` → `searchbox`; other text inputs → `input`; non-text inputs → their type; `<a>` → `link`.
   - BrowserGym: a combobox with the `editable` property → `textbox`.
4. **Viewport**:
   - Playwright `_JS` keeps only elements with `r.bottom > 0 && r.top < innerHeight` and skips `el.disabled`.
   - BrowserGym skips `disabled` and filters by `extra_element_properties[bid].visibility < 0.5` when present.
5. **Enter only for searchboxes**: both envs record each element's role in `observe()` and press Enter after `fill` only for `searchbox`.

Minors:
- A `None` or empty slot value counts as `none_of_these` (scroll, done streak); nothing is filled with `""`.
- `slot_value` returns `(text, Usage)`. It runs after the stuck and payment checks, is counted in `supervisor_calls`, and is traced in `row["supervisor"]` with `call: "slot_value"`. Supervisor usage entries now carry `call`.
- The stuck key is now `(kind, element_id)`, because the slot text isn't known before the stuck check.
- The traced `url` drops the query and fragment.
- Checkbox/radio `value` = checked state, `"true"`/`"false"` in both envs; Playwright falls back to `aria-checked`.
- Element actions use `timeout=5_000`. Any `env.act` exception writes the row with `error: <ExceptionType>` (type only, since messages can echo typed text) and stops with `error`. BrowserGym raises on `last_action_error`.
- BrowserGym ROLES += spinbutton, switch, menuitemcheckbox, menuitemradio, slider, treeitem. `spinbutton` was added to the runner's type roles. The runner stops on `truncated`.
- `Element.href` was dropped. The MiniWoB setup lines use port 8765.
- `MockSupervisor` records `candidates` per `choose`; `calls` counts `choose` and `slot_value`.

Tests added or changed:
- `test_runner.py`:
  - supervisor-chosen payment stops without acting
  - supervisor-chosen `Send` is acted on and recorded
  - exact candidate subset is passed (recording MockSupervisor)
  - subset NONE re-asks with the full view, then scrolls
  - full-view NONE doesn't re-ask
  - a different answer on the 3rd repeat resets the stuck count
  - typed text and URL query never appear in the trace
  - missing slot → scroll then done
  - terminated/truncated
  - action error recorded and stops
- `test_envs.py`: role mapping (combobox input, search, submit, select, `a` → link), checked value, BrowserGym editable combobox, disabled/invisible filtering, spinbutton, Enter only for searchbox, `last_action_error` raises, truncated.
- `test_supervisor.py`: `slot_value` returns usage.
- The live Playwright test now also asserts that disabled and below-the-fold buttons are excluded.

### Test command and output
```
uv run pytest -q
111 passed, 1 skipped in 2.31s
```
The skip is `tests/integration/test_playwright_static.py`, which needs `SURETY_LIVE=1` and playwright. It was not run, because no browser is installed by design. The `_JS` snippet passes a node syntax check. `scripts/run_miniwob.py` exits 2 with the 8765 setup lines when `MINIWOB_URL` is unset, and completes a run against a faked BrowserGym env.

### Remaining concerns
- The env adapters are still unverified against a real browser.
- The BrowserGym visibility threshold of 0.5 is assumed to match BrowserGym's own visible-only filter; this was not checked against its source.
- A non-native `[role=combobox]` that isn't editable maps to `select`, and `select_option` will fail on it, ending the episode with `error`.
