# Surety

A computer-use agent harness where a small local decision model takes routine steps and a reasoning LLM supervises. Escalation carries an error guarantee: autonomous wrong actions stay at or below ε per step, kept online as websites change.

- Executor: [Kev-4B](https://github.com/jaredpalmer/kev) or [Clef-flash](https://huggingface.co/mlx-community/clef-flash-4bit) (open, Jev-compatible, run locally).
- Supervisor: any OpenAI-compatible chat model (default: local Qwen3.6-35B-A3B via MLX).
- Gate: debias → recalibrate → conformal set → critical-action override; adaptive conformal with audits online.

Status: M1 (offline calibration study on [WebJev](https://huggingface.co/datasets/Lexmount/WebJev)) in progress. Design: `docs/superpowers/specs/2026-11-06-surety-design.md`.

```bash
uv sync --group dev
uv run pytest -q
```
