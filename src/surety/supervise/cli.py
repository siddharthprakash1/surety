"""Supervisor flags shared by the live scripts (run_miniwob, run_h4); api_key and parse_params also serve H3."""
import json
import os
import sys
from pathlib import Path

from surety.supervise.client import MockSupervisor, Supervisor


def api_key(name: str | None, dotenv: Path = Path(".env")) -> str | None:
    """The key in environment variable `name`, else a `name=value` line in ./.env (gitignored): a key typed once into
    that file serves every run without appearing in commands or logs. Exits if `name` is given but found in neither."""
    if not name:
        return None
    if key := os.environ.get(name):
        return key
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8-sig").splitlines():
            k, sep, v = line.strip().removeprefix("export ").partition("=")
            if sep and k.strip() == name and (v := v.strip().strip("'\"")):
                return v
    sys.exit(f"{name} is not set (environment or .env)")


def parse_params(items: list[str] | None) -> dict | None:
    """KEY=VALUE strings -> extra request fields; VALUE is read as JSON when it parses (2, true, {...}), else text."""
    if not items:
        return None
    out = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep or not k:
            sys.exit(f"a request field is KEY=VALUE, got {item!r}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def add_supervisor_args(ap) -> None:
    ap.add_argument("--supervisor-url", help="OpenAI-compatible server; omitted = a supervisor that never answers (smoke)")
    ap.add_argument("--supervisor-model", default="default")
    ap.add_argument("--supervisor-api-key-env", help="environment variable (or ./.env entry) holding the API key")
    ap.add_argument("--supervisor-json", action="store_true", help="constrain replies to an option id or NONE")
    ap.add_argument("--supervisor-max-tokens", type=int, default=1024,
                    help="reasoning models (gpt-oss) spend tokens on hidden reasoning before the answer")
    ap.add_argument("--supervisor-reasoning", choices=["low", "medium", "high"])
    ap.add_argument("--supervisor-no-temperature", action="store_true", help="OpenAI reasoning models reject it")
    ap.add_argument("--supervisor-token-param", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    ap.add_argument("--supervisor-param", action="append", metavar="KEY=VALUE",
                    help="extra request field, repeatable: reasoning_effort=low for Groq's gpt-oss")
    ap.add_argument("--supervisor-max-wait", type=float, default=900.0,
                    help="seconds a call may wait out rate limits; a longer wait (a daily cap) stops the run")


def make_supervisor(a):
    if not a.supervisor_url:
        return MockSupervisor(None)
    return Supervisor(a.supervisor_url, a.supervisor_model, max_tokens=a.supervisor_max_tokens,
                      reasoning=a.supervisor_reasoning, api_key=api_key(a.supervisor_api_key_env),
                      json_mode=a.supervisor_json, temperature=None if a.supervisor_no_temperature else 0.0,
                      token_param=a.supervisor_token_param, params=parse_params(a.supervisor_param),
                      max_wait=a.supervisor_max_wait)
