"""Supervisor flags shared by the live scripts (run_miniwob, run_h4)."""
import os
import sys

from surety.supervise.client import MockSupervisor, Supervisor


def add_supervisor_args(ap) -> None:
    ap.add_argument("--supervisor-url", help="OpenAI-compatible server; omitted = a supervisor that never answers (smoke)")
    ap.add_argument("--supervisor-model", default="default")
    ap.add_argument("--supervisor-api-key-env", help="name of the environment variable holding the API key")
    ap.add_argument("--supervisor-json", action="store_true", help="constrain replies to an option id or NONE")
    ap.add_argument("--supervisor-max-tokens", type=int, default=1024,
                    help="reasoning models (gpt-oss) spend tokens on hidden reasoning before the answer")
    ap.add_argument("--supervisor-reasoning", choices=["low", "medium", "high"])
    ap.add_argument("--supervisor-no-temperature", action="store_true", help="OpenAI reasoning models reject it")
    ap.add_argument("--supervisor-token-param", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")


def make_supervisor(a):
    if not a.supervisor_url:
        return MockSupervisor(None)
    key = os.environ.get(a.supervisor_api_key_env) if a.supervisor_api_key_env else None
    if a.supervisor_api_key_env and not key:
        sys.exit(f"{a.supervisor_api_key_env} is not set")
    return Supervisor(a.supervisor_url, a.supervisor_model, max_tokens=a.supervisor_max_tokens,
                      reasoning=a.supervisor_reasoning, api_key=key, json_mode=a.supervisor_json,
                      temperature=None if a.supervisor_no_temperature else 0.0, token_param=a.supervisor_token_param)
