import argparse

import pytest

from surety.supervise import MockSupervisor, Supervisor
from surety.supervise.cli import add_supervisor_args, make_supervisor


def _args(*argv):
    ap = argparse.ArgumentParser()
    add_supervisor_args(ap)
    return ap.parse_args(argv)


def test_defaults_leave_room_for_reasoning_and_no_url_is_a_mock():
    assert isinstance(make_supervisor(_args()), MockSupervisor)
    sup = make_supervisor(_args("--supervisor-url", "http://s", "--supervisor-model", "m"))
    assert isinstance(sup, Supervisor) and sup.max_tokens == 1024 and sup.temperature == 0.0


def test_reasoning_model_flags(monkeypatch):
    monkeypatch.setenv("SUP_KEY", "k")
    sup = make_supervisor(_args("--supervisor-url", "http://s", "--supervisor-no-temperature", "--supervisor-token-param",
                                "max_completion_tokens", "--supervisor-api-key-env", "SUP_KEY", "--supervisor-json"))
    assert (sup.temperature, sup.token_param, sup.headers, sup.json_mode) == (
        None, "max_completion_tokens", {"Authorization": "Bearer k"}, True)


def test_unset_key_variable_exits(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    with pytest.raises(SystemExit):
        make_supervisor(_args("--supervisor-url", "http://s", "--supervisor-api-key-env", "NOPE_KEY"))
