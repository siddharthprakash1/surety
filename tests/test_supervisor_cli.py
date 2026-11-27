import argparse

import pytest

from surety.supervise import MockSupervisor, Supervisor
from surety.supervise.cli import add_supervisor_args, make_supervisor, parse_params


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


def test_key_from_dotenv_when_the_variable_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("DOT_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("﻿# keys\nOTHER=x\nexport DOT_KEY='k2'\n", encoding="utf-8")
    sup = make_supervisor(_args("--supervisor-url", "http://s", "--supervisor-api-key-env", "DOT_KEY"))
    assert sup.headers == {"Authorization": "Bearer k2"}


def test_request_fields_and_max_wait():
    sup = make_supervisor(_args("--supervisor-url", "http://s", "--supervisor-param", "reasoning_effort=low",
                                "--supervisor-param", "top_k=5", "--supervisor-max-wait", "60"))
    assert sup.params == {"reasoning_effort": "low", "top_k": 5} and sup.max_wait == 60
    assert parse_params(None) is None
    with pytest.raises(SystemExit):
        parse_params(["no-equals-sign"])
