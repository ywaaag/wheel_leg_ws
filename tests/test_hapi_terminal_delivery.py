"""Real default notify path with subprocess substituted; no hub messages in tests."""

import subprocess
from types import SimpleNamespace

import pytest

from native_stand.artifacts.io import write_json
from native_stand.integrations import hapi


def test_default_sender_delivers_one_terminal_json_on_stdin(tmp_path, monkeypatch):
    write_json(
        tmp_path / "state.json", dict(status="FAILED", updates_used=7, failure="a; $(x)\n坏")
    )
    monkeypatch.setenv("HAPI_CLI_EXECUTABLE", "/fake path/hapi")
    calls = []
    monkeypatch.setattr(
        hapi.subprocess,
        "run",
        lambda argv, **kw: calls.append((argv, kw)) or SimpleNamespace(returncode=0),
    )
    result = hapi.notify(tmp_path, enabled=True, target="bound-session")
    assert result == dict(status="sent", sent=True, target="bound-session")
    argv, kw = calls[0]
    assert argv == ["/fake path/hapi", "ping-peer", "bound-session", "--message-file", "-"]
    assert len(calls) == 1 and "shell" not in kw and kw["timeout"] == 90
    assert '"updates_used": 7' in kw["input"] and '"status": "FAILED"' in kw["input"]


@pytest.mark.parametrize("enabled,target", [(False, "session"), (True, None)])
def test_disabled_or_missing_target_never_invokes_cli(tmp_path, monkeypatch, enabled, target):
    monkeypatch.setattr(hapi.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected send"))
    assert not hapi.notify(tmp_path, enabled=enabled, target=target)["sent"]


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing"])
def test_transport_failure_preserves_training_terminal(tmp_path, monkeypatch, failure):
    p = write_json(tmp_path / "state.json", dict(status="COMPLETED", updates_used=10000))
    original = p.read_bytes()
    monkeypatch.setenv("HAPI_CLI_EXECUTABLE", "/fake/hapi")
    if failure == "timeout":

        def run(*a, **kw):
            raise subprocess.TimeoutExpired(a[0], 90)

        monkeypatch.setattr(hapi.subprocess, "run", run)
    elif failure == "missing":
        monkeypatch.delenv("HAPI_CLI_EXECUTABLE")
        monkeypatch.setattr(hapi.shutil, "which", lambda value: None)
    else:
        monkeypatch.setattr(hapi.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    result = hapi.notify(tmp_path, enabled=True, target="session")
    assert result["status"] == "failed" and not result["sent"]
    assert p.read_bytes() == original
