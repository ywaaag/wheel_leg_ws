"""Exercise actual constructor/close with Kit I/O replaced, never launch Kit."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from native_stand.artifacts.io import read_json
from native_stand.simulation.recovery_isaac import backend


def test_owned_kit_cannot_hide_constructor_failure_before_worker_settlement(tmp_path, monkeypatch):
    registrations, configurations, closed = [], [], []
    app = SimpleNamespace(close=lambda **kw: closed.append(kw))
    module = ModuleType("isaaclab.app")
    module.AppLauncher = lambda cfg: configurations.append(cfg) or SimpleNamespace(app=app)
    monkeypatch.setitem(sys.modules, "isaaclab.app", module)
    monkeypatch.setattr(backend, "preflight", lambda *a, **kw: None)
    import atexit

    monkeypatch.setattr(atexit, "register", lambda fn: registrations.append(fn))

    def initialize(self, job, output):
        self.env = SimpleNamespace(close=lambda: closed.append("env"))
        raise RuntimeError("actual initial error")

    monkeypatch.setattr(backend.IsaacRecovery, "_initialize", initialize)
    job = SimpleNamespace(runtime=SimpleNamespace(device="cpu"))
    with pytest.raises(RuntimeError, match="actual initial error"):
        backend.IsaacRecovery(job, output=tmp_path)
    assert configurations[0]["fast_shutdown"] is False
    assert closed == ["env"]  # Public worker can now save terminal evidence.
    assert (
        read_json(tmp_path / "construction_failure.json")["error"]
        == "RuntimeError('actual initial error')"
    )
    assert len(registrations) == 1
    registrations[0]()
    assert closed[-1] == dict(exit_code=1)


def test_normal_close_is_idempotent_and_defers_only_owned_app(monkeypatch):
    import atexit

    callbacks, closed = [], []
    monkeypatch.setattr(atexit, "register", lambda fn: callbacks.append(fn))
    env = object.__new__(backend.IsaacRecovery)
    env._owns_app = True
    env.env = SimpleNamespace(close=lambda: closed.append("env"))
    env.app = SimpleNamespace(close=lambda **kw: closed.append(kw))
    env.close()
    env.close()
    assert closed == ["env"] and len(callbacks) == 1
    callbacks[0]()
    assert closed == ["env", dict(exit_code=0)]
