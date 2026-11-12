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


def test_mapped_asset_module_file_is_authoritative_over_unmapped_directory_hint(tmp_path):
    from native_stand.artifacts.io import file_sha256
    from native_stand.simulation.recovery_isaac.bindings import asset_module_root

    root = tmp_path / "copied/source/agent_world"
    package = root / "agent_world/assets"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package.parent / "__init__.py").write_text("")
    module = package / "rmuc_wheel_leg.py"
    module.write_text("VALUE = 42\n")
    template = dict(
        asset_module_root="/unavailable/original/directory",
        asset_module_path=str(module),
        asset_module_sha256=file_sha256(module),
    )
    assert asset_module_root(template) == root
    module.write_text("VALUE = 43\n")
    with pytest.raises(ValueError, match="identity"):
        asset_module_root(template)
