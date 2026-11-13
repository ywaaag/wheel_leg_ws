"""Portable tests for unified web ownership, contracts, and no-save lifecycle."""

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from native_stand.simulation import playback
from native_stand.simulation.playback_panel import PlaybackCommands, WebControls

FROZEN_TEST_SCOPE = "PORTABLE"


def plant_stub():
    return NS(
        plan=NS(
            job=NS(control=NS(height_command_m=0.34)),
            output_contract={"height_command_range_m": [0.2, 0.4]},
        ),
        terrain=NS(spawns={"flat": [0, 0], "steep_waves_H1": [0, 3.1]}),
        recording=False,
        telemetry=lambda: {"vx": 0.12, "height": None},
    )


def test_web_owns_commands_without_clamping_and_explicit_scene_reset():
    p = plant_stub()
    controls = WebControls(p)
    controls.plan.apply({"vx": 5.2, "yaw": -0.8, "height": 0.2})
    controls.key(265)
    controls.sync(p)
    assert (controls.vx, controls.yaw, controls.height) == (5.2, -0.8, 0.2)
    assert controls.plan.snapshot()["target"]["vx"] == 5.2
    controls.plan.apply({"scene": "steep_waves_H1"})
    controls.next_read = 0
    controls.sync(p)
    assert controls.reset_requested and p.spawn == "steep_waves_H1"
    assert (controls.vx, controls.yaw) == (0, 0)
    assert not controls.plan.consume()["reset"]
    controls.fault("closure")
    assert controls.paused and controls.plan.snapshot()["error"] == "closure"


@pytest.mark.parametrize(
    "payload",
    [
        {"height": 0.45},
        {"vx": float("nan")},
        {"skill": "Recovery"},
        {"scene": "fake"},
        {"paused": 1},
        {"vx": True},
    ],
)
def test_invalid_commands_are_atomic(payload):
    p = PlaybackCommands(plant_stub())
    before = p.snapshot()
    with pytest.raises(ValueError):
        p.apply(payload)
    assert p.snapshot() == before


def test_no_save_uses_memory_build_and_cleans_on_exit(monkeypatch):
    from native_stand.artifacts import io

    p = NS(recording=True, close=lambda: None)
    folders = []

    def create(prepared, *, directory):
        folders.append(directory)
        (directory / "control.so").write_bytes(b"build")
        return NS(path=Path("model.pt"), sha256="a" * 64), p, {}, directory

    monkeypatch.setattr(playback, "create_player", create)
    monkeypatch.setattr(playback, "run_loop", lambda *a, **k: {"physics_steps": 2})
    monkeypatch.setattr(io, "write_npz", lambda *a: pytest.fail("unexpected trace write"))
    monkeypatch.setattr(io, "write_json", lambda *a: pytest.fail("unexpected result write"))
    result = playback.run(NS(), headless=True, steps=2, no_save=True)
    assert not result["saved"] and not p.recording
    assert folders[0].parent == Path("/dev/shm") and not folders[0].exists()


def test_atlas_rotation_keeps_actual_grid_vertices_and_boundary_rejects():
    from native_stand.simulation.playback_terrain import Atlas

    h = np.arange(151 * 301, dtype=np.float32).reshape(151, 301)
    a = Atlas([h] * 4, 0.02)
    # First tile x in [-7.5,-4.5], y in [3,9].
    ix, iy = round((-0.0 - 7.5 - a.x0) / 0.02), round((3 - a.y0) / 0.02)
    np.testing.assert_array_equal(a.z[iy : iy + 301, ix : ix + 151], h.T[:, ::-1])
    with pytest.raises(ValueError, match="有限"):
        a.height(np.array([[20.0, 0.0]]))


def test_legacy_wheel_cap_is_rejected_before_engine(tmp_path):
    import json
    from test_motion_production_integration import make_job

    job, _ = make_job(tmp_path, retention=False)
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"config": job.as_run_record()}))
    (run / "model.pt").write_bytes(b"not-loaded")
    with pytest.raises(ValueError, match="4.8"):
        playback.plan(run, "model.pt")


def test_explicit_http_binding_uses_same_command_owner():
    import json
    import urllib.request
    from native_stand.simulation.playback_panel import start
    from native_stand.students.motion_panel import stop

    commands = PlaybackCommands(plant_stub())
    server, _ = start(commands, 0, host="127.0.0.1")
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            url + "/command",
            data=b'{"vx":0.5,"paused":false}',
            headers={"Content-Type": "application/json"},
        )
        result = json.load(opener.open(request, timeout=3))
        assert result["target"]["vx"] == 0.5
        assert commands.consume()["vx"] == 0.5
        snapshot = json.load(opener.open(url + "/telemetry", timeout=3))
        assert snapshot["paused"] is False and snapshot["recording"] is False
    finally:
        stop(server)
