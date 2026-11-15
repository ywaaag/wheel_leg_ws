"""Portable consumer tests: no engine, firmware build, or historical run access."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from native_stand.artifacts.io import file_sha256, read_json, write_json
from native_stand.simulation import recovery_motion_handoff as h
from native_stand.simulation.sim2sim import from_file

FROZEN_TEST_SCOPE = "PORTABLE"


@pytest.fixture
def protocol(tmp_path):
    source = tmp_path / "source.json"
    write_json(source, {"config": {}})
    ref = dict(path=str(source), sha256=file_sha256(source))
    return dict(
        version="recovery_motion_handoff_v1",
        recovery=ref,
        motion=ref,
        source_run=ref,
        gate_source=ref,
        gate_header=ref,
        recovery_spec={},
        motion_spec={"output_contract": {"action_version": "legacy"}},
        seconds=10.0,
        limits=dict(integration_steps=20000, wall_s=60.0, storage_bytes=50_000_000),
        variants=[dict(name="baseline", kind="recovery_only")],
        cases=[dict(case_id="nominal", pose="Swing")],
        gate={},
    )


def test_cli_dispatch_dry_run_is_readonly(protocol, tmp_path):
    config = tmp_path / "protocol.json"
    write_json(config, protocol)
    output = tmp_path / "absent"
    p = from_file(config, output, dry_run=True)
    assert p["maximum_integration_steps"] == 20000
    assert not p["starts_engine"] and not p["writes"] and not output.exists()
    assert p["same_world_switch"] and not p["capability_accepted"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("integration_steps", 19999),
        ("storage_bytes", 1),
        ("wall_s", float("nan")),
        ("wall_s", 0),
        ("integration_steps", True),
    ],
)
def test_preflight_rejects_unfunded_or_invalid_bounds(protocol, field, value):
    protocol["limits"][field] = value
    with pytest.raises(ValueError):
        h.plan(protocol)


def test_preflight_rejects_aliases_and_changed_dependencies(protocol):
    protocol["variants"] *= 2
    with pytest.raises(ValueError, match="Unique"):
        h.plan(protocol)
    protocol["variants"] = protocol["variants"][:1]
    protocol["motion"] = {**protocol["motion"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA"):
        h.plan(protocol)


def test_step_boundary_checks_before_indivisible_call(monkeypatch):
    monkeypatch.setattr(h.time, "monotonic", lambda: 10.0)
    limits = dict(integration_steps=40, wall_s=2)
    h.check_step_budget(limits, 20, 9.0)
    with pytest.raises(RuntimeError, match="before step"):
        h.check_step_budget(limits, 21, 9.0)
    with pytest.raises(TimeoutError):
        h.check_step_budget(limits, 0, 8.0)


@dataclasses.dataclass
class Runtime:
    envs: int = 1
    device: str = "cpu"


@dataclasses.dataclass
class Job:
    assets: dict
    runtime: Runtime = dataclasses.field(default_factory=Runtime)


def install_consumer(monkeypatch, *, failure=False):
    envs = []

    class Env:
        def __init__(self, job):
            self.execution = SimpleNamespace(adopted=torch.zeros(1, 6))
            self.calls = 0
            self.closed = False
            envs.append(self)

        def observations(self):
            return {"student_actor": torch.zeros(1, 31)}

        def step(self, action):
            self.calls += 1
            if failure:
                exc = RuntimeError("partial engine call")
                exc.actual_physics_steps = [6]
                raise exc
            self.execution.adopted.fill_(0.25)
            state = torch.zeros(1, 10, 13)
            state[:, :, 3] = 0.34
            state[:, :, 9:11] = 10
            trace = {"state_1kHz": state}
            for key, width in [
                ("q", 6),
                ("dq", 6),
                ("motor", 6),
                ("NET", 6),
                ("adopted", 6),
                ("dq_HP10", 4),
            ]:
                trace[key + "_1kHz"] = torch.zeros(1, 10, width)
            return dict(
                terminated=torch.tensor([False]),
                truncated=torch.tensor([self.calls == 1000]),
                extras=dict(
                    actual_physics_steps=torch.tensor([20]),
                    collection_trace=trace,
                    collection_conditions={
                        k: torch.tensor([5])
                        for k in (
                            "actor_age_ms",
                            "target_delay_ms",
                            "wheel_left_ms",
                            "wheel_right_ms",
                        )
                    },
                ),
            )

        def close(self):
            self.closed = True

    def observe(*args):
        return 0

    def commit(*args):
        return 0

    monkeypatch.setattr(
        h.ctypes,
        "CDLL",
        lambda _: SimpleNamespace(RLHandoffObserve=observe, RLHandoffCanCommit=commit),
    )
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(h, "RecoveryMuJoCoCollection", Env)
    monkeypatch.setattr(h, "job_from_record", lambda _: Job({"recovery_mujoco": {}}))
    monkeypatch.setattr(h.InferenceSpec, "from_record", lambda _: None)
    actor = SimpleNamespace(mean=lambda x: torch.zeros(1, 6), decode_mean=lambda x: x)
    monkeypatch.setattr(
        h, "load_model", lambda *a, **kw: SimpleNamespace(mean=actor.mean, actor=actor)
    )
    return envs


def test_real_dispatch_baseline_full_hold_and_no_reset(protocol, tmp_path, monkeypatch):
    envs = install_consumer(monkeypatch)
    config = tmp_path / "protocol.json"
    write_json(config, protocol)
    result = from_file(config, tmp_path / "new")
    row = result["episodes"][0]
    assert result["actual_integration_steps"] == 20000
    assert row["recovery_only_hold10"] and not row["post_switch_hold10"]
    assert row["switch_s"] is None and row["truncated"]
    assert envs[0].calls == 1000 and envs[0].closed
    with np.load(row["trace"]["path"]) as trace:
        assert not trace["owner"].any()
    with pytest.raises(FileExistsError):
        h.run(protocol, tmp_path / "new")


def test_partial_engine_failure_is_charged_and_closed(protocol, tmp_path, monkeypatch):
    envs = install_consumer(monkeypatch, failure=True)
    with pytest.raises(RuntimeError, match="partial engine"):
        h.run(protocol, tmp_path / "failed")
    terminal = read_json(tmp_path / "failed" / "terminal.json")
    assert terminal["status"] == "FAILED" and terminal["actual_integration_steps"] == 6
    assert envs[0].closed


def test_gate_switch_keeps_world_and_never_awards_partial_hold(protocol, tmp_path, monkeypatch):
    envs = install_consumer(monkeypatch)

    def observe(state, *args):
        state._obj.requested = 1
        return 1

    def commit(state, config, tick, previous, target):
        assert np.array_equal(previous, np.full(6, 0.25, dtype=np.float32))
        return 1

    monkeypatch.setattr(
        h.ctypes,
        "CDLL",
        lambda _: SimpleNamespace(RLHandoffObserve=observe, RLHandoffCanCommit=commit),
    )
    protocol["variants"] = [dict(name="gate", kind="gated", dwell_ms=100)]
    result = h.run(protocol, tmp_path / "switched")
    row = result["episodes"][0]
    assert row["switch_s"] == 0.01 and not row["post_switch_hold10"]
    assert row["post_switch_raw_hold_s"] == pytest.approx(9.99)
    assert len(envs) == 1 and envs[0].calls == 1000 and envs[0].closed
    with np.load(row["trace"]["path"]) as trace:
        assert not trace["owner"][0] and trace["owner"][1:].all()


@pytest.mark.parametrize(
    "dry,result,expected",
    [
        (True, {}, "只读计划"),
        (False, {"status": "COMPLETED", "actual_integration_steps": 20}, "实际积分20子步"),
        (False, {"extra_simulation_steps": 0}, "无新增积分"),
        (False, {"extra_simulation_steps": 20}, "跨引擎顺序采集完成"),
    ],
)
def test_cli_output_matches_real_operation(monkeypatch, capsys, dry, result, expected):
    from native_stand import cli
    from native_stand.simulation import sim2sim

    monkeypatch.setattr(sim2sim, "from_file", lambda *a, **kw: result)
    argv = ["sim2sim", "--config", "unused.json", "--output", "unused"]
    if dry:
        argv.append("--dry-run")
    assert cli.main(argv) == 0
    assert expected in capsys.readouterr().out


def test_native_profile_is_sha_bound_and_cannot_stack_spring(protocol):
    ref = protocol["source_run"]
    protocol["native_sources"] = {"source": ref}
    v = protocol["variants"][0]
    v["native_firmware"] = dict(gains="native", actor_age_ms=0)
    assert h.plan(protocol)["training_updates"] == 0
    v["spring_compensation"] = True
    protocol["spring_sources"] = {"source": ref}
    with pytest.raises(ValueError, match="no extra spring"):
        h.plan(protocol)
    v.pop("spring_compensation")
    protocol["native_sources"]["source"] = {**ref, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="Native firmware source SHA"):
        h.plan(protocol)
