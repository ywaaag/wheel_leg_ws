"""Host-bound ABI4 sources and static MuJoCo CPU state. No mj_step is allowed."""

import copy
from dataclasses import replace
from pathlib import Path

import mujoco
import pytest
import torch

from native_stand.artifacts.checkpoint import load_actor_source, save_full, load_source
from native_stand.artifacts.io import json_digest, file_sha256
from native_stand.learning.policies.factory import build_policy
from native_stand.simulation.jump_backend import make_env
from native_stand.simulation.jump_response import Response
from test_jump_unified import job

FROZEN_TEST_SCOPE = "HOST"


@pytest.fixture(autouse=True)
def no_integrator(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("physics integration not authorized")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)


def test_actual_actor_mean_all_trainable_fresh_adams():
    from native_stand.learning.ppo.jump_unified import DirectActor
    from test_jump_unified import observation

    j = job()
    source = load_actor_source(j.initialization.checkpoint)
    assert source.sha256 == j.initialization.sha256
    handle = build_policy(j, observation(), source=source)
    expected = DirectActor()
    expected.load_state_dict(source.payload["actor"])
    x = observation()["student_actor"]
    assert torch.equal(handle["ppo"].actor.mean(x), expected.mean(x))
    assert all(p.requires_grad for p in handle["ppo"].actor.parameters())
    assert not handle["ppo"].optimizer.actor.state
    assert not handle["ppo"].optimizer.critic.state
    assert source.payload["PPO_updates"] == 0 and source.payload["supervised_updates"] == 1500
    captured = handle["capture"]({"updates": 0})
    assert captured["source_supervised_updates"] == 1500 and captured["counters"]["updates"] == 0


def test_full_static_critic_state_C_bytes_and_exact_restore(tmp_path):
    j = job()
    env = make_env(j, output=tmp_path / "first")
    try:
        before = env.observations()
        state = env.runtime_state()
        assert before["student_actor"].shape == (1, 37)
        assert before["student_critic"].shape == (1, 6370)
        assert len(env.envs[0].schema.fields) == 235
        assert json_digest(env.envs[0].schema.contract()) == json_digest(
            j.assets["jump_critic_schema"]
        )
        env.load_runtime_state(state)
        after = env.observations()
        for key in before.keys():
            assert torch.equal(before[key], after[key]), key
        assert (
            env.runtime_state()["environments"][0]["base"]["execution"]["C"]
            == state["environments"][0]["base"]["execution"]["C"]
        )
        assert env.envs[0].plant.physics_steps == 0
    finally:
        env.close()


def test_public_worker_update_save_publish_and_restore(tmp_path, monkeypatch):
    from test_phase3_fixes import _production_context
    from native_stand.experiments.worker import _execute_stage
    from native_stand.simulation import jump_backend
    from native_stand.learning.ppo.jump_unified import DirectActor

    j = job()
    j = replace(
        j,
        model=replace(j.model, actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        runtime=replace(j.runtime, steps_per_env=2),
        ppo=replace(j.ppo, epochs=1, minibatches=1),
        diagnostics=dict(
            j.diagnostics, minimum_unique_actor_samples=1, minimum_minibatch_samples=1
        ),
    )
    source = tmp_path / "actor.pt"
    torch.save(
        dict(
            actor=DirectActor((8,)).state_dict(),
            actor_only=True,
            output_contract=j.assets["jump_contract_data"],
            supervised_updates=1500,
        ),
        source,
    )
    j = replace(
        j, initialization=replace(j.initialization, checkpoint=source, sha256=file_sha256(source))
    )
    real_factory = jump_backend.make_env
    seen = []

    def scripted_physics(job):
        env = real_factory(job, output=tmp_path / "execution")
        for element in env.envs:
            p = element.plant

            def advance(p=p):
                # Physical I/O boundary only. Real C/FSM, V4, packet/reward/course remain enabled.
                ex = p.execution
                q = torch.tensor(p.data.qpos[p.qids][None], dtype=torch.float32)
                dq = torch.tensor(p.data.qvel[p.vids][None], dtype=torch.float32)
                motor, net = ex.step(q, dq)
                p.actual_motor = motor[0].numpy().copy()
                p.actual_NET = net[0].numpy().copy()
                p.physics_steps += 1
                p.scoring_force_valid = True
                p.solved_ground_contact[:] = True

            p.integrate_control = advance
        seen.append(env)
        return env

    monkeypatch.setattr(jump_backend, "make_env", scripted_physics)
    context = _production_context(tmp_path, j, smoke=100)
    try:
        result = _execute_stage(context, "smoke", None)
        assert result.updates == 100
        last = context.latest_reference()
        assert last and Path(last["checkpoint"]).is_file()
        saved = load_source(last["checkpoint"])
        assert saved.payload["counters"]["updates"] == 100
        assert saved.payload["jump_consumed_episode"] == seen[0].serial
        assert (
            context.refs.candidate_index(context.run_dir.name)["last_review"]["through_update"]
            == 100
        )
        assert saved.payload["version"] == "jump_unified_raw37_wheel48_v1"
        assert context.refs.consumer_paths()
        from native_stand.simulation.jump_adapter import JumpAdapter
        from native_stand.simulation.driver import Backend

        resumed = JumpAdapter(Backend("mujoco", scripted_physics, {}))
        resumed.build(context, "smoke", saved)
        try:
            assert resumed.cursor == saved.payload["jump_consumed_episode"]
            assert resumed.pending is None
            assert (
                context.refs.candidate_index(context.run_dir.name)["last_review"]["through_update"]
                == 100
            )
            for key, value in saved.payload["actor"].items():
                assert torch.equal(value, resumed.ppo.actor.state_dict()[key])
        finally:
            resumed.close()
    finally:
        context.metrics.close()


def test_partial_reset_and_fresh_reset_preserve_completed_not_dwell(tmp_path):
    j = job()
    j = replace(j, runtime=replace(j.runtime, envs=2))
    env = make_env(j, output=tmp_path / "both")
    try:
        second = copy.deepcopy(env.envs[1].runtime_state())
        env.reset([0])
        assert env.envs[1].course.state_dict() == second["course"]
        assert env.envs[1].plant.execution.c.state_dict() == second["base"]["execution"]["C"]
        state = env.runtime_state()
        state["completed"] = [dict(id=1, ended=True)]
        state["serial"] = 1
        fresh = make_env(
            replace(j, assets=dict(j.assets, jump_restore_mode="fresh_reset")),
            output=tmp_path / "fresh",
        )
        try:
            fresh.load_runtime_state(state)
            assert fresh.completed == state["completed"]
            assert fresh.envs[0].reward.elapsed_s == 0
            assert (
                fresh.envs[0].request["request_id"]
                != state["environments"][0]["request"]["request_id"]
            )
            assert len(fresh.envs[0].course.pending) == 1
        finally:
            fresh.close()
    finally:
        env.close()


def test_jump_playback_dry_run_contract_has_no_engine_or_integration(tmp_path, monkeypatch):
    import sys
    from native_stand.experiments.owner import create_run
    from native_stand.simulation.playback import plan, Controls, run_loop

    j = job()
    run = create_run(j, runs_root=tmp_path / "runs", run_id="play")
    prepared = plan(run, j.initialization.checkpoint)
    assert prepared.record()["driver_caps_nm"][2] == 4.8
    assert prepared.record()["C_core"] == "jump_ABI4"
    assert prepared.output_contract == j.assets["jump_contract_data"]
    assert prepared.record()["physics_steps"] == 0
    controls = Controls()
    controls.key(ord("j"))
    assert controls.jump_requested
    controls.key(32)
    assert controls.paused

    class Clock:
        physics_dt = 0.001

        def __init__(self):
            self.execution = type("Clock", (), {"tick": 0})()
            self.resets = 0

        def reset(self):
            self.resets += 1

        def step(self, controls):
            self.execution.tick += 1

    clock = Clock()
    with pytest.raises(ValueError, match="headless暂停"):
        run_loop(clock, controls, steps=1)
    assert clock.execution.tick == 0
    controls.key(32)
    controls.key(ord("r"))
    run_loop(clock, controls, steps=2)
    assert clock.execution.tick == 2 and clock.resets == 1


def test_V4_explicit_PD_default_matches_frozen_math_and_partial_reset(tmp_path):
    from native_stand.simulation.recovery_isaac.bindings import load_response
    from native_stand.simulation.recovery_isaac.execution import Execution
    from native_stand.artifacts.io import read_json

    identity = dict(
        path="/home/kellen/wheel_leg_native_train/isolated/2026_11_05__motion_supported2048_dqgain_ppo/response_tensor.py",
        sha256="7cdd53f8354fc1b2491b8e3b92e2b28d9e5eb02913c59ce684afe43ff6f227d3",
    )
    v4 = read_json(job().assets["jump_bindings"]["V4"]["path"])
    old = Execution(v4, load_response(identity), 2)
    new = Execution(v4, Response, 2)
    q = torch.zeros(2, 6)
    dq = q.clone()
    old.reset(old.ids, q, dq)
    new.reset(new.ids, q, dq)
    for tick in range(18):
        q = q + 0.0001
        dq = dq + 0.001
        if tick == 7:
            old.reset(torch.tensor([0]), q[:1], dq[:1])
            new.reset(torch.tensor([0]), q[:1], dq[:1])
        for a, b in zip(old.step(q, dq), new.step(q, dq), strict=True):
            assert torch.equal(a, b)
    assert old.tick == new.tick == 18


def test_substep_fault_records_actual_interval_and_terminal_before_reset(tmp_path):
    from test_jump_unified import feedback

    j = job()
    batch = make_env(j, output=tmp_path / "fault")
    env = batch.envs[0]
    p = env.plant
    try:

        def sensor():
            return feedback(p.execution.tick)

        p.feedback = sensor
        batch.reset([0])

        def physical_io():
            if p.physics_steps == 2:
                raise RuntimeError("scripted substep failure")
            q = torch.tensor(p.data.qpos[p.qids][None], dtype=torch.float32)
            dq = torch.tensor(p.data.qvel[p.vids][None], dtype=torch.float32)
            p.execution.step(q, dq)
            p.physics_steps += 1
            p.scoring_force_valid = True
            p.solved_ground_contact[:] = True

        p.integrate_control = physical_io
        outcome = batch.step(torch.zeros(1, 6))
        assert outcome["extras"]["actual_physics_steps"] == [2], [
            r["failure"] for r in batch.completed
        ]
        assert not outcome["extras"]["transition_valid"][0]
        assert outcome["terminated"][0] and not outcome["truncated"][0]
        assert outcome["rewards"].shape == (1,)
        assert batch.completed[0]["fault"] and batch.completed[0][
            "jump_control_s"
        ] == pytest.approx(0.002)
        assert not torch.equal(
            outcome["terminal_observations"]["student_critic"],
            outcome["observations"]["student_critic"],
        )
        assert batch.envs[0].reward.elapsed_s == 0
    finally:
        batch.close()


def test_BC_history_is_not_PPO_resume_and_teacher_labels_use_physical_units():
    from native_stand.learning.distillation.jump_labels import (
        inspect_BC_source,
        convert_teacher_mean,
    )
    from native_stand.artifacts.io import read_json

    refs = job().assets["jump_bindings"]
    bc = inspect_BC_source(refs["BC_state"]["path"], refs["BC_state"]["sha256"])
    assert (
        bc["kind"] == "BC_resume_only"
        and bc["supervised_updates"] == 1500
        and bc["PPO_updates"] == 0
    )
    teacher = read_json(refs["executor_contract"]["path"])
    labels = convert_teacher_mean(
        torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, -0.5]]), teacher, height=torch.tensor([0.34])
    )
    assert labels["raw"][0, 2] == pytest.approx(4.0)
    assert labels["raw"][0, 5] == pytest.approx(-4.0)
    assert labels["inverse_activation"] == "identity" and labels["labels_clipped"] is False
