"""Actual leg-led math/Actor/PPO consumers, no physics/GPU fixture."""

import copy

import pytest
import torch

from native_stand.tasks.recovery.leg_driven import (
    COST_WEIGHTS,
    CRITIC_WIDTH,
    Course,
    Lifecycle,
    review_window,
)

FROZEN_TEST_SCOPE = "PORTABLE"
GEO = dict(
    short_length_m=[0.145, 0.145], short_swing_rad=[-0.65, -0.65], support_swing_rad=[0.0, 0.0]
)


def life(n=2, phase=0):
    lifecycle = Lifecycle(n, "cpu", GEO, 0.9980019986673331)
    ids = torch.arange(n)
    g = torch.tensor([[0.376, 0.376, -1.276, -1.276]]).expand(n, -1).clone()
    if phase == 1:
        g[:, :2] = 0.145
        g[:, 2:] = -0.65
    if phase >= 2:
        g[:, :2] = 0.145
        g[:, 2:] = 0
    lifecycle.reset(ids, torch.full((n,), 0.168), torch.zeros(n))
    lifecycle.configure_reset(ids, g, torch.full((n,), phase), torch.zeros(n, dtype=torch.long))
    return lifecycle, g


def advance(lifecycle, g, *, wheel=0.0, tilt=0.0, t=0.01, height=0.168, terminal=False):
    n = lifecycle.n
    w = torch.full((n, 2), wheel)
    grav = torch.tensor(
        [[0.0, -torch.cos(torch.tensor(tilt)), torch.sin(torch.tensor(tilt))]]
    ).expand(n, -1)
    for _ in range(10):
        lifecycle.substep(
            g, w, grav, torch.zeros(n, 3), torch.zeros(n), torch.zeros(n, 6), torch.ones(n, 6) * 40
        )
    s = torch.zeros(n, 13)
    s[:, 3] = height
    s[:, 9:11] = 3.0
    s[:, 12] = torch.cos(torch.tensor(tilt))
    return lifecycle.observe(
        time_s=torch.full((n,), t),
        state=s,
        target=torch.zeros(n, 6),
        motor=torch.zeros(n, 6),
        student=torch.ones(n, dtype=torch.bool),
        terminal=torch.full((n,), terminal),
        truncated=torch.zeros(n, dtype=torch.bool),
        hp10=torch.zeros(n, 4),
        angular_velocity=torch.zeros(n, 3),
        geometry=g,
    )


def test_phase_advance_requires_both_short_and_dwell():
    lifecycle, g = life()
    g[0, :2] = 0.145
    g[0, 2:] = -0.65
    for i in range(10):
        advance(lifecycle, g, t=(i + 1) * 0.01)
    assert lifecycle.phase.tolist() == [1, 0]
    assert not lifecycle.success.any()


def test_required_geometric_swing_during_contraction_is_not_sequence_failure():
    lifecycle, g = life()
    g[:, :2] = 0.145
    g[:, 2:] = -0.65
    advance(lifecycle, g)
    assert lifecycle.sequence_valid.all()


def test_early_rotation_is_not_shortening_success():
    lifecycle, g = life()
    g[:, 2:] = 0
    for i in range(12):
        advance(lifecycle, g, t=(i + 1) * 0.01)
    assert lifecycle.phase.tolist() == [0, 0]
    assert not lifecycle.sequence_valid.any()


def test_signed_wheel_velocity_cannot_cancel_or_pass_ready():
    lifecycle, g = life(phase=3)
    for i in range(60):
        out = advance(lifecycle, g, wheel=-7.0, height=0.34, t=(i + 1) * 0.01)
    assert not lifecycle.success.any()
    assert not out["leg_driven_quality"].any()


def test_one_ms_peak_is_preserved_even_when_last_sample_zero():
    lifecycle, g = life()
    grav = torch.tensor([[0.0, -1.0, 0.0]]).expand(2, -1)
    for i in range(10):
        lifecycle.substep(
            g,
            torch.full((2, 2), 5.0 if i == 2 else 0.0),
            grav,
            torch.zeros(2, 3),
            torch.zeros(2),
            torch.zeros(2, 6),
            torch.ones(2, 6),
        )
    assert lifecycle.wheel_peak.tolist() == [[5.0, 5.0], [5.0, 5.0]]
    assert lifecycle.current_wheel.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_high_wheel_speed_fault_cannot_receive_success():
    lifecycle, g = life(phase=3)
    advance(lifecycle, g, wheel=10.0, height=0.34)
    out = advance(lifecycle, g, wheel=10.0, height=0.34, t=0.02)
    assert out["terminated"].all() and not out["success"].any()
    assert (out["reward"] < 0).all()


def test_settled_auxiliary_reset_never_claims_full_swing():
    lifecycle, g = life(phase=3)
    for i in range(51):
        advance(lifecycle, g, height=0.34, t=(i + 1) * 0.01)
    assert lifecycle.success.all()
    assert lifecycle.episode_record(0)["initial_phase"] == 3


def test_partial_reset_preserves_other_rows_and_full_snapshot():
    lifecycle, g = life()
    advance(lifecycle, g, wheel=1.0)
    state = lifecycle.state_dict()
    other = copy.deepcopy(state)
    lifecycle.reset(torch.tensor([0]), torch.tensor([0.168]), torch.tensor([0.0]))
    assert torch.equal(lifecycle.wheel_square[1], state["wheel_square"][1])
    restored, _ = life()
    restored.load_state_dict(other)
    for k, v in other.items():
        if torch.is_tensor(v):
            assert torch.equal(getattr(restored, k), v), k
    assert restored.critic(torch.zeros(2, 904)).shape == (2, CRITIC_WIDTH)


def test_missing_real_substeps_rejected():
    lifecycle, g = life()
    with pytest.raises(ValueError, match="ten actual"):
        lifecycle.observe(
            time_s=torch.ones(2),
            state=torch.zeros(2, 13),
            target=torch.zeros(2, 6),
            motor=torch.zeros(2, 6),
            student=torch.ones(2, dtype=torch.bool),
            terminal=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
            hp10=torch.zeros(2, 4),
            angular_velocity=torch.zeros(2, 3),
            geometry=g,
        )


def rows(level=0, passed=True):
    return [
        dict(
            objective="Swing" if i == 0 else "standing",
            ended=300,
            student_success=300 if passed else 0,
            quality=300 if passed else 0,
            sequence_valid=300,
            condition=dict(group=g, course_level=level, DR_strength=0.0),
        )
        for i, g in enumerate(("full_Swing", "shortened", "rotated", "lifted"))
    ]


def test_promotion_uses_three_full_covered_windows_and_all_conditions():
    hist = []
    for i in range(3):
        record, up, stop = review_window(rows(), 0, hist)
        assert up == (i == 2)
        assert not stop
        hist.append(record)
    bad = rows()
    bad[0]["ended"] = 0
    record, up, _ = review_window(bad, 0, hist)
    assert not record["coverage_sufficient"] and not up


def test_dr_missing_stress_cohort_is_unknown():
    record, up, _ = review_window(rows(level=4), 4, [])
    assert not record["coverage_sufficient"] and not up


def test_curriculum_keeps_full_swing_and_has_no_teacher_prefix():
    c = Course(1)
    assert c.choose()["group"] == "full_Swing"
    for stage in range(6):
        c.stage = stage
        for _ in range(100):
            s = c.choose()
            assert s["prefix_s"] == 0.0
            assert s["course_level"] == stage
            c.record(s["group"], student_seconds=1.0, teacher_seconds=0.0, passed=False)
    assert c.counts["full_Swing"]["episodes"] > 0
    assert sum(COST_WEIGHTS) == 1.0


def test_same_physical_actor_factory_accepts_new_critic_and_updates_math():
    from native_stand.learning.inference import InferenceSpec, build_actor

    spec = InferenceSpec(
        "recovery_physical",
        31,
        (8,),
        6,
        {"version": "recovery_cubic_physical_v2"},
        dict(
            action_version="recovery_cubic_physical_Gaussian_v2",
            center_rad=[-0.13, -0.15, 0.13, -0.15],
            leg_lower_rad=[-1.5] * 4,
            leg_upper_rad=[1.5] * 4,
        ),
    )
    a = build_actor(spec)
    x = torch.zeros(2, 31)
    x[:, 25:] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    y = a({"student_actor": x}, stochastic_output=False)
    assert y.shape == (2, 6) and torch.isfinite(y).all()
    loss = y.square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.parameters())


def test_course_window_plateau_does_not_restart_budget():
    history = []
    for _ in range(7):
        record, advance_level, plateau = review_window(rows(passed=False), 0, history)
        history.append(record)
    assert plateau and not advance_level


def test_whole_episode_tilt_cost_applies_before_lift():
    a, g = life()
    b, _ = life()
    ra = advance(a, g, tilt=0.0)["density"]["whole_tilt"]
    rb = advance(b, g, tilt=0.15)["density"]["whole_tilt"]
    assert (rb < ra).all()


def test_reset_zero_wheel_head_keeps_leg_rows_and_backbone(tmp_path):
    from native_stand.learning.ppo.recovery_cubic import PhysicalActor

    actor = PhysicalActor(
        dict(
            center_rad=[-0.13, -0.15, 0.13, -0.15],
            leg_lower_rad=[-1.5] * 4,
            leg_upper_rad=[1.5] * 4,
        ),
        hidden_dims=(8,),
    )
    original = copy.deepcopy(actor.state_dict())
    with torch.no_grad():
        actor.mlp[-1].weight[[2, 5]] = 0.0
        actor.mlp[-1].bias[[2, 5]] = 0.0
    inp = torch.randn(16, 31)
    assert torch.equal(actor.physical_mean(inp)[:, [2, 5]], torch.zeros(16, 2))
    assert torch.equal(actor.state_dict()["mlp.0.weight"], original["mlp.0.weight"])
    assert torch.equal(actor.mlp[-1].weight[[0, 1, 3, 4]], original["mlp.2.weight"][[0, 1, 3, 4]])


def test_settle_checks_vertical_speed_not_yaw_slot():
    lifecycle, g = life(phase=3)
    # State column2 is yaw rate in the public 13D state; vertical speed comes
    # from the explicit 1ms root-velocity source and cannot be substituted.
    n = 2
    for tick in range(60):
        for _ in range(10):
            lifecycle.substep(
                g,
                torch.zeros(n, 2),
                torch.tensor([[0.0, -1.0, 0.0]]).expand(n, -1),
                torch.zeros(n, 3),
                torch.full((n,), 0.2),
                torch.zeros(n, 6),
                torch.ones(n, 6),
            )
        state = torch.zeros(n, 13)
        state[:, 3] = 0.34
        state[:, 9:11] = 3
        state[:, 12] = 1
        out = lifecycle.observe(
            time_s=torch.full((n,), (tick + 1) * 0.01),
            state=state,
            target=torch.zeros(n, 6),
            motor=torch.zeros(n, 6),
            student=torch.ones(n, dtype=torch.bool),
            terminal=torch.zeros(n, dtype=torch.bool),
            truncated=torch.zeros(n, dtype=torch.bool),
            hp10=torch.zeros(n, 4),
            angular_velocity=torch.zeros(n, 3),
            geometry=g,
        )
    assert not out["success"].any()
    assert torch.equal(lifecycle.hold, torch.zeros(n))


def test_promoted_course_stops_after_two_bad_full_swing_windows():
    history = []
    record, _, stop = review_window(rows(level=1, passed=False), 1, history)
    assert not stop
    history.append(record)
    _, _, stop = review_window(rows(level=1, passed=False), 1, history)
    assert stop


def test_progress_and_wheel_improvement_prevent_plateau_stop():
    history = []
    for i in range(8):
        window = rows(passed=False)
        for row in window:
            row["phase_progress_sum"] = i * 20
            row["wheel_rms_sum"] = max(0, 300 - i * 10)
        record, _, stop = review_window(window, 0, history)
        assert not stop
        history.append(record)


def test_reset_cannot_zero_borrowed_execution_caps():
    lifecycle, g = life()
    execution_caps = torch.tensor([[40.0, 40.0, 4.8, 40.0, 40.0, 4.8]]).expand(2, -1).clone()
    lifecycle.effort_caps = execution_caps
    before = execution_caps.clone()
    lifecycle.reset(torch.arange(2), torch.ones(2) * 0.168, torch.zeros(2))
    assert torch.equal(execution_caps, before)
    assert "effort_caps" not in lifecycle.state_dict()
    history = torch.zeros(2, 144) / execution_caps.repeat(1, 24)
    assert torch.isfinite(history).all()


def adapter_fixture(tmp_path, *, source=None):
    import dataclasses
    from pathlib import Path
    from types import SimpleNamespace as NS

    from tensordict import TensorDict

    from native_stand.artifacts.io import file_sha256, write_json
    from native_stand.artifacts.model_refs import ModelIndex
    from native_stand.experiments.config import load_job_config
    from native_stand.simulation.driver import Backend

    job = load_job_config(
        Path(__file__).resolve().parents[1]
        / "configs/jobs/recovery_leg_driven2100_10000_attempt3.json"
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    contract = tmp_path / "contract.json"
    runtime = tmp_path / "runtime.json"
    write_json(
        contract,
        dict(
            action_version=job.action.version,
            center_rad=[0.0] * 4,
            leg_lower_rad=[-1.5] * 4,
            leg_upper_rad=[1.5] * 4,
        ),
    )
    write_json(runtime, dict(bindings={}))
    job = dataclasses.replace(
        job,
        model=dataclasses.replace(job.model, actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        runtime=dataclasses.replace(job.runtime, envs=2, steps_per_env=2, device="cpu"),
        ppo=dataclasses.replace(job.ppo, epochs=1, minibatches=1),
        initialization=dataclasses.replace(
            job.initialization,
            mode="full" if source else "scratch",
            checkpoint=str(source) if source else None,
            sha256=file_sha256(source) if source else None,
        ),
        assets={
            "recovery_training_contract": dict(path=str(contract), sha256=file_sha256(contract)),
            "recovery_isaac_runtime": dict(path=str(runtime), sha256=file_sha256(runtime)),
        },
    )
    obs = TensorDict(
        dict(
            student_actor=torch.randn(2, 31),
            student_critic=torch.randn(2, 942),
            recovery_actor_valid=torch.ones(2, 1, dtype=torch.bool),
        ),
        [2],
    )
    io = NS(
        env=NS(completed=[], completed_count=0),
        observations=lambda: obs,
        runtime_state=lambda: {},
        close=lambda: None,
    )
    io.step = lambda actions: dict(
        rewards=torch.tensor([[0.1], [0.2]]),
        dones=torch.ones(2, dtype=torch.bool),
        extras=dict(
            transition=dict(
                student_mask=torch.ones(2, dtype=torch.bool),
                motor=torch.zeros(2, 6),
                actual_actuator=torch.zeros(2, 6),
            )
        ),
    )
    context = NS(
        job=job,
        run_dir=tmp_path,
        output=tmp_path,
        updates_used=0,
        refs=ModelIndex(tmp_path / "refs.json"),
        store=NS(update=lambda **kw: None),
        total_updates_before=lambda stage: 0,
        curriculum_state=lambda: {},
    )
    return context, Backend("CPU-I/O", lambda job: io)


def test_worker_dispatches_leg_recipe_to_real_adapter(tmp_path, monkeypatch):
    from native_stand.experiments import worker
    from native_stand.simulation import driver
    from native_stand.simulation.recovery_adapter import RecoveryAdapter

    context, backend = adapter_fixture(tmp_path)
    monkeypatch.setattr(worker, "resolve_backend", lambda job: backend)

    def consume(context, adapter, *, stage, resume_from):
        assert type(adapter) is RecoveryAdapter
        adapter.build(context, stage, resume_from)
        assert adapter.leg_driven and adapter.readback is not None
        metrics = adapter.update_once(context, 0)
        context.updates_used = 1
        adapter.after_update(context, 0, metrics)
        state = adapter.capture(context, dict(updates=1))
        assert state["warmup_updates_done"] == 1
        assert "recovery_episode_cursor" in state
        assert "leg_driven_review_history" in state
        assert (tmp_path / "optimization_audit_00001.json").exists()
        return state

    monkeypatch.setattr(driver, "run_stage", consume)
    state = worker._execute_stage(context, "formal", None)
    assert state["counters"]["updates"] == 1


def test_full_source_preserves_warmup_and_first_actor_update(tmp_path):
    from native_stand.artifacts.checkpoint import save_full
    from native_stand.simulation.recovery_adapter import RecoveryAdapter

    context, backend = adapter_fixture(tmp_path / "first")
    first = RecoveryAdapter(backend)
    first.build(context, "formal", None)
    for i in range(6):
        metrics = first.update_once(context, i)
        context.updates_used += 1
        first.after_update(context, i, metrics)
    source = tmp_path / "source.pt"
    state = first.capture(context, dict(updates=6))
    assert state["warmup_updates_done"] == 5
    save_full(source, state)
    restored_context, backend = adapter_fixture(tmp_path / "next", source=source)
    restored = RecoveryAdapter(backend)
    restored.build(restored_context, "formal", None)
    assert restored._warmup == 0 and restored._prior_warmup == 5
    assert restored.ppo.optimizer.state_dict()["actor"]["state"]
    metrics = restored.update_once(restored_context, 0)
    assert metrics["actor_optimizer_steps"] > 0 and not metrics["critic_only"]
    restored_context.updates_used = 1
    restored.after_update(restored_context, 0, metrics)
    assert restored.readback.actor_updates == 1
    for i in range(1, 12):
        metrics = restored.update_once(restored_context, i)
        restored_context.updates_used += 1
        restored.after_update(restored_context, i, metrics)
    import json

    assert json.loads((restored_context.output / "engineering_smoke.json").read_text())[
        "engineering_smoke_passed"
    ]


def test_full_source_retains_remaining_critic_only_updates(tmp_path):
    from native_stand.artifacts.checkpoint import save_full
    from native_stand.simulation.recovery_adapter import RecoveryAdapter

    context, backend = adapter_fixture(tmp_path / "first")
    first = RecoveryAdapter(backend)
    first.build(context, "formal", None)
    for i in range(3):
        metrics = first.update_once(context, i)
        context.updates_used += 1
        first.after_update(context, i, metrics)
    source = tmp_path / "partial_warmup.pt"
    save_full(source, first.capture(context, dict(updates=3)))
    context, backend = adapter_fixture(tmp_path / "next", source=source)
    restored = RecoveryAdapter(backend)
    restored.build(context, "formal", None)
    assert restored._prior_warmup == 3 and restored._warmup == 2
    for i in range(3):
        metrics = restored.update_once(context, i)
        assert metrics["critic_only"] == (i < 2)
        context.updates_used += 1
        restored.after_update(context, i, metrics)
