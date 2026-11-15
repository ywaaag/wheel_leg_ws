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


def adapter_fixture(
    tmp_path, *, source=None, config_name="recovery_leg_driven2100_10000_attempt3.json"
):
    import dataclasses
    from pathlib import Path
    from types import SimpleNamespace as NS

    from tensordict import TensorDict

    from native_stand.artifacts.io import file_sha256, write_json
    from native_stand.artifacts.model_refs import ModelIndex
    from native_stand.experiments.config import load_job_config
    from native_stand.simulation.driver import Backend

    job = load_job_config(Path(__file__).resolve().parents[1] / "configs/jobs" / config_name)
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
            sampling_distribution=dict(
                fixed_std=[
                    job.ppo.exploration_leg_rad,
                    job.ppo.exploration_leg_rad,
                    job.ppo.exploration_wheel_rad_s,
                    job.ppo.exploration_leg_rad,
                    job.ppo.exploration_leg_rad,
                    job.ppo.exploration_wheel_rad_s,
                ]
            ),
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


ANTI_PRONE = dict(
    version="recovery_anti_prone_v1", grace_s=0.8, ramp_s=0.8, max_rate=2.0, height_m=0.26
)


def test_anti_prone_grace_ramp_progress_and_phase_gate():
    from native_stand.tasks.recovery.leg_driven import anti_prone_density

    def cost(age, *, phase=0, progress=0.0, height=0.17, gate=False):
        return float(
            anti_prone_density(
                ANTI_PRONE,
                phase=torch.tensor([phase]),
                progress=torch.tensor([progress]),
                stage_age=torch.tensor([age]),
                height=torch.tensor([height]),
                gate=torch.tensor([gate]),
            )
        )

    assert cost(0.8) == 0
    assert cost(1.2) == pytest.approx(-1.0)
    assert cost(1.6) == pytest.approx(-2.0)
    assert cost(3.0, progress=1.0) == pytest.approx(-1.0)
    assert cost(3.0, gate=True) == 0
    assert cost(3.0, phase=2) == 0
    assert cost(3.0, phase=3) == 0
    assert cost(3.0, height=0.27) == 0


def test_real_reward_consumer_prone_cost_and_failure_tail():
    old, g = life()
    new, _ = life()
    from native_stand.tasks.recovery.leg_driven import anti_prone_config

    new.anti_prone = anti_prone_config(ANTI_PRONE)
    baseline = advance(old, g, t=2.0)
    modified = advance(new, g, t=2.0)
    assert torch.equal(old.phase, new.phase)
    assert torch.equal(old.last_target, new.last_target)
    assert torch.allclose(modified["reward"] - baseline["reward"], torch.full((2,), -0.02))
    assert torch.equal(modified["density"]["anti_prone"], torch.full((2,), -2.0))
    baseline = advance(old, g, t=2.01, terminal=True)
    modified = advance(new, g, t=2.01, terminal=True)
    assert torch.allclose(modified["anti_prone_failure_tail"], torch.full((2,), 7.98))
    assert torch.allclose(
        modified["reward"] - baseline["reward"], torch.full((2,), -8.0), atol=1e-5
    )


def test_prone_cost_does_not_reward_wheel_motion_and_uses_checkpointed_state():
    from native_stand.tasks.recovery.leg_driven import anti_prone_config

    first, g = life()
    first.anti_prone = anti_prone_config(ANTI_PRONE)
    advance(first, g, t=1.0)
    second, _ = life()
    second.anti_prone = anti_prone_config(ANTI_PRONE)
    second.load_state_dict(first.state_dict())
    a = advance(first, g, t=1.4)
    b = advance(second, g, t=1.4, wheel=2.0)
    assert torch.equal(a["density"]["anti_prone"], b["density"]["anti_prone"])
    assert (b["reward"] < a["reward"]).all()
    first.reset(torch.arange(2), torch.full((2,), 0.168), torch.zeros(2))
    first.configure_reset(
        torch.arange(2), g, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)
    )
    assert torch.equal(advance(first, g, t=0.1)["density"]["anti_prone"], torch.zeros(2))


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_rate", float("nan")),
        ("ramp_s", 0.0),
        ("grace_s", -1.0),
        ("height_m", 0.34),
        ("max_rate", True),
    ],
)
def test_invalid_prone_recipe_rejected(key, value):
    from native_stand.tasks.recovery.leg_driven import anti_prone_config

    with pytest.raises(ValueError):
        anti_prone_config(dict(ANTI_PRONE, **{key: value}))


def test_prepared_prone_recipe_public_backend_validation(tmp_path):
    import dataclasses
    from pathlib import Path

    from native_stand.artifacts.io import file_sha256, write_json
    from native_stand.experiments.config import load_job_config
    from native_stand.simulation.recovery_isaac.backend import validate_job

    path = (
        Path(__file__).resolve().parents[1]
        / "configs/jobs/recovery_leg_driven2100_anti_prone_prepared.json"
    )
    job = load_job_config(path)
    geometry = tmp_path / "geometry.json"
    write_json(geometry, GEO)
    job = dataclasses.replace(
        job,
        assets={
            "recovery_leg_driven_geometry": dict(path=str(geometry), sha256=file_sha256(geometry))
        },
    )
    validate_job(job)
    assert job.reward.fields["anti_prone"] == ANTI_PRONE
    assert job.limits.formal_updates == 0
    bad = dataclasses.replace(
        job,
        reward=dataclasses.replace(
            job.reward, fields=dict(anti_prone=dict(ANTI_PRONE, ramp_s=0.0))
        ),
    )
    with pytest.raises(ValueError, match="positive"):
        validate_job(bad)


PROGRESSION = dict(
    version="recovery_stage_progress_v1",
    remaining_rate=3.0,
    completion_bonus=[6.0, 6.0, 10.0],
    failure_tail_rate=9.0,
    terminate_invalid_sequence=True,
)


def test_stage_error_penalizes_partial_prone_and_progress_reduces_it():
    from native_stand.tasks.recovery.leg_driven import progression_config

    a, g = life()
    b, h = life()
    a.progression = progression_config(PROGRESSION)
    b.progression = progression_config(PROGRESSION)
    g[:, :2] = 0.24
    h[:, :2] = 0.19
    # Stay on the legal contraction-angle curve; no artificial premature turn.
    for x in [g, h]:
        f = (0.376 - x[:, :2]) / (0.376 - 0.145)
        x[:, 2:] = -1.276 + f * (-0.65 + 1.276)
    ra = advance(a, g, t=2.0)
    rb = advance(b, h, t=2.0)
    assert not ra["terminated"].any() and not rb["terminated"].any()
    assert (rb["density"]["stage_remaining"] > ra["density"]["stage_remaining"]).all()
    assert (ra["density"]["stage_remaining"] < 0).all()


def test_new_progress_recipe_terminates_irrecoverable_sequence_not_old_recipe():
    from native_stand.tasks.recovery.leg_driven import progression_config

    a, g = life(phase=1)
    b, _ = life(phase=1)
    a.progression = progression_config(PROGRESSION)
    g[:, :2] = 0.21
    ra = advance(a, g, t=1.0)
    rb = advance(b, g, t=1.0)
    assert ra["terminated"].all() and not rb["terminated"].any()
    assert not ra["success"].any()
    assert torch.allclose(ra["failure_tail"], torch.full((2,), 45.0))


def test_turn_progress_starts_at_zero_after_real_shortening_completion():
    from native_stand.tasks.recovery.leg_driven import progression_config

    a, g = life()
    a.progression = progression_config(PROGRESSION)
    g[:, :2] = 0.145
    g[:, 2:] = -0.65
    for i in range(10):
        out = advance(a, g, t=0.01 * (i + 1))
    assert a.phase.tolist() == [1, 1]
    assert torch.allclose(a.progress, torch.zeros(2))
    assert torch.equal(out["phase_bonus"], torch.full((2,), 6.0))
    assert torch.equal(out["density"]["anti_prone"], torch.zeros(2))


def test_new_phase_gate_cannot_reward_short_wrong_angle():
    from native_stand.tasks.recovery.leg_driven import progression_config

    a, g = life()
    a.progression = progression_config(PROGRESSION)
    g[:, :2] = 0.145
    g[:, 2:] = -0.82
    for i in range(15):
        advance(a, g, t=0.01 * (i + 1))
    assert a.phase.tolist() == [0, 0] and a.sequence_valid.all()


def test_progress_tail_bound_rejects_early_failure_reward_escape():
    from native_stand.tasks.recovery.leg_driven import progression_config

    with pytest.raises(ValueError, match="cover"):
        Lifecycle(
            2,
            "cpu",
            GEO,
            0.998,
            anti_prone=ANTI_PRONE,
            progression=dict(PROGRESSION, failure_tail_rate=8.0),
        )
    with pytest.raises(ValueError):
        progression_config(dict(PROGRESSION, completion_bonus=[1.0, 2.0]))


def test_reduced_physical_wheel_std_preserves_deterministic_targets():
    from native_stand.learning.ppo.recovery_cubic import PhysicalActor

    contract = dict(
        center_rad=[-0.13, -0.15, 0.13, -0.15], leg_lower_rad=[-1.5] * 4, leg_upper_rad=[1.5] * 4
    )
    first = PhysicalActor(contract, hidden_dims=(8,))
    changed = copy.deepcopy(contract)
    changed["sampling_distribution"] = dict(fixed_std=[0.02, 0.02, 0.5, 0.02, 0.02, 0.5])
    second = PhysicalActor(changed, hidden_dims=(8,))
    second.mlp.load_state_dict(first.mlp.state_dict())
    x = torch.randn(20, 31)
    assert torch.equal(first.physical_mean(x), second.physical_mean(x))
    assert second.distribution(x).stddev[0, [2, 5]].tolist() == [0.5, 0.5]
    assert first.distribution(x).stddev[0, [2, 5]].tolist() == [2.0, 2.0]


def test_fraction_recipe_preserves_all_groups_and_focuses_bottleneck():
    from native_stand.tasks.recovery.leg_driven import GROUPS, course_fractions

    recipe = [
        dict(groups=list(GROUPS), fractions=f)
        for f in [[0.4, 0.35, 0.15, 0.1], [0.7, 0.1, 0.1, 0.1], [0.85, 0.05, 0.05, 0.05]]
    ]
    c = Course(3, fractions=course_fractions(recipe))
    for _i in range(200):
        s = c.choose()
        c.record(s["group"], student_seconds=1.0, teacher_seconds=0.0, passed=False)
    assert [c.counts[g]["episodes"] for g in GROUPS] == [80, 70, 30, 20]
    with pytest.raises(ValueError):
        course_fractions([dict(groups=list(GROUPS), fractions=[1.0, 0.0, 0.0, 0.0])] * 3)


def test_new_recipe_actual_worker_factory_and_first12_optimization(tmp_path, monkeypatch):
    from native_stand.experiments import worker
    from native_stand.simulation import driver
    from native_stand.simulation.recovery_adapter import RecoveryAdapter

    c, backend = adapter_fixture(tmp_path, config_name="recovery_leg_progress_remaining8594.json")
    monkeypatch.setattr(worker, "resolve_backend", lambda job: backend)

    def consume(context, adapter, *, stage, resume_from):
        assert type(adapter) is RecoveryAdapter
        adapter.build(context, stage, resume_from)
        assert adapter.ppo.actor.physical_std[[2, 5]].tolist() == [0.5, 0.5]
        for i in range(12):
            metrics = adapter.update_once(context, i)
            assert metrics["critic_only"] == (i < 5)
            context.updates_used += 1
            adapter.after_update(context, i, metrics)
        state = adapter.capture(context, dict(updates=12))
        assert state["warmup_updates_done"] == 5
        assert len(state["optimizer"]["actor"]["state"]) > 0
        assert "recovery_training_window" in state
        return state

    monkeypatch.setattr(driver, "run_stage", consume)
    result = worker._execute_stage(c, "formal", None)
    assert result["distribution"]["fixed_std"][2] == 0.5
    import json

    assert json.loads((tmp_path / "engineering_smoke.json").read_text())["engineering_smoke_passed"]


def test_progression_legal_sequence_can_still_reach_real_lifecycle_success():
    from native_stand.tasks.recovery.leg_driven import anti_prone_config, progression_config

    a, g = life()
    a.progression = progression_config(PROGRESSION)
    a.anti_prone = anti_prone_config(ANTI_PRONE)
    t = 0.0
    for phase in [0, 1, 2, 3]:
        g[:, :2] = 0.145
        g[:, 2:] = -0.65 if phase == 0 else 0.0
        for _i in range(12 if phase < 3 else 51):
            t += 0.01
            out = advance(a, g, t=t, height=0.168 if phase < 2 else 0.34)
    assert out["success"].all() and a.sequence_valid.all()
    assert (out["reward"] > 0).all()
    assert torch.equal(out["failure_tail"], torch.zeros(2))


def test_revised_backend_rejects_old_noise_and_covers_allocation_config(tmp_path):
    import dataclasses
    from pathlib import Path

    from native_stand.artifacts.io import file_sha256, write_json
    from native_stand.experiments.config import load_job_config
    from native_stand.simulation.recovery_isaac.backend import validate_job

    j = load_job_config(
        Path(__file__).resolve().parents[1]
        / "configs/jobs/recovery_leg_progress_remaining8594.json"
    )
    g = tmp_path / "geometry.json"
    write_json(g, GEO)
    j = dataclasses.replace(
        j, assets=dict(recovery_leg_driven_geometry=dict(path=str(g), sha256=file_sha256(g)))
    )
    validate_job(j)
    assert j.limits.formal_updates == 8594 and j.ppo.exploration_wheel_rad_s == 0.5
    with pytest.raises(ValueError, match="std"):
        validate_job(
            dataclasses.replace(j, ppo=dataclasses.replace(j.ppo, exploration_wheel_rad_s=2.0))
        )


def test_phase_observability_records_fractional_improvement(tmp_path):
    from native_stand.simulation.recovery_adapter import RecoveryAdapter

    c, backend = adapter_fixture(tmp_path, config_name="recovery_leg_progress_remaining8594.json")
    adapter = RecoveryAdapter(backend)
    adapter.build(c, "formal", None)
    io = adapter.env
    io.env.completed = [
        dict(
            group="full_Swing",
            nominal_anchor=True,
            command_height_m=0.34,
            height_dynamic=False,
            DR_strength=0.0,
            student_seconds=6.0,
            teacher_seconds=0.0,
            steps=600,
            success=False,
            physical_quality=False,
            leg_driven=dict(
                course_level=0,
                initial_phase=0,
                final_phase=0,
                progress=0.6,
                wheel_rms=[1.0, 1.0],
                sequence_valid=True,
                wheel_peak=[2.0, 2.0],
                tilt_peak_rad=0.1,
                quality=False,
            ),
        )
    ]
    io.env.completed_count = 1
    adapter.readback.before_update(adapter.ppo, allow_actor=False)
    c.updates_used = 1
    adapter.after_update(c, 0, {})
    row = next(iter(adapter.window.values()))
    assert row["phase_progress_sum"] == 0 and row["continuous_progress_sum"] == 0.6
    assert row["phase_reached"] == [1, 0, 0, 0]


def test_progress_recipe_public_preflight_enforces_acceptance_and_budget(tmp_path):
    import dataclasses
    from pathlib import Path

    from native_stand.artifacts.io import file_sha256, write_json
    from native_stand.experiments.config import load_job_config
    from native_stand.learning.ppo.recovery_cubic import PhysicalActor
    from native_stand.simulation.recovery_isaac.backend import preflight

    j = load_job_config(
        Path(__file__).resolve().parents[1]
        / "configs/jobs/recovery_leg_progress_remaining8594.json"
    )

    def ref(name, value):
        p = tmp_path / name
        write_json(p, value)
        return dict(path=str(p), sha256=file_sha256(p))

    contract = dict(
        center_rad=[-0.13, -0.15, 0.13, -0.15],
        leg_lower_rad=[-1.5] * 4,
        leg_upper_rad=[1.5] * 4,
        sampling_distribution=dict(fixed_std=[0.02, 0.02, 0.5, 0.02, 0.02, 0.5]),
    )
    actor = PhysicalActor(contract)
    path = tmp_path / "actor.pt"
    torch.save(dict(actor_only=True, actor=actor.state_dict(), output_contract=contract), path)
    actor_ref = dict(path=str(path), sha256=file_sha256(path))
    runtime = ref("runtime.json", dict(bindings={}))
    state = ref("state.json", dict(status="STOPPED_BOUNDED", updates_used=1406))
    budget = ref("budget.json", dict(used=dict(formal=1406, smoke=0)))
    allocation = ref(
        "allocation.json",
        dict(
            prior_updates_used=1406,
            authorized_total_updates=10000,
            new_run_updates=8594,
            prior_attempts=[dict(state=state, budget=budget, used=1406)],
        ),
    )
    accept = ref(
        "accept.json",
        dict(
            CPU_leg_driven_wiring_accepted=False,
            actor_sha256=actor_ref["sha256"],
            runtime_contract_sha256=runtime["sha256"],
        ),
    )
    assets = dict(
        recovery=j.assets["recovery"],
        recovery_initial_actor=actor_ref,
        recovery_training_contract=ref("contract.json", contract),
        recovery_isaac_runtime=runtime,
        recovery_isaac_acceptance=accept,
        recovery_leg_driven_geometry=ref("geometry.json", GEO),
        training_authorization=ref(
            "authorization.json", dict(new_training_updates=10000, wall_s=None)
        ),
        training_allocation=allocation,
    )
    j = dataclasses.replace(
        j,
        assets=assets,
        initialization=dataclasses.replace(
            j.initialization, checkpoint=str(path), sha256=actor_ref["sha256"]
        ),
    )
    with pytest.raises(ValueError, match="CPU consumer"):
        preflight(j)
    assets["recovery_isaac_acceptance"] = ref(
        "accepted.json",
        dict(
            CPU_leg_driven_wiring_accepted=True,
            actor_sha256=actor_ref["sha256"],
            runtime_contract_sha256=runtime["sha256"],
        ),
    )
    assert preflight(j)["PPO_updates"] == 0
    # Public load_job_config represents initialization paths as Path objects;
    # a JSON-bound transition necessarily stores a string. Compare identities.
    from native_stand.artifacts.io import json_digest

    full = dataclasses.replace(
        j, initialization=dataclasses.replace(j.initialization, mode="full", checkpoint=path)
    )
    transition = ref(
        "transition.json",
        dict(
            version="recovery_full_recipe_transition_v1",
            source_checkpoint=dict(path=str(path), sha256=file_sha256(path)),
            to_reward_sha256=json_digest(full.as_run_record()["reward"]),
            to_curriculum_sha256=json_digest(full.as_run_record()["curriculum"]),
            fresh_physical_reset=True,
        ),
    )
    full = dataclasses.replace(
        full, assets=dict(full.assets, recovery_recipe_transition=transition)
    )
    assert preflight(full)["PPO_updates"] == 0
    invalid = dataclasses.replace(j, limits=dataclasses.replace(j.limits, formal_updates=8595))
    with pytest.raises(ValueError, match="refund"):
        preflight(invalid)


def overlap_life(phase=0):
    recipe = dict(PROGRESSION, coordination="overlap_short_turn_v1")
    a = Lifecycle(2, "cpu", GEO, 0.9980019986673331, anti_prone=ANTI_PRONE, progression=recipe)
    _, g = life(phase=phase)
    a.reset(torch.arange(2), torch.full((2,), 0.168), torch.zeros(2))
    a.configure_reset(torch.arange(2), g, torch.full((2,), phase), torch.zeros(2, dtype=torch.long))
    return a, g


def test_overlap_accepts_rotation_during_contraction_outside_old_corridor():
    a, g = overlap_life()
    old, _ = life()
    old.progression = copy.deepcopy(PROGRESSION)
    g[:, :2] = 0.22
    g[:, 2:] = -0.1
    new = advance(a, g, t=0.5)
    legacy = advance(old, g, t=0.5)
    assert not new["terminated"].any() and not new["success"].any()
    assert legacy["terminated"].all()
    assert a.episode_record(0)["sequence_valid"] is None
    assert a.episode_record(0)["coordination"] == "overlap_short_turn_v1"
    assert "sequence" not in new["density"] and "leg_symmetry" in new["density"]


def test_overlap_has_no_fixed_intermediate_angle_or_pause_before_turning():
    a, g = overlap_life()
    g[:, :2] = 0.145
    g[:, 2:] = 0.0
    first = advance(a, g, t=0.01)
    assert a.phase.tolist() == [1, 1]
    assert torch.equal(first["phase_bonus"], torch.full((2,), 6.0))
    assert not first["terminated"].any()
    for i in range(11):
        advance(a, g, t=0.02 + i * 0.01)
    assert a.phase.tolist() == [2, 2]
    assert a.episode_record(0)["reconfiguration_completed"]


def test_overlap_temporary_extension_is_recoverable_and_not_complete():
    a, g = overlap_life(phase=1)
    g[:, :2] = 0.22
    g[:, 2:] = 0.0
    out = advance(a, g, t=1.0)
    assert not out["terminated"].any() and a.phase.tolist() == [1, 1]
    assert (out["density"]["stage_remaining"] < 0).all()
    g[:, :2] = 0.145
    for i in range(12):
        advance(a, g, t=1.01 + i * 0.01)
    assert a.phase.tolist() == [2, 2]


def test_overlap_keeps_physical_failures_and_prevents_height_only_bypass():
    a, g = overlap_life()
    out = advance(a, g, t=1.0, height=0.34)
    assert a.phase.tolist() == [0, 0] and not out["success"].any()
    assert (out["density"]["stage_remaining"] < 0).all()
    out = advance(a, g, t=1.01, height=0.34, terminal=True)
    assert out["terminated"].all() and not out["success"].any()
    b, h = overlap_life()
    for i in range(3):
        out = advance(b, h, t=0.01 * (i + 1), tilt=0.6)
    assert out["terminated"].all()


def test_overlap_success_still_requires_ready_half_second():
    a, g = overlap_life()
    g[:, :2] = 0.145
    g[:, 2:] = 0.0
    t = 0.0
    for _i in range(25):
        t += 0.01
        out = advance(a, g, t=t, height=0.34)
    assert a.phase.tolist() == [3, 3] and not out["success"].any()
    for _i in range(51):
        t += 0.01
        out = advance(a, g, t=t, height=0.34)
    assert out["success"].all() and a.episode_record(0)["sequence_valid"] is None


def test_recipe_change_requires_bound_transition_and_preserves_full_learning(tmp_path):
    import dataclasses

    from native_stand.artifacts.checkpoint import save_full
    from native_stand.artifacts.io import file_sha256, json_digest, write_json
    from native_stand.simulation.recovery_adapter import RecoveryAdapter

    c, b = adapter_fixture(tmp_path / "old", config_name="recovery_leg_progress_remaining8594.json")
    old = RecoveryAdapter(b)
    old.build(c, "formal", None)
    for i in range(6):
        metrics = old.update_once(c, i)
        c.updates_used += 1
        old.after_update(c, i, metrics)
    state = old.capture(c, dict(updates=6))
    cp = tmp_path / "source.pt"
    save_full(cp, state)
    c, b = adapter_fixture(
        tmp_path / "new", source=cp, config_name="recovery_leg_progress_remaining8594.json"
    )
    fields = copy.deepcopy(c.job.reward.fields)
    fields["progression"]["coordination"] = "overlap_short_turn_v1"
    c.job = dataclasses.replace(
        c.job,
        reward=dataclasses.replace(c.job.reward, fields=fields),
        initialization=dataclasses.replace(c.job.initialization, checkpoint=cp),
    )
    with pytest.raises(ValueError, match="transition identity"):
        RecoveryAdapter(b).build(c, "formal", None)
    recipe = c.job.as_run_record()
    transition = dict(
        version="recovery_full_recipe_transition_v1",
        source_checkpoint=dict(path=str(cp), sha256=file_sha256(cp)),
        from_comparison_sha256=state["training_candidates"]["comparison_sha256"],
        to_reward_sha256=json_digest(recipe["reward"]),
        to_curriculum_sha256=json_digest(recipe["curriculum"]),
        fresh_physical_reset=True,
    )
    receipt = tmp_path / "transition.json"
    write_json(receipt, transition)
    c.job = dataclasses.replace(
        c.job,
        assets=dict(
            c.job.assets,
            recovery_recipe_transition=dict(path=str(receipt), sha256=file_sha256(receipt)),
        ),
    )
    restored = RecoveryAdapter(b)
    restored.build(c, "formal", None)
    assert restored._prior_warmup == 5 and restored._warmup == 0
    assert torch.equal(torch.get_rng_state(), state["training_rng"]["torch_cpu"])
    for owner in ["actor", "critic"]:
        assert all(
            torch.equal(v, getattr(restored.ppo, owner).state_dict()[k])
            for k, v in state[owner].items()
        )
        expected = state["optimizer"][owner]["state"]
        actual = restored.ppo.optimizer.state_dict()[owner]["state"]
        assert set(expected) == set(actual)
        for key in expected:
            assert all(
                torch.equal(v, actual[key][k]) if torch.is_tensor(v) else v == actual[key][k]
                for k, v in expected[key].items()
            )
    assert restored.window == {} and restored.leg_review_history == [] and restored.cursor == 0
    assert restored.pending is None
    assert (
        restored.candidates["comparison_sha256"]
        != state["training_candidates"]["comparison_sha256"]
    )
    metrics = restored.update_once(c, 0)
    assert metrics["actor_optimizer_steps"] > 0 and not metrics["critic_only"]
    c.updates_used = 1
    restored.after_update(c, 0, metrics)
    assert restored.capture(c, dict(updates=1))["recovery_recipe_transition"] == transition


def test_overlap_review_does_not_turn_inapplicable_sequence_into_failures():
    window = rows()
    for row in window:
        row["sequence_applicable"] = False
        row["sequence_valid"] = 0
    record, _, _ = review_window(window, 0, [])
    assert record["passed"]
    window[0]["student_success"] = 0
    record, _, _ = review_window(window, 0, [])
    assert not record["passed"]


def test_public_worker_consumes_overlap_recipe_and_inapplicable_sequence(tmp_path, monkeypatch):
    from native_stand.experiments import worker
    from native_stand.simulation import driver
    from native_stand.simulation.recovery_adapter import RecoveryAdapter

    c, b = adapter_fixture(tmp_path, config_name="recovery_leg_overlap_prepared.json")
    monkeypatch.setattr(worker, "resolve_backend", lambda job: b)

    def consume(context, adapter, *, stage, resume_from):
        assert type(adapter) is RecoveryAdapter
        adapter.build(context, stage, resume_from)
        assert context.job.reward.fields["progression"]["coordination"] == "overlap_short_turn_v1"
        metrics = adapter.update_once(context, 0)
        context.updates_used = 1
        adapter.env.env.completed = [
            dict(
                group="full_Swing",
                nominal_anchor=True,
                command_height_m=0.34,
                height_dynamic=False,
                DR_strength=0.0,
                student_seconds=1.0,
                teacher_seconds=0.0,
                steps=100,
                success=False,
                physical_quality=False,
                leg_driven=dict(
                    course_level=0,
                    initial_phase=0,
                    final_phase=1,
                    progress=0.5,
                    wheel_rms=[0.5, 0.5],
                    sequence_valid=None,
                    coordination="overlap_short_turn_v1",
                    reconfiguration_completed=False,
                    wheel_peak=[1.0, 1.0],
                    tilt_peak_rad=0.1,
                    quality=True,
                ),
            )
        ]
        adapter.env.env.completed_count = 1
        adapter.after_update(context, 0, metrics)
        state = adapter.capture(context, dict(updates=1))
        row = next(iter(state["recovery_training_window"].values()))
        assert row["sequence_applicable"] is False and row["sequence_valid"] == 0
        assert row["continuous_progress_sum"] == 1.5
        return state

    monkeypatch.setattr(driver, "run_stage", consume)
    assert worker._execute_stage(c, "formal", None)["warmup_updates_done"] == 1
