"""ABI4 production contracts and CPU lifecycle; never integrates physics."""

import copy
import ctypes as C
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from native_stand.artifacts.checkpoint import LoadedSource, save_full
from native_stand.artifacts.io import file_sha256, json_digest
from native_stand.experiments.config import load_job_config
from native_stand.learning.policies.factory import build_policy
from native_stand.learning.ppo.jump_unified import DirectActor, validate_phases
from native_stand.simulation.jump_build import build
from native_stand.simulation.jump_unified import UnifiedJump
from native_stand.simulation.jump_shared_c import declared_input, Frame, array
from native_stand.tasks.jump.unified import contract, decode_raw

FROZEN_TEST_SCOPE = "PORTABLE"
ROOT = Path(__file__).parents[1]


def job():
    return load_job_config(ROOT / "configs/jobs/jump_unified_raw37_wheel48.json")


def observation(n=2):
    x = torch.zeros(n, 37)
    x[:, 25:31] = torch.tensor([0, 0, 1, 0, 0, 1])
    x[:, 24] = 0.68
    return TensorDict(
        dict(
            student_actor=x,
            student_critic=torch.zeros(n, 6370),
            recovery_actor_valid=torch.ones(n, 1, dtype=torch.bool),
            jump_C_phase=torch.full((n, 1), 9),
            jump_C_owner=torch.full((n, 1), 2),
        ),
        batch_size=[n],
    )


@pytest.fixture(scope="module")
def binding(tmp_path_factory):
    return build(tmp_path_factory.mktemp("ABI4") / "control.so")


def feedback(tick=0):
    return declared_input(
        now=tick,
        gyro=[0] * 3,
        gravity=[0, -1, 0],
        q=[0] * 4,
        dq=[0] * 6,
        command=[0, 0, 0.34],
        adopted=[0] * 6,
        adopted_sequence=1,
        sensor_tick=tick,
        imu_tick=tick,
        feedback_ticks=[tick] * 6,
        accel=[0, -9.81, 0],
        driver_tau=[0] * 4,
        internal_tau=[10.0, 0.0, 10.0, 0.0],
        support_J=np.array([0.1, 0, 0, 1] * 2),
        support_tick=tick,
        support_valid=3,
        leg_length=[0.28, 0.28],
        leg_spread=[1, 1],
        motion_entry=True,
        upstream_owner=2,
    )


def test_C_ABI4_linear_gain_guards_balance_and_state(binding):
    c = UnifiedJump(binding["path"], 1, library_sha=binding["sha256"])
    assert c.lib.RLJumpSize(0) == 4
    raw = np.array([4, -4, 12, 2, -2, -12], np.float32)
    request = np.empty(6, np.float32)
    protected = request.copy()
    assert c.lib.RLJumpDecodeMean(raw.ctypes.data, request.ctypes.data, protected.ctypes.data) == 0
    np.testing.assert_allclose(request, [2, -2, 120, 1, -1, -120])
    np.testing.assert_allclose(protected, [1.5, -1.5, 80, 1, -1, -80])
    cfg = c.config
    cfg.enabled = cfg.model_available = 1
    cfg.model_id = 1
    cfg.model_input_id = c.INPUT_CONTRACT
    cfg.model_output_id = c.OUTPUT_CONTRACT
    c.step([feedback()])
    obs, valid = c.observe()
    assert valid[0] and c.states[0].phase == 9 and c.states[0].owner == 2
    assert not obs[0, 31:].any()
    assert c.publish(raw[None], valid)[0] >= 0
    c.inputs[0] = feedback(5)
    rc, frames = c.pd()
    assert rc[0] >= 0
    assert abs(frames[0]["final_model_nm"][2]) <= 4.8
    f = Frame()
    assert c.lib.RLJumpMotorCommand(C.byref(feedback()), protected.ctypes.data, C.byref(f)) == 0
    np.testing.assert_allclose(array(f.final_model_nm)[[2, 5]], [4.8, -4.8], atol=1e-6)
    state = c.state_dict()
    c.load_state_dict(state)
    assert c.state_dict() == state
    bad = copy.deepcopy(state)
    bad["abi"] = 3
    with pytest.raises(ValueError):
        c.load_state_dict(bad)


def test_balance_not_crouch_and_invalid_zero_rejected():
    obs = observation()
    validate_phases(obs, torch.ones(2, dtype=torch.bool))
    obs["jump_C_phase"][0] = 1
    with pytest.raises(ValueError):
        validate_phases(obs, torch.ones(2, dtype=torch.bool))
    obs["student_actor"][0, 31] = 1
    validate_phases(obs, torch.ones(2, dtype=torch.bool))
    obs["jump_C_owner"][1] = 1
    with pytest.raises(ValueError):
        validate_phases(obs, torch.ones(2, dtype=torch.bool))


def test_raw_gaussian_and_two_adam_full_resume(tmp_path):
    j = job()
    j = replace(
        j,
        model=replace(j.model, actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        runtime=replace(j.runtime, envs=2, steps_per_env=4),
        ppo=replace(j.ppo, epochs=2, minibatches=1),
        diagnostics=dict(
            j.diagnostics, minimum_unique_actor_samples=1, minimum_minibatch_samples=1
        ),
    )
    obs = observation()
    h = build_policy(j, obs)
    p = h["ppo"]
    with torch.no_grad():
        p.actor.mlp[-1].bias.fill_(3.0)
    assert (p.actor(obs) > 1).all()
    actions = []
    with torch.inference_mode():
        for _ in range(4):
            a = p.act(obs)
            actions.append(a.clone())
            p.process_env_step(obs, torch.ones(2, 1), torch.zeros(2, dtype=torch.bool), {})
        p.compute_returns(obs)
    assert torch.equal(p.storage.actions[:, 0], torch.stack(actions)[:, 0])
    result = p.update()
    assert result and p.mask_summary["actor_optimizer_steps"] > 0
    state = h["capture"]({"updates": 1})
    save_full(tmp_path / "full.pt", state)
    h2 = build_policy(j, obs)
    h2["restore"](state)
    for k, v in p.actor.state_dict().items():
        assert torch.equal(v, h2["ppo"].actor.state_dict()[k])
    assert all(v.requires_grad for v in p.actor.parameters())
    bad = dict(state, version="ABI3")
    with pytest.raises(ValueError):
        h2["restore"](bad)


def test_contract_and_backend_reject_old_and_isaac():
    from native_stand.simulation.dispatch import require_supported, execution_service
    from native_stand.tasks.jump.spec import build_task

    j = job()
    assert build_task(j).contract == contract()
    assert execution_service(j)["module"] == "native_stand.simulation.jump_backend"
    with pytest.raises(ValueError, match="独立评估"):
        require_supported(j, "evaluate")
    assert execution_service(replace(j, backend="isaac"))["service"] == "unsupported"
    with pytest.raises(ValueError):
        require_supported(replace(j, backend="isaac"), "train")
    bad = copy.deepcopy(j.assets["jump_contract_data"])
    bad["ABI"] = 2
    with pytest.raises(ValueError):
        build_task(replace(j, assets=dict(j.assets, jump_contract_data=bad)))


def test_course_time_old_generation_duplicate_and_restore():
    from native_stand.tasks.jump.adaptive_course import AdaptiveCourse, CourseConfig

    config = CourseConfig(window_episodes=1, promote_windows=2, demote_windows=2)
    c = AdaptiveCourse(seed=1, plant_id="p", DR_id="nominal", config=config)
    old = c.next_episode()
    c._move(1, "test revision")
    state = c.state_dict()
    restored = AdaptiveCourse.restore(state)
    for course in (c, restored):
        assert course.record(old, jump_control_s=0.2, success=False, fault=False)[
            "stale_generation"
        ]
        assert sum(course.control_s.values()) == 0
        with pytest.raises(ValueError):
            course.record(old, jump_control_s=0.2, success=False, fault=False)
    assert c.state_dict() == restored.state_dict()


def packet(**changes):
    from native_stand.tasks.jump.unified_reward import UnifiedPacket

    values = dict(
        dt_s=0.001,
        phase=9,
        owner=2,
        COM_height_m=0.4,
        COM_velocity_m_s=(0.0, 0.0, 0.0),
        root_height_m=0.34,
        tilt_rad=(0.0, 0.0),
        omega_rad_s=(0.0, 0.0, 0.0),
        leg_length_m=(0.28, 0.28),
        leg_swing_rad=(0.0, 0.0),
        spring_length_m=(0.18, 0.18),
        contact=(True, True),
        tire_clearance_m=(0.0, 0.0),
        wheel_normal_force_N=(100.0, 100.0),
        contact_slip_m_s=(0.0, 0.0),
        actual_motor_Nm=(0.0,) * 6,
        actual_dq_rad_s=(0.0,) * 6,
        adopted_target=(0.0,) * 6,
        force_valid=True,
    )
    values.update(changes)
    return UnifiedPacket(**values)


def test_unified_reward_dt_balance_active_and_passive_provenance():
    from native_stand.tasks.jump.unified_reward import UnifiedReward, command_relative
    from native_stand.tasks.jump.dense_reward import TaskSpec

    spec = TaskSpec(kind="ground_active_jump", horizon_s=16.0, mass_kg=20.0)
    first = UnifiedReward.start(spec, packet(dt_s=0.0))
    second = UnifiedReward.restore(first.state_dict())
    a = first.step(packet(dt_s=0.01, root_height_m=0.24))
    b = [second.step(packet(root_height_m=0.24)) for _ in range(10)]
    assert a["total"] == pytest.approx(sum(x["total"] for x in b))
    assert first.jump_control_s == pytest.approx(0.01)
    crouch = UnifiedReward.start(spec, packet(dt_s=0.0, phase=1))
    r = crouch.step(packet(phase=1, root_height_m=0.20, leg_length_m=(0.145, 0.145)))
    assert r["terms"]["crouch_progress"] > 0 and "balance_height" not in r["terms"]
    r = crouch.step(packet(phase=2, COM_velocity_m_s=(0.0, 0.0, 1.0)))
    assert r["terms"]["push_progress"] > 0
    timeout = UnifiedReward.start(replace(spec, horizon_s=0.001), packet(dt_s=0.0))
    finite = timeout.step(packet())
    assert (
        finite["terminated"] and not finite["truncated"] and not finite["timeout_bootstrap_allowed"]
    )
    # A passive airborne start can cushion but cannot claim active takeoff/apex rewards.
    drop = UnifiedReward.start(
        replace(spec, kind="landing_only"),
        packet(dt_s=0.0, phase=4, contact=(False, False), tire_clearance_m=(0.03, 0.03)),
    )
    for _ in range(10):
        result = drop.step(
            packet(
                phase=3,
                contact=(False, False),
                tire_clearance_m=(0.03, 0.03),
                COM_velocity_m_s=(0, 0, 1.0),
            )
        )
        assert result["terms"]["takeoff"] == result["terms"]["apex"] == 0
        assert not result["active_jump_success"]
    relative = command_relative(packet(COM_velocity_m_s=(1, 0, 0)), [1, 0], own_completed=True)
    assert relative.COM_velocity_m_s == (0, 0, 0) and relative.motion_confirmed
    # Completion signal alone is insufficient without true Jump progress/landing.
    result = first.step(relative)
    assert not result["active_jump_success"] and result["terms"]["own_completion"] == 0


def test_course_promotes_and_demotes_from_ended_training_episodes():
    from native_stand.tasks.jump.adaptive_course import AdaptiveCourse, CourseConfig

    cfg = CourseConfig(window_episodes=2, promote_windows=2, demote_windows=2)
    c = AdaptiveCourse(seed=8, plant_id="p", DR_id="nominal", config=cfg)
    assert c.metrics()["jump_success_rate"] is None
    for _ in range(100):
        r = c.next_episode()
        c.record(
            r, jump_control_s=0.1 if r["kind"] == "landing_only" else 1.6, success=True, fault=False
        )
        if c.level == 1:
            break
    assert c.level == 1 and c.generation == 1
    for _ in range(100):
        r = c.next_episode()
        c.record(r, jump_control_s=0.1, success=False, fault=False)
        if c.level == 0:
            break
    assert c.level == 0 and c.generation == 2


def test_candidates_do_not_borrow_landing_or_balance_for_active():
    from dataclasses import asdict
    from native_stand.tasks.jump.adaptive_course import CourseConfig
    from native_stand.simulation.jump_adapter import consider

    index = dict(candidates_by_objective={"balance": {}, "landing_only": {}, "active_jump": {}})
    rows = []
    for i in range(32):
        rows.append(
            dict(
                id=i,
                request=dict(
                    kind="landing_only", plant_id="p", DR_id="nominal", level=0, generation=0
                ),
                success=True,
                fault=False,
                balance_seconds=1.5,
                balance_good_seconds=1.5,
            )
        )
    window = dict(rows=rows, selection=asdict(CourseConfig()))
    ranked = consider(
        index, window, dict(update=100, checkpoint="/tmp/example.pt", sha256="a" * 64)
    )
    assert ranked["candidates_by_objective"]["landing_only"]
    assert ranked["candidates_by_objective"]["balance"]
    assert ranked["candidates_by_objective"]["active_jump"] == {}
    # Two generations with16 rows each are UNKNOWN; no32-row pool can be fabricated.
    for row in rows[16:]:
        row["request"]["generation"] = 1
    ranked = consider(
        index, window, dict(update=100, checkpoint="/tmp/example.pt", sha256="a" * 64)
    )
    assert not any(ranked["candidates_by_objective"].values())


def test_C_own_settle_returns_to_balance_without_motion(binding):
    c = UnifiedJump(binding["path"], 1, library_sha=binding["sha256"])
    assert not c.config.enabled
    cfg = c.config
    cfg.enabled = cfg.model_available = 1
    cfg.model_id = 1
    cfg.model_input_id = c.INPUT_CONTRACT
    cfg.model_output_id = c.OUTPUT_CONTRACT
    for tick in range(0, 125, 5):
        i = feedback(tick)
        i.key_down = int(tick == 20)
        i.motion_success = 0
        c.step([i])
    assert c.states[0].phase == 1 and c.states[0].owner == 2
    c.states[0].phase = 5
    c.states[0].phase_tick = 125
    c.states[0].stable_valid = 0
    for tick in range(125, 1135, 5):
        i = feedback(tick)
        i.motion_success = 0
        c.step([i])
    assert c.states[0].phase == 9 and c.states[0].owner == 2
    obs, valid = c.observe()
    assert valid[0] and not obs[0, 31:].any()


def test_continuous_and_segmented_training_match_two_Adam_std_and_RNG():
    j = job()
    j = replace(
        j,
        model=replace(j.model, actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        runtime=replace(j.runtime, envs=2, steps_per_env=4),
        ppo=replace(j.ppo, epochs=1, minibatches=1),
        diagnostics=dict(
            j.diagnostics, minimum_unique_actor_samples=1, minimum_minibatch_samples=1
        ),
    )
    obs = observation()

    def train(handle):
        p = handle["ppo"]
        with torch.inference_mode():
            for _ in range(4):
                action = p.act(obs)
                p.process_env_step(
                    obs,
                    1 - action.square().mean(-1, keepdim=True),
                    torch.zeros(2, dtype=torch.bool),
                    {},
                )
            p.compute_returns(obs)
        p.update()

    torch.manual_seed(421)
    first = build_policy(j, obs)
    train(first)
    saved = first["capture"]({"updates": 1})
    train(first)
    continuous = first["capture"]({"updates": 2})
    second = build_policy(j, obs)
    second["restore"](saved)
    train(second)
    segmented = second["capture"]({"updates": 2})

    def equal(a, b):
        if torch.is_tensor(a):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        return a == b

    for key in ("actor", "critic", "optimizer", "std", "training_rng", "counters"):
        assert equal(continuous[key], segmented[key]), key


def test_shared_reward_shape_and_bootstrap_are_per_environment_once():
    from types import SimpleNamespace as NS
    from native_stand.simulation.engine import GenericPPOAdapter
    from native_stand.simulation.driver import Backend

    adapter = GenericPPOAdapter(Backend("fixture", lambda j: None, {}))
    calls = []
    adapter.ppo = NS(
        gamma=0.9,
        critic=lambda named: (
            calls.append(named["student_critic"].clone()) or torch.tensor([[10.0], [20.0]])
        ),
    )
    adapter.env = NS(num_envs=2, infinite_horizon=True)
    outcome = dict(
        rewards=torch.tensor([1.0, 2.0]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([True, True]),
        terminal_observations={"student_critic": torch.tensor([[7.0], [8.0]])},
    )
    value = adapter._stored_reward(outcome)
    assert torch.equal(value, torch.tensor([[10.0], [2.0]])) and len(calls) == 1
    adapter.env.infinite_horizon = False
    assert torch.equal(adapter._stored_reward(outcome), torch.tensor([[1.0], [2.0]]))
    assert len(calls) == 1
    with pytest.raises(ValueError):
        adapter._stored_reward(dict(outcome, rewards=torch.ones(2, 2)))
