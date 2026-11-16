"""Physical-event course semantics through scalar/device and publication consumers."""

import copy
import math
from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from test_jump_unified import packet

from native_stand.simulation.jump_adapter import consider, episode_metrics
from native_stand.tasks.jump.adaptive_course import AdaptiveCourse, CourseConfig
from native_stand.tasks.jump.dense_reward import policy_substep_reward
from native_stand.tasks.jump.device_reward import BEST, MEMORY, DeviceReward
from native_stand.tasks.jump.skill_course import SkillLevel, SkillTaskSpec, reward_config
from native_stand.tasks.jump.unified_reward import UnifiedReward

FROZEN_TEST_SCOPE = "PORTABLE"


def scorer(objective=2, **goals):
    return UnifiedReward.start(
        SkillTaskSpec(
            "ground_active_jump",
            6,
            20,
            target_rise_m=0.02,
            target_clearance_m=0.005,
            objective=objective,
            **goals,
        ),
        packet(dt_s=0),
        reward_config(),
    )


def jump_packet(tick, *, height=0.04, **changes):
    air = 20 <= tick < 100
    return packet(
        phase=2 if tick < 20 else 3 if air else 5 if tick < 399 else 9,
        COM_height_m=0.4 + (height * math.sin((tick - 20) * math.pi / 80) if air else 0),
        COM_velocity_m_s=(0, 0, (1 if tick < 60 else -0.2) if air else 0),
        contact=(not air, not air),
        tire_clearance_m=(0.006, 0.006) if air else (0, 0),
        motion_confirmed=tick >= 399,
        **changes,
    )


def episode(reward, *, height=0.04, fault_tick=None):
    out = []
    for tick in range(6000):
        result = reward.step(jump_packet(tick, height=height, fault=tick == fault_tick))
        out.append(result)
        if result["terminated"]:
            break
    return out


def test_objective_boundaries_pay_real_events_without_claiming_full_jump():
    results = [episode(scorer(i)) for i in range(3)]
    assert [len(seq) for seq in results] == [40, 300, 400]
    for i, seq in enumerate(results):
        last = seq[-1]
        assert last["course_success"] and not last["timeout_bootstrap_allowed"]
        assert last["active_jump_success"] == (i == 2)
        assert sum(x["terms"]["takeoff"] for x in seq) == (1.5 if i == 2 else 6)
        assert sum(x["terms"]["own_completion"] for x in seq) == (42 if i == 2 else 0)
    # The landing objective requires a physical apex and landing, not a C completion bit.
    assert results[1][-1]["landing_stable"]
    assert not results[1][-1]["terms"]["own_completion"]


@pytest.mark.parametrize(
    "case", ["passive", "no_ground_push", "assist", "single_wheel", "5mm", "downward", "19ms"]
)
def test_airborne_appearance_without_active_takeoff_provenance_is_rejected(case):
    r = scorer(0)
    if case == "passive":
        r = UnifiedReward.start(
            r.spec, packet(dt_s=0, contact=(False, False), tire_clearance_m=(0.02, 0.02)), r.config
        )
    if case not in ("passive", "no_ground_push"):
        r.step(packet(phase=2))
    for _ in range(19 if case == "19ms" else 40):
        out = r.step(
            packet(
                phase=3 if case != "no_ground_push" else 2,
                contact=(True, False) if case == "single_wheel" else (False, False),
                tire_clearance_m=(0.005, 0.005) if case == "5mm" else (0.01, 0.01),
                COM_velocity_m_s=(0, 0, -0.1 if case == "downward" else 0.4),
                external_takeoff_assist=case == "assist",
            )
        )
        assert not out["actual_takeoff"] and not out["course_success"]
        assert out["terms"]["takeoff"] == 0
    if case == "19ms":
        r.step(packet(phase=2))  # Contact interrupts the dwell.
        assert r.off_s == 0


@pytest.mark.parametrize("tick", [39, 60, 299, 399])
@pytest.mark.parametrize(
    "fault",
    [dict(fault=True), dict(leg_length_m=(0.373, 0.28)), dict(spring_length_m=(0.22, 0.18))],
)
def test_fault_wins_over_takeoff_apex_landing_and_completion_in_the_same_tick(tick, fault):
    r = scorer(2)
    for t in range(tick):
        r.step(jump_packet(t))
    p = replace(jump_packet(tick), **fault)
    out = r.step(p)
    assert out["terminated"] and not out["course_success"] and not out["active_jump_success"]
    assert out["terms"]["failure"] == -40
    for name in ("takeoff", "air_rise_progress", "apex", "landing_stable", "own_completion"):
        assert out["terms"][name] == 0
    if tick == 39:
        assert not r.has_taken_off
    if tick == 60:
        assert not r.apex_paid
    if tick == 299:
        assert not r.landing_paid
    if tick == 399:
        assert not r.handover_paid


def test_squat_and_compression_cannot_bank_reward_and_timeout_remains_negative():
    r = scorer(0)
    discounted_potential = total = 0.0
    for tick in range(3000):
        out = r.step(
            packet(
                phase=1 if tick % 60 < 20 else 2,
                leg_length_m=(0.145, 0.145) if tick % 40 < 20 else (0.36, 0.36),
                COM_velocity_m_s=(0, 0, 0.5 if tick % 40 < 20 else 0),
            )
        )
        total += out["total"]
        discounted_potential += math.exp(-0.2 * 0.001 * tick) * out["terms"]["velocity_potential"]
        for name in ("crouch_progress", "push_progress", "extension_progress", "takeoff"):
            assert out["terms"][name] == 0
    assert out["terminated"] and r.timeout_paid and not out["course_success"]
    assert abs(discounted_potential) < 1e-10 and total < -24


def test_tiny_hop_landing_cannot_earn_completion_and_partial_failure_is_negative():
    r = scorer()
    tiny = episode(r, height=0.001)
    assert r.has_taken_off and r.landing_paid and not r.handover_paid
    assert not tiny[-1]["course_success"]
    assert sum(x["terms"]["own_completion"] for x in tiny) == 0
    assert sum(x["total"] for x in tiny) < 0
    failed = episode(scorer(), fault_tick=399)
    assert sum(x["terms"]["takeoff"] for x in failed) == 1.5  # Earlier true event is retained.
    assert sum(x["total"] for x in failed) < 0
    assert sum(x["total"] for x in episode(scorer())) > 45


def test_repeated_hops_cannot_repay_events_before_terminal():
    r = scorer()
    totals = {k: 0 for k in ("takeoff", "air_rise_progress", "apex", "landing_stable")}
    for tick in range(600):
        p = replace(jump_packet(tick % 300), motion_confirmed=False, fault=tick == 599)
        out = r.step(p)
        for k in totals:
            totals[k] += out["terms"][k]
    assert totals == dict(takeoff=1.5, air_rise_progress=1, apex=0.5, landing_stable=2.5)
    assert out["terminated"] and not out["course_success"]


@pytest.mark.parametrize("objective", [1, 2])
@pytest.mark.parametrize("failure", [False, True])
def test_delaying_incomplete_jump_to_horizon_has_negative_actual_ppo_discounted_return(
    objective, failure
):
    r = scorer(objective)
    discounted = 0.0
    for tick in range(6000):
        p = replace(
            jump_packet(tick, height=0.001 if objective == 2 else 0.04),
            tilt_rad=(0.1, 0) if objective == 1 and tick >= 100 else (0, 0),
            motion_confirmed=False,
            fault=failure and tick == 5999,
        )
        out = r.step(p)
        shaping = out["terms"].get("velocity_potential", 0.0)
        discounted += math.exp(-0.2 * 0.01 * (tick // 10)) * policy_substep_reward(
            out["total"], shaping, index=tick % 10, dt_s=0.001, discount_rate_s=0.2
        )
        if out["terminated"]:
            break
    assert r.has_taken_off and out["terminated"] and not out["course_success"]
    assert tick == 5999 and discounted < 0


def test_mixed_goals_device_matches_scalar_memory_critic_mask_and_partial_reset():
    rows = [scorer(i % 3) for i in range(9)]
    exported = copy.deepcopy(rows)
    dev = DeviceReward(exported, "cpu")
    enabled = torch.ones(len(rows), dtype=torch.bool)
    for tick in range(500):
        packets = [
            replace(
                jump_packet(tick, height=0.001 if i == 5 else 0.04),
                fault=(i == 6 and tick == 39) or (i == 7 and tick == 299),
                external_takeoff_assist=i == 8,
            )
            for i in range(len(rows))
        ]
        p = {
            f.name: torch.as_tensor(np.array([getattr(x, f.name) for x in packets]))
            for f in fields(packets[0])
        }
        out = dev.step(p, enabled)
        critic = dev.critic_values()
        done = []
        for i in range(len(rows)):
            if not enabled[i]:
                assert out["total"][i] == 0 and not out["terminated"][i]
                continue
            ref = rows[i].step(packets[i])
            assert float(out["total"][i]) == pytest.approx(ref["total"], abs=2e-12)
            for k in ("terminated", "course_success", "active_jump_success"):
                assert bool(out[k][i]) == ref[k]
            for name in MEMORY:
                assert float(dev.state[name][i]) == pytest.approx(
                    float(getattr(rows[i], name)), abs=2e-12
                )
            for name in BEST:
                assert float(dev.best[name][i]) == pytest.approx(rows[i].best[name], abs=2e-12)
            for name, (value, _, _) in rows[i].critic_fields().items():
                np.testing.assert_allclose(critic[name][i].numpy(), value, atol=2e-12, rtol=0)
            if ref["terminated"]:
                done.append(i)
        enabled[done] = False
    dev.export(list(range(len(rows))), exported)
    assert [x.state_dict() for x in exported] == [x.state_dict() for x in rows]
    for r in rows:
        assert UnifiedReward.restore(r.state_dict()).state_dict() == r.state_dict()
    before = {k: v.clone() for k, v in dev.state.items()}
    rows[0] = scorer(1)
    dev.reset([0], rows)
    assert dev.goals["objective"][0] == 1
    assert all(torch.equal(v[1:], dev.state[k][1:]) for k, v in before.items())


def test_mixed_objective_device_reward_captures_a_full_graph_without_gpu():
    rows = [scorer(i) for i in range(3)]
    for tick in range(39):
        for row in rows:
            row.step(jump_packet(tick))
    reference, device = DeviceReward(copy.deepcopy(rows), "cpu"), DeviceReward(rows, "cpu")
    packet39 = jump_packet(39)
    values = {
        f.name: torch.as_tensor(np.array([getattr(packet39, f.name)] * 3)) for f in fields(packet39)
    }
    enabled = torch.ones(3, dtype=torch.bool)
    # Eager backend checks Dynamo fullgraph capture, not CUDA kernel execution.
    compiled = torch.compile(device.step, fullgraph=True, backend="eager")
    actual, expected = compiled(values, enabled), reference.step(values, enabled)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    for name in MEMORY:
        assert torch.equal(device.state[name], reference.state[name])


def test_course_progression_restore_and_publication_keep_skill_success_separate():
    cfg = CourseConfig(
        window_episodes=2, promote_windows=2, demote_windows=2, require_landing_rehearsal=False
    )
    levels = [SkillLevel(str(i), 1, 0.02, 0.005, 0.18, i) for i in range(3)]
    course = AdaptiveCourse(seed=4, plant_id="p", DR_id="nominal", config=cfg, levels=levels)
    stale = course.next_episode()
    completed = []
    for _ in range(4):
        request = course.next_episode()
        assert request["objective"] == 0 and request["kind"] == "ground_active_jump"
        course.record(request, jump_control_s=0.4, success=True, fault=False)
        completed.append(
            dict(
                request=request,
                success=False,
                course_success=True,
                objective=0,
                fault=False,
                jump_control_s=0.4,
                balance_seconds=0.3,
                balance_good_seconds=0.3,
            )
        )
    assert course.level == 1
    assert course.record(stale, jump_control_s=1, success=True, fault=False)["stale_generation"]
    assert not any(course.buffers.values())
    with pytest.raises(ValueError):
        course.record(stale, jump_control_s=1, success=True, fault=False)
    course = AdaptiveCourse.restore(course.state_dict())
    assert isinstance(course.levels[1], SkillLevel) and course.next_episode()["objective"] == 1
    for _ in range(4):
        course.record(course.next_episode(), jump_control_s=1, success=False, fault=False)
    assert course.level == 0
    index = dict(candidates_by_objective={"active_jump": {}, "landing_only": {}, "balance": {}})
    actual = consider(index, dict(rows=completed, selection=vars(cfg)), dict(update=100))
    assert not actual["candidates_by_objective"]["active_jump"]
    stats = episode_metrics(completed)["ground_active_jump"]
    assert stats["course_successes"] == 4 and stats["successes"] == 0
    assert stats["objectives"] == {"0": 4, "1": 0, "2": 0}
