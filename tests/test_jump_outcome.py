"""Real controller and scalar/device outcome rewards; no physical integration."""

import copy
import ctypes as C
import math
from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from test_jump_unified import feedback, packet

from native_stand.simulation.jump_build import build
from native_stand.simulation.jump_shared_c import Config
from native_stand.simulation.jump_unified import UnifiedJump
from native_stand.tasks.jump import unified
from native_stand.tasks.jump.adaptive_course import AdaptiveCourse, CourseConfig, Level
from native_stand.tasks.jump.dense_reward import TaskSpec, policy_substep_reward
from native_stand.tasks.jump.device_reward import BEST, MEMORY, DeviceReward
from native_stand.tasks.jump.outcome import CONFIG_ID, reward_config
from native_stand.tasks.jump.unified_reward import UnifiedReward

FROZEN_TEST_SCOPE = "PORTABLE"


def controller_config():
    return Config(
        4,
        unified.INPUT_ID,
        unified.OUTPUT_ID,
        CONFIG_ID,
        1,
        1,
        1,
        unified.INPUT_ID,
        unified.OUTPUT_ID,
    )


def test_pending_request_enters_push_after_old_wait_timeout_without_crouch(tmp_path):
    binding = build(tmp_path / "c.so")
    c = UnifiedJump(binding["path"], 1, library_sha=binding["sha256"], config=controller_config())
    entered = None
    for tick in range(0, 1000, 5):
        i = feedback(tick)
        i.key_down = int(tick >= 300)
        # Stable posture only becomes available after the old 500ms WAIT window.
        i.gyro[0] = 0.4 if tick < 850 else 0
        assert c.step([i]) >= 0
        if c.states[0].phase == 2:
            entered = tick
            break
    assert entered is not None and entered >= 950
    assert c.states[0].request_id == c.states[0].epoch == 1
    assert c.states[0].status == 2
    assert list(i.leg_length) == pytest.approx([0.28, 0.28])
    c.load_state_dict(c.state_dict())
    # Support and posture entry guards still apply, a held request cannot bypass them.
    c.lib.RLJumpInit(C.byref(c.states[0]), 0)
    for tick in range(0, 900, 5):
        i = feedback(tick)
        i.key_down = 1
        i.gyro[0] = 0.4
        c.step([i])
    assert c.states[0].phase == 9 and c.states[0].epoch == 0


def reward(kind="ground_active_jump", **config_changes):
    return UnifiedReward.start(
        TaskSpec(kind, 6, 10, target_rise_m=0.02, target_clearance_m=0.005),
        packet(dt_s=0, phase=9, leg_length_m=(0.28, 0.28)),
        replace(reward_config(), **config_changes),
    )


def test_squat_hold_and_wait_have_no_positive_reward_and_end_by_two_seconds():
    for phase, length in [(9, 0.28), (1, 0.145), (2, 0.145)]:
        r = reward()
        total = 0
        for tick in range(2000):
            out = r.step(packet(dt_s=0.001, phase=phase, leg_length_m=(length, length)))
            total += out["total"]
            assert out["terms"]["crouch_progress"] == 0
            assert out["terms"]["extension_progress"] == 0
            if tick < 1999:
                assert not out["terminated"]
        assert out["terminated"] and r.timeout_paid and total < -24
        assert not out["active_jump_success"] and not r.has_taken_off


def test_velocity_potential_cannot_be_farmed_by_repeated_compression_or_hold():
    r = reward()
    discounted = 0.0
    discount = 1.0
    for tick in range(2000):
        vz = 0.45 if (tick // 20) % 2 else 0.0
        out = r.step(packet(dt_s=0.001, phase=2, COM_velocity_m_s=(0, 0, vz)))
        discounted += discount * out["terms"]["velocity_potential"]
        discount *= math.exp(-0.2 * 0.001)
    assert abs(discounted) < 1e-11
    assert r.timeout_paid and not r.has_taken_off


def completed_jump(r, *, height=0.04, fault_at_end=False):
    result = []
    for t in range(400):
        phase = 2 if t < 20 else 3 if t < 100 else 5 if t < 399 else 9
        air = 20 <= t < 100
        z = 0.34 + height * math.sin((t - 20) * math.pi / 80) if air else 0.34
        p = packet(
            dt_s=0.001,
            phase=phase,
            COM_height_m=z,
            COM_velocity_m_s=(0, 0, (1 if t < 60 else -0.2) if air else 0),
            tire_clearance_m=(0.02, 0.02) if air else (0, 0),
            contact=(not air, not air),
            leg_length_m=(0.28, 0.28),
            motion_confirmed=t == 399,
            fault=fault_at_end and t == 399,
        )
        out = r.step(p)
        result.append((p, out))
    return result


def test_complete_jump_beats_process_and_tiny_hop_cannot_claim_completion():
    r = reward()
    seq = completed_jump(r)
    assert r.has_taken_off and not r.crouch_legal
    assert seq[-1][1]["active_jump_success"]
    assert sum(o["total"] for _, o in seq) > 35
    tiny = reward()
    seq = completed_jump(tiny, height=0.001)
    assert not tiny.landing_paid and not seq[-1][1]["active_jump_success"]
    failed = reward()
    seq = completed_jump(failed, fault_at_end=True)
    assert not seq[-1][1]["active_jump_success"]
    assert seq[-1][1]["terms"]["own_completion"] == 0
    assert sum(o["total"] for _, o in seq) < 0


@pytest.mark.parametrize("complete", [False, True])
def test_policy_discount_cancels_all_process_credit_even_when_failure_is_delayed(complete):
    reference = reward()
    trace = completed_jump(reference)
    r = reward()
    shaping_return = total_return = 0.0
    for t in range(6000):
        p = trace[t][0] if t < len(trace) else packet(dt_s=0.001, phase=5)
        p = replace(p, motion_confirmed=complete and p.motion_confirmed)
        out = r.step(p)
        shape = sum(out["terms"].get(k, 0.0) for k in ["velocity_potential", "progress_potential"])
        factor = math.exp(-0.2 * 0.01 * (t // 10))
        shaping_return += factor * policy_substep_reward(
            shape, shape, index=t % 10, dt_s=0.001, discount_rate_s=0.2
        )
        total_return += factor * policy_substep_reward(
            out["total"], shape, index=t % 10, dt_s=0.001, discount_rate_s=0.2
        )
        if out["terminated"]:
            break
    assert out["terminated"]
    assert abs(shaping_return) < 1e-10
    assert (total_return > 30) if complete else (total_return < 0)
    assert out["active_jump_success"] == complete


def test_gpu_math_matches_scalar_events_fault_deadline_and_partial_reset():
    rows = [reward(), reward(), reward(), reward()]
    dev = DeviceReward(copy.deepcopy(rows), "cpu")
    reference = reward()
    trace = completed_jump(reference)
    for t in range(2000):
        base = trace[t][0] if t < len(trace) else packet(dt_s=0.001, phase=9)
        packets = [
            base,
            replace(base, external_takeoff_assist=True),
            packet(dt_s=0.001, phase=2, COM_velocity_m_s=(0, 0, 0.3 if t % 2 else 0)),
            replace(base, fault=t == 399),
        ]
        p = {
            f.name: torch.as_tensor(
                np.array([getattr(x, f.name) for x in packets]), dtype=torch.float64
            )
            for f in fields(base)
        }
        enabled = torch.ones(4, dtype=torch.bool)
        actual = dev.step(p, enabled)
        for i in range(4):
            out = rows[i].step(packets[i])
            assert float(actual["total"][i]) == pytest.approx(out["total"], abs=2e-12)
            for k in ["terminated", "active_jump_success", "landing_only_success"]:
                assert bool(actual[k][i]) == out[k]
            for k in MEMORY:
                assert float(dev.state[k][i]) == pytest.approx(
                    float(getattr(rows[i], k)), abs=2e-12
                )
            for k in BEST:
                assert float(dev.best[k][i]) == pytest.approx(rows[i].best[k], abs=2e-12)
    before = {k: v.clone() for k, v in dev.state.items()}
    rows[0] = reward()
    dev.reset([0], rows)
    assert all(torch.equal(v[1:], dev.state[k][1:]) for k, v in before.items())
    assert UnifiedReward.restore(rows[0].state_dict()).state_dict() == rows[0].state_dict()


def test_jump_only_curriculum_promotes_only_complete_jump_results():
    levels = (Level("low", 1.0, 0.02, 0.005, 0.18), Level("next", 1.0, 0.05, 0.01, 0.18))
    cfg = CourseConfig(window_episodes=2, promote_windows=2, require_landing_rehearsal=False)
    c = AdaptiveCourse(seed=5, plant_id="same", DR_id="nominal", config=cfg, levels=levels)
    for _ in range(4):
        req = c.next_episode()
        assert req["kind"] == "ground_active_jump"
        c.record(req, jump_control_s=2.0, success=False, fault=False)
    assert c.level == 0
    for _ in range(4):
        req = c.next_episode()
        c.record(req, jump_control_s=2.0, success=True, fault=False)
    assert c.level == 1
    restored = AdaptiveCourse.restore(c.state_dict())
    assert restored.state_dict() == c.state_dict()
