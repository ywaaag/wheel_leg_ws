"""Device reward is equivalent to scalar ABI4 task through event boundaries."""

import copy
from dataclasses import fields

import numpy as np
import torch
from test_jump_unified import packet

from native_stand.tasks.jump.dense_reward import TaskSpec
from native_stand.tasks.jump.device_reward import BEST, MEMORY, DeviceReward
from native_stand.tasks.jump.unified_reward import UnifiedReward

FROZEN_TEST_SCOPE = "PORTABLE"


def test_device_reward_mixed_provenance_events_timeout_and_mask():
    n = 8
    initial = [
        packet(
            dt_s=0.0,
            phase=9,
            contact=(i % 2 == 0, i % 2 == 0),
            tire_clearance_m=(0.0, 0.0) if i % 2 == 0 else (0.04, 0.04),
        )
        for i in range(n)
    ]
    rows = [
        UnifiedReward.start(
            TaskSpec("ground_active_jump" if i % 2 == 0 else "landing_only", 0.35, 10), p
        )
        for i, p in enumerate(initial)
    ]
    actual = copy.deepcopy(rows)
    device = DeviceReward(actual, "cpu")
    rng = np.random.default_rng(101)
    for tick in range(420):
        packets = []
        for i in range(n):
            phase = (
                9
                if tick < 10
                else 1
                if tick < 30
                else 2
                if tick < 60
                else 3
                if tick < 120
                else 4
                if tick < 130
                else 5
                if tick < 370
                else 9
            )
            airborne = 60 <= tick < 120
            packets.append(
                packet(
                    dt_s=0.001,
                    phase=phase,
                    owner=2,
                    contact=(not airborne, not airborne),
                    tire_clearance_m=(0.07, 0.07) if airborne else (0.0, 0.0),
                    COM_height_m=0.34
                    + (0.08 * np.sin((tick - 60) * np.pi / 60) if airborne else 0),
                    COM_velocity_m_s=(0.0, 0.0, (1.0 if tick < 90 else -0.4) if airborne else 0.0),
                    leg_length_m=(0.145, 0.145)
                    if phase == 1
                    else (0.18, 0.18)
                    if airborne
                    else (0.28, 0.28),
                    actual_motor_Nm=tuple(rng.uniform(-10, 10, 6)),
                    actual_dq_rad_s=tuple(rng.uniform(-2, 2, 6)),
                    adopted_target=tuple(rng.uniform(-0.1, 0.1, 6)),
                    motion_confirmed=tick > 330,
                    fault=i == 6 and tick == 230,
                )
            )
        enabled = torch.tensor([not (i == 7 and tick > 160) for i in range(n)])
        p = {
            f.name: torch.tensor(
                np.array([getattr(x, f.name) for x in packets]), dtype=torch.float64
            )
            for f in fields(packets[0])
        }
        out = device.step(p, enabled)
        for i in range(n):
            if not enabled[i]:
                continue
            ref = rows[i].step(packets[i])
            np.testing.assert_allclose(float(out["total"][i]), ref["total"], rtol=0, atol=2e-12)
            assert bool(out["terminated"][i]) == ref["terminated"]
            assert bool(out["active_jump_success"][i]) == ref["active_jump_success"]
            assert bool(out["landing_only_success"][i]) == ref["landing_only_success"]
            for k in MEMORY:
                np.testing.assert_allclose(
                    float(device.state[k][i]), getattr(rows[i], k), rtol=0, atol=2e-12, err_msg=k
                )
            for k in BEST:
                np.testing.assert_allclose(
                    float(device.best[k][i]), rows[i].best[k], rtol=0, atol=2e-12, err_msg=k
                )
    device.export(list(range(n)), actual)
    for i in range(n):
        for k in MEMORY:
            np.testing.assert_allclose(
                getattr(actual[i], k), getattr(rows[i], k), rtol=0, atol=2e-12
            )
    device.reset([0, 3], rows)
    assert float(device.state["elapsed_s"][1]) == rows[1].elapsed_s
