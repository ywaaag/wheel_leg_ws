"""Portable adapter checks; source-specific C oracle is a separate bounded run."""

from types import SimpleNamespace

import numpy as np
import torch

from native_stand.simulation.native_recovery_firmware import execution_factory

FROZEN_TEST_SCOPE = "PORTABLE"


class Base:
    def __init__(self):
        self.n, self.dtype, self.tick = 1, torch.float32, 0
        self.requested = torch.zeros(1, 6)
        self.adopted = self.requested.clone()
        self.bias = torch.zeros(1, 4)
        self.motor = self.requested.clone()
        self.motors = torch.zeros(128, 1, 6)
        self.packet_age = torch.zeros(1, dtype=torch.long)
        self.target_delay = self.packet_age.clone()
        self.wheel_delay = torch.zeros(1, 2, dtype=torch.long)
        self.effective_packet_age = self.packet_age.clone()
        self.leg = SimpleNamespace(
            delay=torch.zeros(2, dtype=torch.long), ids=torch.arange(2), alpha=torch.ones(2, 2)
        )

    def reset(self, ids, q, dq):
        self.adopted[ids] = q

    def report(self, f, q, dq):
        pass

    def submit(self, raw):
        self.requested.copy_(raw.clamp(-0.1, 0.1))

    def pair(self, v):
        return v[:, [[3, 4], [0, 1]]].reshape(-1, 2)

    def unpair(self, v):
        out = torch.zeros(1, 6)
        out[:, [3, 4, 0, 1]] = v.reshape(1, 4)
        return out

    def step(self, q, dq):
        self.tick += 1
        return torch.zeros(1, 6), torch.ones(1, 6) * 7


class Firmware:
    def observation(self, frame, previous):
        return np.r_[frame, previous, np.zeros(9, np.float32)].astype(np.float32)

    def torque(self, frame, target, current_gains):
        physical = target.copy() * (2 if current_gains else 3)
        physical[[2, 5]] = np.clip(physical[[2, 5]], -2.446948, 2.446948)
        return target.copy(), target.copy(), physical


def test_original_previous_zero_then_decoded_target_and_no_range_projection():
    cls = execution_factory(Base, Firmware(), dict(actor_age_ms=0, gains="native"))
    ex = cls()
    q = torch.ones(1, 6) * 0.2
    ex.reset(torch.tensor([0]), q, q * 0)
    f = torch.zeros(1, 16)
    assert not ex.report(f, q, q)[0, 16:22].any()
    target = torch.tensor([[0.5, -0.4, 80, -0.3, 0.2, -90]])
    ex.submit(target)
    torch.testing.assert_close(ex.requested, target)
    torch.testing.assert_close(ex.adopted, target)
    torch.testing.assert_close(ex.report(f, q, q)[0, 16:22], target[0])
    assert ex.target_delay.item() == 0 and not ex.wheel_delay.any()


def test_pd_is_held_for_policy_interval_while_net_stays_from_v4():
    cls = execution_factory(Base, Firmware(), dict(actor_age_ms=0, gains="native"))
    ex = cls()
    q = torch.zeros(1, 6)
    ex.submit(torch.tensor([[0.5, -0.4, 80, -0.3, 0.2, -90]]))
    first, net = ex.step(q, q)
    first = first.clone()
    for _ in range(9):
        motor, _ = ex.step(q + 1, q + 100)
        torch.testing.assert_close(motor, first)
    torch.testing.assert_close(net, torch.ones(1, 6) * 7)
    assert abs(first[0, 2].item() - 2.446948) < 1e-6


def test_declared_age_affects_sensing_without_delaying_virtual_previous():
    cls = execution_factory(Base, Firmware(), dict(actor_age_ms=20, gains="native"))
    ex = cls()
    q, f = torch.zeros(1, 6), torch.zeros(1, 16)
    ex.report(f, q, q)
    ex.tick = 10
    ex.report(f + 1, q, q)
    ex.submit(torch.ones(1, 6))
    ex.tick = 20
    x = ex.report(f + 2, q, q)
    assert x[0, 0] == 0 and x[0, 16] == 1
    assert ex.effective_packet_age.item() == 20
