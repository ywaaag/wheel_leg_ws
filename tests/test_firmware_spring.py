"""Portable integration tests for compensation scheduling, mapping and bounds."""

import ctypes

import numpy as np
import torch

from native_stand.simulation.firmware_spring import FirmwareSpring

FROZEN_TEST_SCOPE = "PORTABLE"


class FakeFirmware:
    @staticmethod
    def RLMotion31MapPolicyTargets(q, angles, wheels):
        angles[:] = [1.0, 0.5, 1.0, 0.5]
        wheels[:] = 0
        return 0

    @staticmethod
    def getPhi(*args):
        ctypes.cast(args[-2], ctypes.POINTER(ctypes.c_float))[0] = 0.3

    @staticmethod
    def RLMotion31MapPolicyLegTorqueToMotorOrder(t, out):
        out[:] = [-t[4], -t[3], t[1], -t[0]]
        return 0

    @staticmethod
    def RLControlCompensateSpring(t, length, spread, out):
        out[:] = np.clip(t + [-10, 10, 10, -10], -40, 40)
        return 0


def test_only_compensation_is_held_not_v4_motor_response():
    f = FirmwareSpring.__new__(FirmwareSpring)
    f.lib = FakeFirmware()
    f.held = None
    f.delta = []
    f.saturation = 0
    q, dq = torch.zeros(1, 6), torch.zeros(1, 6)
    bias = torch.zeros(1, 4)
    baseline = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    first = f(q, dq, baseline, 0, bias)
    second = f(q, dq, baseline + 1, 1, bias)
    # V4 can evolve at1kHz between200Hz PD updates: retain that response.
    torch.testing.assert_close(second - first, torch.ones_like(first))
    assert second[0, 2] == baseline[0, 2] + 1
    assert second[0, 5] == baseline[0, 5] + 1
    saturated = f(q, dq, baseline * 100, 5, bias)
    assert saturated[:, [0, 1, 3, 4]].abs().max() <= 40
