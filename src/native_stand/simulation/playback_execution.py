"""CPU共享C控制＋V4机械/NET；没有物理积分或固件反馈人为延迟。"""

from types import SimpleNamespace

import numpy as np
import torch

from .control_core import ControlCore
from .isaac.sensor_packets import Packets
from .v4_response import Response

LEGS = [0, 1, 3, 4]
WHEELS = [2, 5]
COLUMNS = torch.tensor([[3, 4], [0, 1]])
SIGN = torch.tensor([-1.0, 1.0])


def pair(value):
    return (value[COLUMNS] * SIGN).reshape(2, 2)


def unpair(value):
    out = value.new_zeros(6)
    out[COLUMNS.flatten()] = (value.reshape(2, 2) * SIGN).flatten()
    return out


class PlaybackExecution:
    def __init__(self, core: ControlCore, V4, timing, control, lower, upper):
        self.core = core
        self.timing = dict(timing)
        self.control = control
        if timing["version"] not in (
            "motion_v4_wheel_reported_delay_actor20_dr_v1",
            "motion48_v4_wheel_reported_delay_actor20_dr_v2",
            "recovery_complete_V4_actor20_target5_wheel7_11_v1",
        ):
            raise ValueError("播放器需要明确的V4/观测/运输时序版本")
        self.lower = np.asarray(lower, np.float32)
        self.upper = np.asarray(upper, np.float32)
        rows = [V4["sides"][s] for s in ("L", "R")]
        parameters = torch.tensor(
            [[r["parameters"][k] for k in V4["parameter_names"]] for r in rows]
        )
        if parameters.shape != (2, 13):
            raise ValueError("播放器要求完整V4腿响应")
        self.leg = Response(parameters, [r["effective_observer"] for r in rows], pd=self._pd)
        self.tick = 0
        self.q = torch.zeros(6)
        self.dq = self.q.clone()
        self.requested = np.zeros(6, np.float32)
        self.targets = np.zeros((32, 6), np.float32)
        self.wheels = np.zeros((64, 2), np.float32)
        self.wheel_command = np.zeros(2, np.float32)
        self.raw_PD = np.zeros(6, np.float32)
        self.driver = np.zeros(6, np.float32)
        dr = SimpleNamespace(
            episodes=torch.zeros(1, dtype=torch.long),
            bias=torch.zeros(1, 16),
            scales=torch.zeros(1, 5),
            packet_age=torch.tensor([timing["actor_packet_nominal_ms"]]),
            leg_packet_age=torch.tensor([timing["actor_leg_packet_nominal_ms"]]),
            dq=SimpleNamespace(read=lambda x, t: x),
        )
        packet_clock = SimpleNamespace(
            n=1, device="cpu", ids=torch.arange(1), dr=dr, tick=0, measurement=self.measurement
        )
        self.packet_clock = packet_clock
        self.packets = Packets(packet_clock, noise=lambda e, i, w, s: torch.zeros(len(i), w))
        self.motor = np.zeros(6, np.float32)
        self.NET = np.zeros(6, np.float32)

    def measurement(self):
        q = unpair(self.leg.heldq)
        dq = unpair(self.leg.heldv)
        q[WHEELS] = self.q[WHEELS]
        dq[WHEELS] = self.dq[WHEELS]
        return q[None], dq[None]

    def _pd(self, q, dq, target):
        physical_q = unpair(q)
        physical_dq = unpair(dq)
        physical_dq[WHEELS] = self.dq[WHEELS]
        gains = (self.control.leg_kp, self.control.leg_kd, self.control.wheel_kv)
        physical_target = unpair(target).numpy()
        physical_target[WHEELS] = self.core.adopted[WHEELS]
        raw, limited = self.core.pd(
            physical_q[LEGS].numpy(),
            physical_dq.numpy(),
            target=physical_target,
            gains=gains,
            caps=self.control.effort_limits_nm,
        )
        self.raw_PD = raw
        caps = np.asarray(self.control.effort_limits_nm)
        self.driver = np.clip(
            limited,
            -caps,
            caps,
        ).astype(np.float32)
        self.wheel_command = self.driver[WHEELS].copy()
        return pair(torch.from_numpy(raw))

    def reset(self, q, dq):
        self.tick = 0
        self.packet_clock.tick = 0
        self.q = torch.as_tensor(q, dtype=torch.float32).clone()
        self.dq = torch.as_tensor(dq, dtype=torch.float32).clone()
        initial = self.q.numpy().copy()
        initial[WHEELS] = 0
        self.core.adopt(initial, self.lower, self.upper)
        self.requested = self.core.adopted.copy()
        self.targets[:] = self.core.adopted
        self.leg.reset(pair(self.q), pair(self.dq), pair(torch.from_numpy(self.core.adopted)))
        self.wheels[:] = self.wheel_command
        self.packets.reset(torch.tensor([0]))
        self.motor.fill(0)
        self.NET.fill(0)

    def submit(self, physical):
        self.requested = np.asarray(physical, np.float32).copy()

    def observation(self, gyro, gravity, command, *, skill=0):
        q, dq = self.measurement()
        frame = np.r_[
            np.asarray(gyro) * 0.25, gravity, q[0, LEGS].numpy() / 1.5, dq[0].numpy() * 0.05
        ].astype(np.float32)
        self.packet_clock.tick = self.tick
        sensed = self.packets.read(torch.from_numpy(frame[None]), self.tick)[0].numpy()
        return self.core.encode(
            sensed[:3] / 0.25,
            sensed[3:6],
            sensed[6:10] * 1.5,
            sensed[10:16] / 0.05,
            command,
            skill=skill,
        )

    def step(self, q, dq):
        self.q = torch.as_tensor(q, dtype=torch.float32)
        self.dq = torch.as_tensor(dq, dtype=torch.float32)
        tick = self.tick
        self.targets[tick % 32] = self.requested
        if tick % 5 == 0:
            self.core.adopt(
                self.targets[(tick - self.timing["target_nominal_ms"]) % 32], self.lower, self.upper
            )
        motor, NET = self.leg.step(
            pair(self.q), pair(self.dq), pair(torch.from_numpy(self.core.adopted))
        )
        self.motor = unpair(motor).numpy().copy()
        self.NET = unpair(NET).numpy().copy()
        self.wheels[tick % 64] = self.wheel_command
        delay = np.asarray(self.timing["wheel_transport_nominal_ms"])
        if any(self.timing["wheel_tau_ms"]):
            raise ValueError("该显式时序版本不允许重复轮tau")
        self.motor[WHEELS] = self.wheels[(tick - delay) % 64, np.arange(2)]
        self.NET[WHEELS] = 0
        self.tick += 1
        return self.motor.copy(), self.NET.copy()
