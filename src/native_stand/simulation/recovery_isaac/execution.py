"""Batched causal V4/queue kernel, engine independent and CPU-testable.

Explicit Response dependency is the SHA-bound immutable full V4 tensor kernel.
There is no Actor, optimizer, scene import or physical state write here.
"""

from __future__ import annotations

import copy

import torch

VERSION = "recovery_Isaac_parallel_execution_v1"
LEGS = [0, 1, 3, 4]
WHEELS = [2, 5]
SCALE = [1.5, 1.5, 100.0, 1.5, 1.5, 100.0]
CAPS = [40.0, 40.0, 2.44694786377709, 40.0, 40.0, 2.44694786377709]


def nominal_timing():
    """该执行版本的名义时序；实机反馈年龄与仿真注入保持区分。"""
    return dict(
        version="recovery_complete_V4_actor20_target5_wheel7_11_v1",
        actor_packet_nominal_ms=20,
        actor_leg_packet_nominal_ms=20,
        target_nominal_ms=5,
        wheel_transport_nominal_ms=[7, 11],
        wheel_tau_ms=[0.0, 0.0],
        leg_response="complete V4; no added software compensation",
        input_scales=[1.5, 100.0],
    )


class Execution:
    def __init__(
        self, contract, response_class, n, *, device="cpu", dtype=torch.float32, effort_caps=CAPS
    ):
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype
        self.tick = 0
        self.ids = torch.arange(n, device=device)
        self.columns = torch.tensor([[3, 4], [0, 1]], device=device)
        self.sign = torch.tensor([-1.0, 1.0], device=device, dtype=dtype)
        rows = [contract["sides"][s] for s in ["L", "R"]]
        x = torch.tensor(
            [[r["parameters"][k] for k in contract["parameter_names"]] for r in rows],
            device=device,
            dtype=dtype,
        ).repeat(n, 1)
        self.leg = response_class(x, [r["effective_observer"] for r in rows] * n)
        self.leg.reset(x.new_zeros(2 * n, 2), x.new_zeros(2 * n, 2), x.new_zeros(2 * n, 2))
        if tuple(effort_caps) not in (tuple(CAPS), (40.0, 40.0, 4.8, 40.0, 40.0, 4.8)):
            raise ValueError("未知Recovery电机限矩合同")
        self.caps = x.new_tensor(effort_caps).expand(n, -1).clone()
        self.lower = x.new_tensor([-1.5, -1.5, -100, -1.5, -1.5, -100]).expand(n, -1).clone()
        self.upper = -self.lower
        self.raw = x.new_zeros(n, 6)
        self.range_request = self.raw.clone()
        self.requested = self.raw.clone()
        self.adopted = self.raw.clone()
        self.motor = self.raw.clone()
        self.NET = self.raw.clone()
        self.wheel_PD = x.new_zeros(n, 2)
        self.wheel_raw = self.wheel_PD.clone()
        self.request_tick = torch.zeros(n, device=device, dtype=torch.long)
        self.adopted_origin = self.request_tick.clone()
        self.targets = x.new_zeros(128, n, 6)
        self.origins = torch.zeros(128, n, device=device, dtype=torch.long)
        self.wheel_queue = x.new_zeros(128, n, 2)
        self.motors = self.targets.clone()
        self.NETs = self.targets.clone()
        timing = nominal_timing()
        self.target_delay = torch.full(
            (n,), timing["target_nominal_ms"], device=device, dtype=torch.long
        )
        self.wheel_delay = (
            torch.tensor(timing["wheel_transport_nominal_ms"], device=device).expand(n, -1).clone()
        )
        self.packet_age = torch.full(
            (n,), timing["actor_packet_nominal_ms"], device=device, dtype=torch.long
        )
        self.bias = x.new_zeros(n, 4)
        self.theta = x.new_zeros(2 * n, 2)
        self.omega = self.theta.clone()
        self.last_q = self.raw.clone()
        self.last_dq = self.raw.clone()
        self.packet = x.new_zeros(16, n, 16)
        self.qhistory = x.new_zeros(16, n, 6)
        self.dqhistory = self.qhistory.clone()
        self.target_policy_history = self.qhistory.clone()
        self.motor_policy_history = self.qhistory.clone()
        self.effective_packet_age = torch.zeros(n, device=device, dtype=torch.long)
        self.birth = torch.zeros(n, device=device, dtype=torch.long)
        self.observed_tick = torch.full_like(self.birth, -1)
        self.hp_low = x.new_zeros(n, 4)
        self.hp_power = x.new_zeros(n, 4)
        self.hp_count = torch.zeros(n, device=device, dtype=torch.long)

    def pair(self, v):
        return (v[:, self.columns] * self.sign).reshape(-1, 2)

    def unpair(self, v):
        out = v.new_zeros(self.n, 6)
        out[:, self.columns.flatten()] = (v.reshape(self.n, 2, 2) * self.sign).reshape(self.n, 4)
        return out

    def reset(self, ids, q, dq):
        ids = ids.to(device=self.device, dtype=torch.long)
        q = q.to(self.device, self.dtype)
        dq = dq.to(self.device, self.dtype)
        if q.shape != (len(ids), 6) or dq.shape != q.shape:
            raise ValueError("Measured reset rows required")
        fullq = self.last_q.clone()
        fullv = self.last_dq.clone()
        fullq[ids] = q
        fullv[ids] = dq
        pairs = (ids[:, None] * 2 + torch.arange(2, device=self.device)).flatten()
        target = q.clone()
        target[:, WHEELS] = 0
        allt = self.adopted.clone()
        allt[ids] = target
        self.leg.reset(
            self.pair(fullq)[pairs], self.pair(fullv)[pairs], self.pair(allt)[pairs], ids=pairs
        )
        self.theta[pairs] = self.pair(fullq)[pairs]
        self.omega[pairs] = self.pair(fullv)[pairs]
        for name in ["raw", "range_request", "requested", "adopted"]:
            getattr(self, name)[ids] = target
        for name in ["targets", "target_policy_history"]:
            getattr(self, name)[:, ids] = target
        self.last_q[ids] = q
        self.last_dq[ids] = dq
        self.qhistory[:, ids] = q
        self.dqhistory[:, ids] = dq
        self.wheel_queue[:, ids] = 0
        self.wheel_PD[ids] = 0
        self.wheel_raw[ids] = 0
        self.motor[ids] = 0
        self.NET[ids] = 0
        self.motors[:, ids] = 0
        self.NETs[:, ids] = 0
        self.motor_policy_history[:, ids] = 0
        self.request_tick[ids] = self.tick
        self.origins[:, ids] = self.tick
        self.adopted_origin[ids] = self.tick
        self.effective_packet_age[ids] = 0
        self.birth[ids] = self.tick
        self.observed_tick[ids] = -1
        self.hp_low[ids] = 0
        self.hp_power[ids] = 0
        self.hp_count[ids] = 0

    def submit(self, raw):
        if raw.shape != (self.n, 6) or not bool(torch.isfinite(raw).all()):
            raise ValueError("Finite raw physical N×6 required")
        self.raw.copy_(raw)
        lo = raw.new_tensor([-1.5, -1.5, -100, -1.5, -1.5, -100])
        hi = -lo
        self.range_request.copy_(torch.maximum(lo, torch.minimum(hi, raw)))
        self.requested.copy_(
            torch.maximum(self.lower, torch.minimum(self.upper, self.range_request))
        )
        self.request_tick.fill_(self.tick)
        # This submits a request. No adoption, queue clear, PD or motor update.
        return self.requested

    def step(self, q, dq):
        t = self.tick
        if self.leg.tick != t:
            raise RuntimeError("Native/target physical clocks differ")
        if q.shape != (self.n, 6) or not bool(torch.isfinite(torch.cat((q, dq), -1)).all()):
            raise ValueError("Finite physical q/dq")
        self.targets[t % 128] = self.requested
        self.origins[t % 128] = self.request_tick
        if t % 5 == 0:
            index = (t - self.target_delay) % 128
            self.adopted.copy_(self.targets[index, self.ids])
            self.adopted_origin.copy_(self.origins[index, self.ids])
        bias = q.new_zeros(self.n, 6)
        bias[:, LEGS] = self.bias
        # Unknown residual q_report=q_true+epsilon. Report targets/previous
        # stay in that coordinate; native PD sees shifted target, not truth.
        # Add bias AFTER native sensor quantisation, exactly like CPU backend.
        motor, net = self.leg.step(self.pair(q), self.pair(dq), self.pair(self.adopted - bias))
        self.motor.copy_(self.unpair(motor))
        self.NET.copy_(self.unpair(net))
        if t % 5 == 0:
            self.wheel_raw.copy_(0.2 * (self.adopted[:, WHEELS] - dq[:, WHEELS]))
            self.wheel_PD.copy_(self.wheel_raw.clamp(-self.caps[:, WHEELS], self.caps[:, WHEELS]))
        self.wheel_queue[t % 128] = self.wheel_PD
        self.motor[:, WHEELS] = self.wheel_queue[
            (t - self.wheel_delay) % 128,
            self.ids[:, None],
            torch.arange(2, device=self.device)[None],
        ]
        self.motors[t % 128] = self.motor
        self.NETs[t % 128] = self.NET
        self.theta.copy_(self.pair(q))
        self.omega.copy_(self.pair(dq))
        self.last_q.copy_(q)
        self.last_dq.copy_(dq)
        alpha = 1 - torch.exp(q.new_tensor(-2 * torch.pi * 10 * 0.001))
        self.hp_low += alpha * (dq[:, LEGS] - self.hp_low)
        self.hp_power += (dq[:, LEGS] - self.hp_low).square()
        self.hp_count += 1
        self.tick += 1
        if bool((abs(self.NET[:, LEGS]) > 200).any()):
            raise RuntimeError("Original V4 hypothesis force envelope")
        return self.motor, self.NET

    def physical_HP10(self):
        value = (self.hp_power / self.hp_count.clamp_min(1)[:, None]).sqrt()
        self.hp_power.zero_()
        self.hp_count.zero_()
        return value

    def report(self, true_frame, q, dq):
        fresh = self.observed_tick != self.tick
        new = self.observed_tick < 0
        slot = (self.tick // 10) % 16
        current = true_frame.clone()
        current[:, 6:10] = (self.unpair(self.leg.heldq)[:, LEGS] + self.bias) / 1.5
        current[:, [10, 11, 13, 14]] = self.unpair(self.leg.heldv)[:, LEGS] * 0.05
        for name, value in [
            ("packet", current),
            ("qhistory", q),
            ("dqhistory", dq),
            ("target_policy_history", self.adopted),
            ("motor_policy_history", self.motor),
        ]:
            array = getattr(self, name)
            array[:, new] = value[new]
            array[slot, fresh] = value[fresh]
        self.observed_tick[fresh] = self.tick
        age = torch.minimum(self.packet_age, (self.tick - self.birth) // 10 * 10)
        self.effective_packet_age.copy_(age)
        sensed = self.packet[((self.tick - age) // 10) % 16, self.ids]
        return torch.cat(
            (
                sensed,
                self.adopted[:, LEGS] / 1.5,
                self.adopted[:, WHEELS] / 100,
                q.new_tensor([0.0, 0.0, 0.68]).expand(self.n, -1),
                q.new_tensor([0, 1, 0, 1, 0, 0]).expand(self.n, -1),
            ),
            -1,
        )

    def recent(self, array, count, *, policy=False):
        cursor = self.tick // 10 if policy else self.tick
        values = [array[(cursor - age) % len(array)] for age in range(1, count + 1)]
        return torch.stack(values, 1).flatten(1)

    def state_dict(self):
        return dict(
            version=VERSION,
            n=self.n,
            tick=self.tick,
            tensors={k: v.detach().clone() for k, v in vars(self).items() if torch.is_tensor(v)},
            leg={
                k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in vars(self.leg).items()
                if torch.is_tensor(v) or k == "tick"
            },
        )

    def load_state_dict(self, state):
        if state["version"] != VERSION or state["n"] != self.n:
            raise ValueError("Execution state identity")
        caps = state["tensors"].get("caps")
        if caps is None or not torch.equal(caps.to(self.caps.device, self.caps.dtype), self.caps):
            raise ValueError("Execution电机限矩合同不同")
        for collection, obj in [(state["tensors"], self), (state["leg"], self.leg)]:
            for k, v in collection.items():
                if torch.is_tensor(v):
                    current = getattr(obj, k)
                    if current.shape != v.shape:
                        raise ValueError("Execution tensor shape " + k)
                    current.copy_(v.to(current.device, current.dtype))
                else:
                    setattr(obj, k, copy.deepcopy(v))
        self.tick = state["tick"]
