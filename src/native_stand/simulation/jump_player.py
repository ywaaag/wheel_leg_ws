"""Task control adapter for the shared player loop, same ABI4 C/V4 as training."""

from collections import deque

import numpy as np
import torch

from native_stand.learning.inference import build_actor, spec_for_job

from .jump_backend import make_plant


class JumpPlayer:
    physics_dt = 0.001

    def __init__(self, plan, payload, output):
        if payload.get("output_contract", payload.get("contract")) != plan.output_contract:
            raise ValueError("Jump播放模型合同不同")
        self.plan = plan
        self.actor = (
            build_actor(spec_for_job(plan.job, plan.output_contract)).eval().requires_grad_(False)
        )
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.plant = make_plant(plan.job, output=output, charge_step=lambda n: None)
        self.execution = self.plant.execution
        self.model = self.plant.model
        self.data = self.plant.data
        self.records = deque(maxlen=10000)
        self.reset()

    def reset(self):
        self.plant.initialize_case(
            dict(command=[0, 0, 0.34], leg_length_m=0.28, drop_m=0.0, jump_request_after_ms=None)
        )

    def step(self, controls):
        p, e = self.plant, self.execution
        envelope = self.plan.output_contract["control"]["balance_command_envelope"]
        if abs(controls.vx) > envelope["vx_m_s"] or abs(controls.yaw) > envelope["yaw_rad_s"]:
            raise ValueError("Jump BALANCE命令超合同")
        p.command[:2] = [controls.vx, controls.yaw]
        if controls.jump_requested:
            if controls.vx or controls.yaw or e.c.states[0].phase != 9:
                raise ValueError("主动Jump请求只允许静止BALANCE")
            p.jump_request_after_ms = e.tick - p.episode_start_tick
            controls.jump_requested = False
        e.update([p.feedback()])
        if e.tick % 10 == 0:
            obs, valid = e.observe()
            with torch.inference_mode():
                raw = self.actor.mean(torch.tensor(obs))
            if (e.publish_raw(raw, valid) < 0).any():
                raise RuntimeError("Jump播放C拒绝raw请求")
        p.integrate_control()
        self.records.append(p.readback())

    def trace(self):
        return {
            name: np.asarray([r[name] for r in self.records])
            for name in ("time_s", "phase", "owner", "requested", "adopted", "motor_Nm", "NET_Nm")
        }

    def close(self):
        pass
