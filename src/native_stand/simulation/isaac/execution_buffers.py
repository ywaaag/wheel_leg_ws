"""完整V4执行实际消费的队列；不构造已被覆盖的旧Plant/PD/观测链。"""

import torch

from native_stand.students.motion_mcu_execution import MotorRing


class ExecutionBuffers:
    def __init__(
        self, env, config, *, armature_level, effort_caps=(40.0, 40.0, 5.5, 40.0, 40.0, 5.5)
    ):
        self.env = env
        self.config = config
        self.n = env.num_envs
        self.device = env.device
        self.tick = 0
        self.ids = torch.arange(self.n, device=self.device)

        def z(*shape):
            return torch.zeros(*shape, device=self.device)

        self.limits = z(self.n, 6) + torch.tensor(effort_caps, device=self.device)
        self.targets = z(32, self.n, 6)
        self.sensors = z(32, self.n, 12)
        self.tail = z(1000, 44)
        self.episodes = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.adopted_origin = torch.zeros(self.n, 6, device=self.device, dtype=torch.long)
        self.delays = torch.full_like(self.adopted_origin, 5)
        self.feedback = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.geometry_valid = torch.ones(self.n, device=self.device, dtype=torch.bool)
        for name in (
            "requested",
            "adopted",
            "actual",
            "demand",
            "driver",
            "compensation",
            "motor_actual",
            "spring_actual",
            "pd_limited",
            "prev_torque",
            "critic_previous_torque",
            "prev_target",
            "older_target",
        ):
            setattr(self, name, z(self.n, 6))
        self.last_pd_sensors = z(self.n, 12)
        self.motor_ring = MotorRing(self.n, self.device)
        from native_stand.control_readback import tensor
        from native_stand.students.motion_antioscillation import ARMATURES
        from native_stand.students.motion_wheel_matching import setup

        self.wheel_readback = setup(env, config["wheel_execution"])
        ids = torch.tensor(
            [env.ids[i] for i in (0, 1, 3, 4)], device=self.device, dtype=torch.int32
        )
        expected = z(self.n, 4) + ARMATURES[armature_level]
        env.robot.write_joint_armature_to_sim_index(armature=expected.contiguous(), joint_ids=ids)
        self.actual_armature = torch.as_tensor(
            tensor(env.robot.root_view.get_dof_armatures()), device=self.device
        )[:, ids.long()]
        if not torch.allclose(self.actual_armature, expected, atol=1e-7, rtol=0):
            raise RuntimeError("V4腿armature实际读回不符")
        self.history_valid = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self.last_reward_tick = None

    def reset(self, ids):
        e = self.env
        q = e.robot.data.joint_pos.torch[:, e.ids]
        dq = e.robot.data.joint_vel.torch[:, e.ids]
        target = q[ids].clone()
        target[:, [2, 5]] = 0
        self.requested[ids] = target
        self.adopted[ids] = target
        self.targets[:, ids] = target[None]
        self.sensors[:, ids] = torch.cat((q[ids], dq[ids]), -1)[None]
        self.last_pd_sensors[ids] = self.sensors[0, ids]
        self.episodes[ids] += 1
        self.adopted_origin[ids] = self.tick
        self.history_valid[ids] = False
        for name in ("prev_torque", "critic_previous_torque"):
            getattr(self, name)[ids] = 0
        self.prev_target[ids] = target
        self.older_target[ids] = target

    def configure(self, level=None):
        # No runtime Plant mutation; curriculum DR owns reset-time parameters.
        if level is not None:
            raise ValueError("完整V4参数由冻结spec与显式DR定义")

    def features(self):
        raise RuntimeError("完整V4 Critic使用实际执行列，旧中间特权编码已移除")
