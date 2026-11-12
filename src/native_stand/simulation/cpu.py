"""CPU后端：用真实网络、真实受控PPO与真实优化器跑工程验证。

它不是物理仿真：状态推进是显式声明的替代动力学，奖励是工程代理；
指标与run记录会写明 `backend=cpu`、`reward_kind=cpu_fixture`，
不允许把它当作能力结果。
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from native_stand.experiments.config import JobConfig
from native_stand.tasks.motion.encoding import (
    LEGS,
    WHEELS,
    decode_targets,
    encode_actor_input,
    encode_frame,
    encode_task,
    kinematic_center,
)

ENGINE_AVAILABLE = True

# CPU fixture的Critic列布局：显式块声明，总宽由配置声明并核对。
CRITIC_BLOCKS = (
    ("actor_input", 31),
    ("q_rad", 6),
    ("dq_rad_s", 6),
    ("contact", 4),
    ("effort_nm", 6),
    ("adopted_target", 6),
    ("command", 3),
    ("height_error", 1),
    ("pitch_roll", 1),
)
CRITIC_DIM = sum(width for _, width in CRITIC_BLOCKS)


def unavailable_reason(job: JobConfig) -> str:
    if job.model.critic_input != CRITIC_DIM:
        return (
            f"CPU后端的特权列宽固定为{CRITIC_DIM}"
            f"（{'+'.join(f'{name}:{width}' for name, width in CRITIC_BLOCKS)}），"
            f"配置声明{job.model.critic_input}；真实Motion请用backend=isaac"
        )
    return ""


def validate_job(job: JobConfig) -> None:
    reason = unavailable_reason(job)
    if reason:
        raise ValueError(reason)
    if job.model.actor_input != 31:
        raise ValueError(f"CPU后端的Actor列宽固定为31，配置声明{job.model.actor_input}")
    if job.action.dim != 6:
        raise ValueError("CPU后端的动作维数必须为6")


class CpuEnvFixture:
    """可复现的替代动力学；状态全部显式，不隐藏随机源。"""

    def __init__(self, job: JobConfig):
        validate_job(job)
        self.job = job
        self.num_envs = job.runtime.envs
        self.device = job.runtime.device
        self.generator = torch.Generator(device="cpu").manual_seed(job.runtime.seed)
        self.q = torch.zeros(self.num_envs, 6)
        self.dq = torch.zeros(self.num_envs, 6)
        self.previous_legs = torch.zeros(self.num_envs, 4)
        self.previous_wheels = torch.zeros(self.num_envs, 2)
        self.contact = torch.ones(self.num_envs, 4, dtype=torch.bool)
        self.effort = torch.zeros(self.num_envs, 6)
        self.height = torch.full((self.num_envs,), job.control.height_command_m)
        self.pitch = torch.zeros(self.num_envs)
        self.command = torch.zeros(self.num_envs, 3)
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.long)
        self.observation_checks: list[str] = []
        self._center()

    # ---- 观测 ------------------------------------------------------
    def _center(self) -> torch.Tensor:
        center = dict(self.job.assets.get("center") or {})
        required = (
            "hip_above_root_m",
            "wheel_radius_m",
            "equivalent_upper_m",
            "equivalent_lower_m",
            "hip_zero_deg",
            "active_zero_deg",
        )
        missing = [key for key in required if key not in center]
        if missing:
            raise ValueError(f"assets.center缺少几何参数: {missing}")
        offset = tuple(center.get("support_offset_rad") or (0.0, 0.0, 0.0, 0.0))
        height_range = tuple(center.get("height_range_m") or (0.30, 0.40))
        self.leg_center = kinematic_center(
            self.height,
            hip_above_root_m=float(center["hip_above_root_m"]),
            wheel_radius_m=float(center["wheel_radius_m"]),
            equivalent_upper_m=float(center["equivalent_upper_m"]),
            equivalent_lower_m=float(center["equivalent_lower_m"]),
            hip_zero_deg=float(center["hip_zero_deg"]),
            active_zero_deg=float(center["active_zero_deg"]),
            support_offset_rad=tuple(float(v) for v in offset),
            height_range_m=(float(height_range[0]), float(height_range[1])),
        )
        return self.leg_center

    def _frame(self) -> torch.Tensor:
        gravity = torch.stack(
            (torch.sin(self.pitch), torch.zeros_like(self.pitch), torch.cos(self.pitch)), -1
        )
        angular = torch.zeros(self.num_envs, 3)
        reference = torch.zeros(self.num_envs, 6)
        reference[:, LEGS] = self.leg_center
        return encode_frame(
            angular, gravity, self.q, self.dq, self.previous_legs, self.previous_wheels, reference
        )

    def _critic(self, actor_input: torch.Tensor) -> torch.Tensor:
        blocks = [
            actor_input,
            self.q,
            self.dq,
            self.contact.to(actor_input.dtype),
            self.effort,
            torch.cat((self.previous_legs * 1.5, self.previous_wheels * 100.0), -1),
            self.command,
            (self.height - self.job.control.height_command_m).unsqueeze(-1),
            self.pitch.unsqueeze(-1),
        ]
        wide = torch.cat(blocks, -1)
        if wide.shape[-1] != CRITIC_DIM:
            raise ValueError(f"Critic列宽{wide.shape[-1]}与声明{CRITIC_DIM}不一致")
        return wide

    def set_command(self, command) -> None:
        """替代动力学也支持显式命令条件；评估用它隔离不同条件。"""
        values = torch.tensor(command, dtype=self.command.dtype)
        if values.numel() != self.command.shape[-1]:
            raise ValueError(f"命令维数{values.numel()}与环境{self.command.shape[-1]}不一致")
        self.command.copy_(values)

    def observations(self, *, actor_valid: torch.Tensor | None = None) -> TensorDict:
        frame = self._frame()
        task = encode_task(self.command, skill="Motion", jump="NONE")
        actor_input = encode_actor_input(frame, task)
        valid = (
            torch.ones(self.num_envs, 1, dtype=torch.bool) if actor_valid is None else actor_valid
        )
        return TensorDict(
            {
                "student_actor": actor_input,
                "student_critic": self._critic(actor_input),
                "recovery_actor_valid": valid,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    # ---- 推进 ------------------------------------------------------
    def step(self, raw_action: torch.Tensor) -> dict[str, torch.Tensor]:
        """执行动作：解码目标、更新previous、推进替代动力学、给出代理奖励。"""
        adopted = decode_targets(raw_action.detach(), self.leg_center)["protected_target"]
        self.q = 0.95 * self.q + 0.05 * adopted
        self.dq = 0.9 * self.dq + 0.1 * adopted
        self.previous_legs = adopted[:, LEGS]
        self.previous_wheels = adopted[:, WHEELS]
        self.effort = 0.1 * self.effort + 0.9 * adopted
        self.episode_step += 1
        # 工程代理奖励：高度跟踪减姿态偏差；不是Motion能力指标。
        height_error = (self.height - self.job.control.height_command_m).abs()
        upright = self.pitch.abs()
        reward = 1.0 - height_error - upright
        timeout = self.episode_step >= max(
            1, int(self.job.evaluation.seconds / self.job.control.policy_dt_s)
        )
        reset = timeout
        dones = reset.unsqueeze(-1)
        if reset.any():
            self.q[reset] = 0.0
            self.dq[reset] = 0.0
            self.previous_legs[reset] = 0.0
            self.previous_wheels[reset] = 0.0
            self.effort[reset] = 0.0
            self.episode_step[reset] = 0
        return {
            "rewards": reward.unsqueeze(-1),
            "dones": dones,
            # rsl_rl按[N]广播time_outs，多一维会broadcast成[N,N,1]。
            "time_outs": timeout.to(dtype=torch.float32),
            "reset_envs": int(reset.sum()),
        }

    # ---- 运行状态 --------------------------------------------------
    def runtime_state(self) -> dict:
        """CPU fixture的完整环境状态；没有物理现场，只有显式替代状态。"""
        return {
            "episode_step": self.episode_step.tolist(),
            "q": self.q.tolist(),
            "dq": self.dq.tolist(),
            "previous_legs": self.previous_legs.tolist(),
            "previous_wheels": self.previous_wheels.tolist(),
            "effort": self.effort.tolist(),
            "command": self.command.tolist(),
            "environment_reset": True,
            "physics_restored": False,
            "note": "CPU后端没有物理现场；这里只还原替代动力学状态",
        }

    def load_runtime_state(self, state: dict) -> None:
        steps = state.get("episode_step")
        if steps and len(steps) == self.num_envs:
            self.episode_step = torch.tensor(steps, dtype=torch.long)
        for key, field in (
            ("q", self.q),
            ("dq", self.dq),
            ("previous_legs", self.previous_legs),
            ("previous_wheels", self.previous_wheels),
            ("effort", self.effort),
            ("command", self.command),
        ):
            values = state.get(key)
            if values is None:
                continue
            tensor = torch.tensor(values, dtype=field.dtype)
            if tensor.shape == field.shape:
                field.copy_(tensor)


def make_env(job: JobConfig, *, resume_from=None) -> CpuEnvFixture:
    return CpuEnvFixture(job)
