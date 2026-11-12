"""Isaac后端：把新配置接到既有Motion环境与真实受控PPO。

本模块只做适配：
- `AppLauncher` 先于任何Isaac runtime导入；
- 环境、物理、PD、限矩、奖励与课程全部复用既有实现；
- Critic列由既有`fullchain_runtime_impl.critic`给出，宽863，不是拼零。

未在本次重构中执行物理运行；本模块的路径由CPU后端测试与静态检查覆盖。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from native_stand.experiments.config import JobConfig

ENGINE_AVAILABLE = True

SEMANTICS = {
    "actor_columns": "31D：22帧+9任务，q/previous按1.5rad与100rad/s归一化",
    "critic_columns": "863D：既有fullchain特权列，含历史、接触、力矩与执行反馈",
    "action_envelope": "腿±0.15rad、轮±40rad/s，中心由命令高度几何加固定支撑偏移",
    "physics": "物理1ms、PD5ms、策略10ms；PD与限矩来自冻结spec",
}


def legacy_spec_path(job: JobConfig) -> Path | None:
    """真实Motion配方通过assets.legacy_spec_path绑定冻结spec。"""
    value = job.assets.get("legacy_spec_path")
    return Path(value) if value else None


def available(job: JobConfig) -> tuple[bool, str]:
    """不导入Isaac、不查GPU的判断：解释器有isaaclab，且声明的环境来源存在。

    两种声明都接受，但执行服务不同（见 dispatch.execution_service）：
    - assets.runtime_template：生产支撑中心Motion适配器（motion_backend）；
    - assets.legacy_spec_path：通用Isaac适配器（本模块）。
    """
    if runtime_template_path(job) is not None:
        template = runtime_template_path(job)
        if not template.is_file():
            return False, f"runtime_template不存在: {template}"
    else:
        spec_path = legacy_spec_path(job)
        if spec_path is None:
            return False, "Isaac需要assets.runtime_template或assets.legacy_spec_path"
        if not spec_path.is_file():
            return False, f"冻结spec不存在: {spec_path}"
    if importlib.util.find_spec("isaaclab") is None:
        return False, "当前解释器没有isaaclab；Isaac训练需在Isaac Lab环境中运行"
    return True, ""


def runtime_template_path(job: JobConfig) -> Path | None:
    value = job.assets.get("runtime_template")
    return Path(value) if value else None


def unavailable_reason(job: JobConfig) -> str:
    return available(job)[1]


def validate_job(job: JobConfig) -> None:
    ok, reason = available(job)
    if not ok:
        raise ValueError(reason)
    if job.backend != "isaac":
        raise ValueError("validate_job只接受backend=isaac")
    if job.task not in ("motion", "recovery"):
        raise ValueError(f"Isaac后端不支持task={job.task}")


def _load_spec(job: JobConfig) -> dict[str, Any]:
    import json

    path = legacy_spec_path(job)
    if path is None:
        raise ValueError("缺少assets.legacy_spec_path")
    spec = json.loads(path.read_text())
    for field, expected in (("observation", job.model.critic_input),):
        declared = (spec.get(field) or {}).get("critic_dim")
        if declared is not None and int(declared) != int(expected):
            raise ValueError(
                f"冻结spec的critic_dim={declared}与新配置{expected}不一致；拒绝以不同列语义训练"
            )
    actor_declared = (spec.get("observation") or {}).get("actor_dim")
    if actor_declared is not None and int(actor_declared) != int(job.model.actor_input):
        raise ValueError(
            f"冻结spec的actor_dim={actor_declared}与新配置{job.model.actor_input}不一致"
        )
    return spec


class IsaacMotionEnv:
    """既有Motion环境的适配器：动作解码、previous、奖励与课程都在原实现里。"""

    def __init__(self, job: JobConfig):
        from isaaclab.app import AppLauncher  # 必须先于Isaac runtime导入

        self.job = job
        self.app = AppLauncher(
            {"headless": True, "device": job.runtime.device, "multi_gpu": False}
        ).app
        import torch

        torch.manual_seed(job.runtime.seed)
        self.spec = _load_spec(job)
        from native_stand.students.motion_runtime import (
            environment,  # 冻结实现
            motion_observations,
        )

        self._observations = motion_observations
        self.env = environment(self.spec, "MotionZeroStudent", envs=job.runtime.envs)
        self.env.curriculum_training = True
        self.num_envs = job.runtime.envs
        self.device = job.runtime.device
        # 接口形态在构造时确定一次：不再每个step调用inspect.signature()。
        import inspect

        try:
            parameters = inspect.signature(self.env.step).parameters
        except (TypeError, ValueError):
            parameters = {}
        self._return_extras = "return_extras" in parameters
        observation, _ = self.env.reset()
        self._obs = observation
        self._last: torch.Tensor | None = None

    # ---- 观测 ------------------------------------------------------
    def observations(self):
        import torch
        from tensordict import TensorDict

        observation = self._observations(
            self.env,
            self._obs,
            (self.spec.get("student_training") or {}).get("source_policy_retention"),
        )
        actor = observation["student_actor"]
        critic = observation["student_critic"]
        if actor.shape[-1] != self.job.model.actor_input:
            raise ValueError(f"Actor列宽{actor.shape[-1]}与声明{self.job.model.actor_input}不一致")
        if critic.shape[-1] != self.job.model.critic_input:
            raise ValueError(
                f"Critic列宽{critic.shape[-1]}与声明{self.job.model.critic_input}不一致"
            )
        valid = observation.get("recovery_actor_valid")
        if valid is None:
            valid = torch.ones(self.num_envs, 1, dtype=torch.bool)
        return TensorDict(
            {"student_actor": actor, "student_critic": critic, "recovery_actor_valid": valid},
            batch_size=[self.num_envs],
            device=self.device,
        )

    # ---- 推进 ------------------------------------------------------
    def step(self, raw_action):
        """执行一步；terminated/truncated/reset 原样保留，不再清零。"""
        import torch

        extras = {}
        if self._return_extras:
            outcome = self.env.step(
                raw_action, motion_observations_key="student_actor", return_extras=True
            )
            observation, reward, terminated, truncated, extras = outcome[:5]
        else:
            observation, reward, terminated, *rest = self.env.step(raw_action)
            extras = (
                rest[0]
                if rest and isinstance(rest[0], dict)
                else (rest[1] if len(rest) > 1 and isinstance(rest[1], dict) else {})
            )
            has_truncated = bool(rest) and not isinstance(rest[0], dict)
            truncated = (
                torch.as_tensor(rest[0], dtype=torch.bool)
                if has_truncated
                else torch.as_tensor(terminated, dtype=torch.bool)
            )
        if isinstance(observation, tuple):
            observation = observation[0]
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool).reshape(self.num_envs)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool).reshape(self.num_envs)
        dones = terminated_t | truncated_t
        # 超时bootstrap需要的终态Critic观测；reset后的下一帧单独给出。
        terminal_obs = extras.get("terminal_observations")
        self._obs = observation
        return {
            "rewards": torch.as_tensor(reward, dtype=torch.float32).reshape(self.num_envs, 1),
            "terminated": terminated_t.unsqueeze(-1),
            "truncated": truncated_t.unsqueeze(-1),
            "dones": dones.unsqueeze(-1),
            "time_outs": truncated_t.to(dtype=torch.float32),
            "reset_envs": int(dones.sum()),
            "reset": dones,
            "extras": extras,
            "terminal_observations": terminal_obs,
            "next_observations": observation,
        }

    # ---- 运行状态 --------------------------------------------------
    def runtime_state(self) -> dict[str, Any]:
        state = {
            "environment_reset": True,
            "physics_restored": False,
            "note": "Isaac环境reset；物理现场不随checkpoint恢复",
        }
        course = getattr(self.env, "course_state", None)
        if isinstance(course, dict):
            state["course"] = dict(course)
        return state

    def load_runtime_state(self, state: dict[str, Any]) -> None:
        course = state.get("course")
        if isinstance(course, dict) and hasattr(self.env, "course_state"):
            self.env.course_state = dict(course)


def make_env(job: JobConfig) -> IsaacMotionEnv:
    return IsaacMotionEnv(job)


def preflight(job: JobConfig) -> dict[str, Any]:
    """启动前核对，不创建环境。"""
    validate_job(job)
    return {"backend": "isaac", "spec": str(legacy_spec_path(job)), "semantics": SEMANTICS}
