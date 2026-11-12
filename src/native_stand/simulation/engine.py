"""通用PPO任务适配器：环境、网络、算法与一次更新的具体实现。

调度、计费、保存与边界不在这里，而在 `simulation/driver.py`。任务适配者
只负责"这一阶段怎么构造"和"一次更新要做什么"，因此同一套驱动器既能跑通用
PPO，也能跑项目里已有的受控Motion循环。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from native_stand.learning.ppo import assembly

from .driver import Backend, LimitReached, StageResult
from .driver import run_stage as driver_run_stage

__all__ = [
    "Backend",
    "GenericPPOAdapter",
    "LimitReached",
    "StageResult",
    "initialization_payload",
    "run_stage",
]


def default_action_mask(job) -> bool:
    """受控PPO要求 `recovery_actor_valid` 列；CPU fixture全部有效。"""
    return job.task in ("motion", "recovery", "jump")


def initialization_payload(job):
    """三种来源只在一个入口解析：显式恢复→阶段接续→配置声明的完整初始化。

    配置声明 full 却拿不到来源时必须报错，不能静默按 scratch 训练。
    """
    from native_stand.artifacts.checkpoint import load_source

    init = job.initialization
    if init.mode != "full":
        return None
    if not init.checkpoint:
        raise ValueError("initialization.mode=full 但没有声明 checkpoint")
    path = Path(init.checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"初始化来源不存在: {path}")
    payload = load_source(path)
    if init.sha256 and payload.sha256 != init.sha256:
        raise ValueError(f"初始化来源SHA与声明不一致: {path}")
    return payload


_initialization_payload = initialization_payload  # 兼容旧名，仅本模块使用


class GenericPPOAdapter:
    """通用PPO：CPU fixture与未迁移任务共用；任务差异由后端与配置表达。"""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.env = None
        self.ppo = None
        self.job = None
        self.stage = None
        self._warmup = 0
        self._steps_per_env = 0
        self._reward_total = 0.0
        self._reward_samples_device = None
        self._reward_total_device = None
        self._actor_steps = 0

    # ---- 构造与恢复 ------------------------------------------------
    def build(self, context, stage: str, resume_from: dict[str, Any] | None) -> None:
        job = context.job
        self.job = job
        self.stage = stage
        from native_stand.learning.ppo.performance import Measurement

        self.measurement = getattr(context, "measurement", None) or Measurement(
            job.diagnostics.get("performance_measurement", False), job.runtime.device
        )
        with self.measurement.section("environment_construction"):
            measured_factory = getattr(self.backend, "make_measured_env", None)
            self.env = (
                measured_factory(job, self.measurement)
                if measured_factory is not None
                else self.backend.make_env(job)
            )
        bind = getattr(self.env, "set_measurement", None)
        if callable(bind):
            bind(self.measurement)
        with self.measurement.section("first_observation_readback"):
            observations = self.env.observations()
        self._policy_handle = None
        # 任务声明的policy factory经registry进入；选择依据是动作/分布契约声明，
        # 不是task名称或输入维数。
        from native_stand.learning.policies.factory import build_policy

        if resume_from is not None:
            from native_stand.artifacts.checkpoint import LoadedSource

            if not isinstance(resume_from, LoadedSource):
                raise TypeError("同run恢复必须携带LoadedSource路径和SHA身份")
            current_run = getattr(context, "run_dir", None)
            from native_stand.artifacts.handoff import validate_resume_identity

            if current_run is None:
                raise ValueError("恢复缺少run目录")
            validate_resume_identity(resume_from, current_run)
            if int(resume_from.payload["counters"]["updates"]) != context.updates_used:
                raise ValueError("恢复点计数与本run账本不同")
        payload = resume_from if resume_from is not None else initialization_payload(job)
        if payload is None and job.initialization.mode == "actor_only":
            from native_stand.artifacts.checkpoint import load_actor_source

            source = load_actor_source(job.initialization.checkpoint)
            if source.sha256 != job.initialization.sha256:
                raise ValueError("actor_only来源SHA与配置声明不一致")
            payload = source
        handle = build_policy(job, observations, device=job.runtime.device, source=payload)
        self.ppo = handle["ppo"]
        self.ppo.measurement = self.measurement
        self._policy_handle = handle if "capture" in handle else None
        restored_payload = getattr(payload, "payload", payload)
        if restored_payload and "optimizer" in restored_payload:
            if self._policy_handle is not None:
                self._policy_handle["restore"](restored_payload)
                if resume_from is not None:
                    context.restore_runtime_state(restored_payload)
            else:
                restored = assembly.restore_full(self.ppo, restored_payload)
                if resume_from is not None:
                    context.restore_runtime_state(restored)
        payload = restored_payload
        if isinstance(payload, dict):
            # checkpoint里的runtime_state必须交给环境恢复；actor_only来源不携带。
            runtime = payload.get("runtime_state", payload.get("execution_state"))
            if resume_from is not None and runtime and hasattr(self.env, "load_runtime_state"):
                self.env.load_runtime_state(runtime)
            if (
                resume_from is None
                and runtime
                and hasattr(self.env, "initialize_from_training_state")
            ):
                self.env.initialize_from_training_state(runtime)
            from native_stand.students.resume import restore_rng

            if payload.get("training_rng") is not None:
                restore_rng(payload["training_rng"])
            elif payload.get("rng") is not None:
                import torch

                torch.set_rng_state(payload["rng"])
                cuda = payload.get("CUDA_rng")
                if cuda is not None:
                    # Legacy Recovery used get_rng_state(device), not the
                    # list returned by get_rng_state_all(). Both are explicit
                    # stored formats; never reinterpret bytes as device rows.
                    if isinstance(cuda, torch.Tensor):
                        cuda = [cuda]
                    if not torch.cuda.is_initialized() or len(cuda) != torch.cuda.device_count():
                        raise ValueError("legacy恢复CUDA RNG映射不同")
                    torch.cuda.set_rng_state_all([x.cpu() for x in cuda])
        prior = (payload or {}).get("warmup_updates_done")
        if (
            prior is None
            and (payload or {}).get("version") == "recovery_cubic_physical_Gaussian_v2"
            and "optimizer" in payload
        ):
            # This explicit legacy schema counted critic-only updates in updates.
            prior = min(job.ppo.critic_warmup_updates, int(payload["updates"]))
            if payload["actor_update_enabled"] and prior != job.ppo.critic_warmup_updates:
                raise ValueError("legacy Recovery warmup与Actor控制状态冲突")
        if prior is None:
            prior = min(job.ppo.critic_warmup_updates, context.total_updates_before(stage))
        self._prior_warmup = int(prior)
        self._restore_updates = context.updates_used
        self._warmup = context.updates_used + max(
            0, job.ppo.critic_warmup_updates - self._prior_warmup
        )
        self._steps_per_env = job.runtime.steps_per_env
        self._reward_total = 0.0
        self._reward_samples_device = None

    # ---- 一次更新 --------------------------------------------------
    def update_once(self, context, index: int) -> dict[str, Any]:
        job = self.job
        ppo = self.ppo
        self.completed_optimization = None
        started = time.monotonic()
        # 本轮统计从零开始：reward_mean是本轮均值，不是跨update累计均值。
        self._reward_total_device = None
        self._reward_samples_device = None
        import torch

        from native_stand.learning.ppo.performance import Measurement

        timer = getattr(self, "measurement", None)
        if timer is None:
            timer = Measurement(
                job.diagnostics.get("performance_measurement", False), job.runtime.device
            )
        ppo.performance_measurement = timer.enabled
        # Storage is allocated in build, outside inference_mode. Training targets
        # must also be ordinary tensors, not inference tensors needed by backward.
        with timer.section("rollout"), torch.inference_mode():
            for _ in range(self._steps_per_env):
                observations = self.env.observations()
                actions = ppo.act(observations)
                outcome = self.env.step(actions)
                transition_valid = outcome.get("extras", {}).get("transition_valid")
                if transition_valid is not None:
                    ppo.transition.observations["recovery_actor_valid"] &= transition_valid.reshape(
                        -1, 1
                    )
                # 本轮奖励总和与实际样本数都在设备端累计；
                # 指标边界一次性转标量，避免每个policy step回读GPU。
                step_sum = outcome["rewards"].sum()
                self._reward_total_device = (
                    step_sum
                    if self._reward_total_device is None
                    else self._reward_total_device + step_sum
                )
                self._reward_samples_device = (
                    outcome["rewards"].numel()
                    if self._reward_samples_device is None
                    else self._reward_samples_device + outcome["rewards"].numel()
                )
                stored = self._stored_reward(outcome)
                # 已手动修正timeout奖励，不再让底层按time_outs重复修正。
                ppo.process_env_step(observations, stored, outcome["dones"], {})
        with timer.section("returns"), torch.no_grad():
            observations = self.env.observations()
            ppo.compute_returns(observations)
            # normalize_valid creates a normal tensor under no_grad. Keep this
            # boundary explicit for backends returning inference observations.
        allow_actor = context.updates_used + 1 > self._warmup
        ppo.actor_update_enabled = allow_actor
        with timer.section("learning"):
            losses = ppo.update()
        # Receipt precedes telemetry conversion: a diagnostic exception must not
        # refund an optimization that actually completed. It is not a claim that
        # task postprocessing or the current physical state is checkpoint-safe.
        self.completed_optimization = {"update": context.updates_used + 1}
        self.performance_timings = {**timer.seconds, **getattr(ppo, "performance_timings", {})}
        self._actor_steps = int(ppo.mask_summary.get("actor_optimizer_steps", 0) or 0)
        return {
            # 本轮均值 = 本轮奖励总和 / 实际奖励样本数；零样本不得除零。
            **(
                {
                    "performance_seconds": self.performance_timings,
                    "performance_clock": "see performance_clocks; inclusive wall sections",
                }
                if timer.enabled
                else {}
            ),
            "reward_mean": (
                float(self._reward_total_device) / self._reward_samples_device
                if self._reward_total_device is not None and self._reward_samples_device
                else 0.0
            ),
            "value_loss": losses.get("value"),
            "surrogate": losses.get("surrogate"),
            "entropy": losses.get("entropy"),
            "critic_warmup": not allow_actor,
            "actor_optimizer_steps": self._actor_steps,
            "critic_optimizer_steps": ppo.mask_summary.get("critic_optimizer_steps"),
            "actor_lr": job.ppo.learning_rate,
            "critic_lr": job.ppo.critic_learning_rate,
            "seconds_per_update": time.monotonic() - started,
            "actor_steps": self._actor_steps,
            "critic_only": (not allow_actor),
        }

    def _stored_reward(self, outcome):
        """截断终点的bootstrap策略。

        形状契约：奖励统一为每环境一个标量[N,1]；[N]输入显式转换，
        环境数量不匹配立即报错——不得产生[N,N]广播矩阵。
        终点策略由任务合同决定：只有声明infinite_horizon的环境，
        truncated & ~terminated 才用reset前终态Critic做一次bootstrap；
        有限任务（如23秒timeout）保持原始reward，不bootstrap。
        """
        import torch

        rewards = outcome["rewards"]
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        num_envs = getattr(self.env, "num_envs", rewards.shape[0])
        if rewards.shape != (num_envs, 1):
            raise ValueError(
                f"reward形状{tuple(rewards.shape)}与环境数{num_envs}不符；每环境必须一个标量"
            )
        truncated = outcome.get("truncated")
        terminated = outcome.get("terminated")
        terminal = outcome.get(
            "terminal_observations",
            outcome.get("terminal_critic", outcome.get("extras", {}).get("terminal_critic")),
        )
        # 有限任务的timeout是任务终点，不是中断；只有无限任务才需要bootstrap。
        infinite = bool(getattr(self.env, "infinite_horizon", False))
        if not infinite or truncated is None or terminated is None or terminal is None:
            return rewards
        mask = truncated.reshape(-1, 1).bool() & ~terminated.reshape(-1, 1).bool()
        if not bool(mask.any()):
            return rewards
        with torch.no_grad():
            named = terminal if hasattr(terminal, "keys") else {"student_critic": terminal}
            values = self.ppo.critic(named).reshape(-1, 1)
        return rewards + float(self.ppo.gamma) * values * mask.to(rewards.dtype)

    def after_update(self, context, index, metrics):
        """通用路径没有任务后处理；保留空实现以统一adapter契约。"""
        return None

    # ---- 完整状态 --------------------------------------------------
    def capture(self, context, counters: dict[str, Any]) -> dict[str, Any]:
        from native_stand.students.resume import rng_state

        if self._policy_handle is not None:
            state = self._policy_handle["capture"](
                counters,
                curriculum=context.curriculum_state(),
                execution_state=self.env.runtime_state()
                if hasattr(self.env, "runtime_state")
                else None,
            )
            state["training_rng"] = rng_state()
            state["warmup_updates_done"] = min(
                self.job.ppo.critic_warmup_updates,
                self._prior_warmup
                + int(counters.get("updates", context.updates_used))
                - self._restore_updates,
            )
            return state
        state = assembly.capture_full(
            self.ppo,
            counters=counters,
            curriculum=context.curriculum_state(),
            mixing={"symmetry_mix": None},
            source_identity=context.source_identity,
        )
        state["warmup_updates_done"] = min(
            self.job.ppo.critic_warmup_updates,
            self._prior_warmup
            + int(counters.get("updates", context.updates_used))
            - self._restore_updates,
        )
        state["training_rng"] = rng_state()
        state["runtime_state"] = self.env.runtime_state()
        return state

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


def run_stage(
    context,
    *,
    backend: Backend,
    stage: str,
    restore: dict[str, Any] | None = None,
    resume_from: dict[str, Any] | None = None,
) -> StageResult:
    """通用PPO入口；真正的调度与保存交给公共驱动器。"""
    payload = resume_from or restore
    return driver_run_stage(context, GenericPPOAdapter(backend), stage=stage, resume_from=payload)
