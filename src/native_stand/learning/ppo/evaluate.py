"""策略评估：每个条件独立环境，结果按唯一身份保存。

- 只接受完整恢复checkpoint，按SHA记录身份；
- 每次评估一个后端会话；每个条件新建环境，用完关闭，避免课程/延迟/随机化串扰；
- 使用 evaluation.envs / evaluation.seed，而不是训练环境数量；
- 逐条件逐环境保留原始结果，提前失败与未知项分别标注；
- best与final由调用方区分，本模块不自行授予capability accepted。
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from pathlib import Path
from typing import Any

from native_stand.artifacts import evaluation as evaluation_format
from native_stand.artifacts.checkpoint import load_complete_state
from native_stand.artifacts.io import file_sha256
from native_stand.experiments.config import JobConfig
from native_stand.simulation import dispatch

from . import assembly


class EvaluationError(ValueError):
    """评估配置或环境不支持声明的条件。"""


def default_conditions(job: JobConfig) -> tuple[dict[str, Any], ...]:
    if job.evaluation.conditions:
        return tuple(dict(row) for row in job.evaluation.conditions)
    return ({"name": "nominal", "command": [0.0, 0.0, 0.0]},)


def evaluation_job(job: JobConfig) -> JobConfig:
    """独立评估使用自己的环境数量与seed，不继承训练环境数。"""
    return dataclasses.replace(
        job,
        runtime=dataclasses.replace(
            job.runtime, envs=job.evaluation.envs, seed=job.evaluation.seed
        ),
    )


CONDITION_KEYS = {"name", "command"}


def _apply_condition(env, condition: dict[str, Any], hook=None) -> None:
    """条件必须实际作用于环境；未知字段明确拒绝，不能只写进结果JSON。"""
    if hook is not None:
        hook(env, condition)
        return
    unknown = set(condition) - CONDITION_KEYS
    if unknown:
        raise EvaluationError(f"当前环境不支持条件字段: {sorted(unknown)}")
    command = condition.get("command")
    if command is None:
        return
    if not hasattr(env, "set_command"):
        raise EvaluationError(
            f"环境不支持命令条件 {condition.get('name')!r}；拒绝生成看似完成的评估"
        )
    env.set_command(command)


def _rollout(env, actor, *, seconds: float, policy_dt: float) -> dict[str, Any]:
    import torch

    steps = max(1, int(round(seconds / policy_dt)))
    device = torch.device(getattr(env, "device", "cpu"))
    returns = torch.zeros(env.num_envs, device=device)
    first_failure: dict[int, dict[str, Any]] = {}
    active = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    task_success = [False] * env.num_envs
    task_failure = [None] * env.num_envs
    for step in range(steps):
        with torch.no_grad():
            actions = actor(env.observations(), stochastic_output=False)
        outcome = env.step(actions)
        rewards = outcome["rewards"].reshape(-1)
        # 首次失败后冻结该环境原episode的成绩：reset后的奖励不进入本次窗口。
        returns = returns + rewards * active.to(rewards.dtype)
        terminated = outcome.get("terminated")
        truncated = outcome.get("truncated")
        if terminated is None:
            terminated = outcome["dones"]
            truncated = outcome["dones"]
        terminated = terminated.reshape(-1).bool()
        if truncated is None:
            truncated = torch.zeros_like(terminated)
        truncated = truncated.reshape(-1).bool()
        for index in range(env.num_envs):
            if not bool(active[index]):
                continue
            if bool(terminated[index]) or bool(truncated[index]):
                # 走完整个窗口的正常截断是窗口终点，不算提前失败。
                if bool(truncated[index]) and not bool(terminated[index]) and step == steps - 1:
                    continue
                # 逐环境成功/失败语义：环境结果结构声明，不用批次级布尔。
                success_flags = outcome.get("task_success")
                failure_flags = outcome.get("task_failure")
                success_i = bool(success_flags[index]) if success_flags is not None else False
                failure_i = (
                    failure_flags[index]
                    if failure_flags is not None and failure_flags[index]
                    else None
                )
                if bool(terminated[index]) and success_i:
                    task_success[index] = True
                    active[index] = False
                    continue
                if bool(terminated[index]) and failure_i:
                    task_failure[index] = failure_i
                    first_failure[index] = {
                        "step": step,
                        "seconds": (step + 1) * policy_dt,
                        "reason": f"task_failure:{failure_i}",
                    }
                    active[index] = False
                    continue
                if bool(terminated[index]):
                    # 无成功或失败证据的终止：UNKNOWN，不能猜。
                    first_failure[index] = {
                        "step": step,
                        "seconds": (step + 1) * policy_dt,
                        "reason": "terminated_unknown",
                    }
                    active[index] = False
                    continue
                first_failure[index] = {
                    "step": step,
                    # 首步结束也要记一个完整policy interval，而不是0秒。
                    "seconds": (step + 1) * policy_dt,
                    "reason": "terminated" if bool(terminated[index]) else "timeout",
                }
                active[index] = False
    return {
        "steps": steps,
        "returns": returns,
        "first_failure": first_failure,
        "active": active,
        "task_success": task_success,
        "task_failure": task_failure,
    }


def evaluate_checkpoint(
    job: JobConfig,
    checkpoint: str | Path,
    *,
    kind: str = "final",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """执行完整矩阵评估；每个条件独立环境，结果按 evaluation_id 保存。"""
    # checkpoint只加载一次：权重、环境来源与结果身份使用同一个LoadedSource。
    source = load_complete_state(checkpoint)
    payload = source.payload
    evaluation = evaluation_job(job)
    service = dispatch.execution_service(evaluation)
    if service["service"] == "unsupported":
        raise EvaluationError(service["reason"])
    backend = dispatch.backend_for_evaluation(evaluation)
    actor = assembly.build_actor(evaluation)
    actor.load_state_dict(payload["actor"], strict=True)
    # 网络、观测与统计必须与执行环境同一设备。
    actor.to(evaluation.runtime.device)
    actor.eval()

    conditions: list[dict[str, Any]] = []
    session = (
        backend.session_factory(evaluation, resume_from=source) if backend.session_factory else None
    )
    try:
        for condition in default_conditions(evaluation):
            # 每个条件一个新环境：课程、延迟缓存与随机化状态互不继承。
            env = (
                session.make_env()
                if session is not None
                else backend.make_env(evaluation, resume_from=source)
            )
            try:
                _apply_condition(env, condition, backend.condition_hook)
                started = time.monotonic()
                outcome = _rollout(
                    env,
                    actor,
                    seconds=evaluation.evaluation.seconds,
                    policy_dt=evaluation.control.policy_dt_s,
                )
                wall = time.monotonic() - started
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            per_env = []
            for index in range(evaluation.evaluation.envs):
                failure = outcome["first_failure"].get(index)
                scored = index < evaluation.evaluation.scored_envs
                per_env.append(
                    {
                        "env_index": index,
                        "scored": scored,
                        "status": "FAILED" if failure else "COMPLETED",
                        "return": float(outcome["returns"][index]),
                        "failure": failure,
                        "ended_reason": "failed" if failure else "window_complete",
                        "window_seconds": (
                            failure["seconds"]
                            if failure
                            else outcome["steps"] * evaluation.control.policy_dt_s
                        ),
                    }
                )
            conditions.append(
                {
                    "condition": condition.get("name") or "unnamed",
                    "command": condition.get("command"),
                    "parameters": {
                        k: v for k, v in condition.items() if k not in ("name", "command")
                    },
                    "envs": evaluation.evaluation.envs,
                    "scored_envs": evaluation.evaluation.scored_envs,
                    "wall_s": wall,
                    "mean_return": float(outcome["returns"].mean()),
                    "failures": len(outcome["first_failure"]),
                    "per_env": per_env,
                }
            )

    finally:
        if session is not None:
            session.close()

    evaluation_id = f"{kind}_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    result = evaluation_format.build_result(
        model={
            "checkpoint": str(Path(checkpoint).resolve()),
            "sha256": file_sha256(checkpoint),
            "update": payload.get("update"),
            "stage": payload.get("stage"),
            "interface_version": payload.get("interface_version"),
            "action_version": payload.get("action_version"),
        },
        protocol={
            "version": evaluation.evaluation.protocol_version,
            "envs": evaluation.evaluation.envs,
            "scored_envs": evaluation.evaluation.scored_envs,
            "seconds": evaluation.evaluation.seconds,
            "seed": evaluation.evaluation.seed,
            "backend": evaluation.backend,
            "service": service["service"],
            "deterministic_actions": True,
            "conditions": [row.get("name") for row in default_conditions(evaluation)],
            "capability_judged": False,
        },
        conditions=conditions,
        kind=kind,
        status="COMPLETED",
        notes=(
            "backend=cpu：只证明工程行为，不是物理能力结论" if evaluation.backend == "cpu" else None
        ),
    )
    result["evaluation_id"] = evaluation_id
    if output_dir is not None:
        directory = Path(output_dir) / evaluation_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "result.json"
        evaluation_format.write_result(path, result)
        result["path"] = str(path.resolve())
        result["sha256"] = file_sha256(path)
    return result
