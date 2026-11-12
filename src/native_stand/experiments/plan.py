"""只读计划估算：--dry-run 与实际提交共用同一份解析结果。

估算不创建run、不写预算、不启动引擎、不发送通知。
"""

from __future__ import annotations

from typing import Any

from .config import JobConfig


def _mlp_parameters(inputs: int, hidden: tuple[int, ...], outputs: int) -> int:
    total = 0
    previous = inputs
    for width in hidden:
        total += previous * width + width
        previous = width
    return total + previous * outputs + outputs


def model_parameters(job: JobConfig) -> dict[str, int]:
    """按配置的隐层与输入输出估算参数量，不构造网络。"""
    actor = _mlp_parameters(job.model.actor_input, job.model.actor_hidden_dims, job.action.dim)
    critic = _mlp_parameters(job.model.critic_input, job.model.critic_hidden_dims, 1)
    return {"actor": actor, "critic": critic, "total": actor + critic}


def checkpoint_bytes(job: JobConfig) -> int:
    """完整恢复状态估算：参数fp32 + Adam两个动量 + 固定开销。"""
    parameters = model_parameters(job)["total"]
    fixed_overhead = 8 * 1024 * 1024
    return parameters * 4 * 3 + fixed_overhead


def stage_updates(job: JobConfig) -> list[dict[str, Any]]:
    """阶段与update/sample分配；smoke为0时不生成该阶段。"""
    stages: list[dict[str, Any]] = []
    if job.limits.smoke_updates:
        stages.append(
            {
                "stage": "smoke",
                "updates": job.limits.smoke_updates,
                "samples": job.limits.smoke_updates * job.runtime.samples_per_update,
            }
        )
    if job.limits.formal_updates:
        stages.append(
            {
                "stage": "formal",
                "updates": job.limits.formal_updates,
                "samples": job.limits.formal_updates * job.runtime.samples_per_update,
            }
        )
    return stages


def estimated_storage(job: JobConfig) -> dict[str, int]:
    """新run默认保存量估算：最近若干完整checkpoint + best + 报告。"""
    per_checkpoint = checkpoint_bytes(job)
    kept = job.storage.keep_recent_checkpoints + 1
    metrics_rows = (
        job.limits.smoke_updates + job.limits.formal_updates
    ) // job.storage.metrics_every_updates + 2
    return {
        "checkpoint_bytes": per_checkpoint,
        "checkpoints_kept": kept,
        "checkpoint_total_bytes": per_checkpoint * kept,
        "metrics_bytes": metrics_rows * 512,
        "evaluation_bytes": 4 * 1024 * 1024,
        "estimated_total_bytes": per_checkpoint * kept + metrics_rows * 512 + 4 * 1024 * 1024,
    }


def build_plan(job: JobConfig) -> dict[str, Any]:
    """完整计划。只读：不创建文件、不reserve、不spawn。"""
    stages = stage_updates(job)
    return {
        "name": job.name,
        "task": job.task,
        "algorithm": job.algorithm,
        "backend": job.backend,
        "preset": job.preset,
        "config_path": str(job.config_path),
        "config_sha256": job.config_sha256,
        "source_root": str(job.source_root),
        "model": {
            "interface": job.model.interface,
            "actor_input": job.model.actor_input,
            "critic_input": job.model.critic_input,
            "parameters": model_parameters(job),
        },
        "initialization": {
            "mode": job.initialization.mode,
            "checkpoint": str(job.initialization.checkpoint)
            if job.initialization.checkpoint
            else None,
            "sha256": job.initialization.sha256,
        },
        "semantics": {
            "input_scales": {
                "leg_rad": job.observation.leg_input_scale_rad,
                "wheel_rad_s": job.observation.wheel_input_scale_rad_s,
            },
            "output_scales": {
                "leg_rad": job.action.leg_scale_rad,
                "wheel_rad_s": job.action.wheel_scale_rad_s,
            },
            "action_version": job.action.version,
            "action_order": list(job.action.order),
            "periods_s": {
                "physics": job.control.physics_dt_s,
                "pd": job.control.pd_dt_s,
                "policy": job.control.policy_dt_s,
            },
            "critic_warmup_updates": job.ppo.critic_warmup_updates,
        },
        "stages": stages,
        "total_updates": sum(row["updates"] for row in stages),
        "storage": estimated_storage(job),
        "writes": False,
        "starts_engine": False,
        "notifies": False,
    }
