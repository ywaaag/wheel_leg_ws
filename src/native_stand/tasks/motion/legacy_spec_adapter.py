"""新JobConfig到冻结Motion环境的显式数值适配；不读取或写入文件。"""

from __future__ import annotations

import copy

from .scale_corrections import VERSION, validate_configuration

# 只复用已验证的场景／执行参数；授权、旧账本、旧来源校验不迁入新作业。
ENVIRONMENT_FIELDS = (
    "asset_path",
    "asset_sha256",
    "asset_module_root",
    "asset_module_path",
    "asset_module_sha256",
    "colleague_config_path",
    "colleague_config_sha256",
    "ground",
    "control",
    "termination",
    "recovery",
    "reward",
    "curriculum",
    "observation",
    "history",
    "action",
    "interface_version",
    "student_training",
    "student_motion_validation",
    "student_interface",
    "motion_training",
    "motion_course",
    "motion_timing",
    "mcu_execution",
    "motion_height_dr",
    "motion_symmetry",
    "motion_assessment",
    "motion_parallel_assessment",
    "motion_identified_response",
    "motion_comprehensive_dr",
    "motion_dq_robustness",
    "performance",
    "action_output",
    "network_architecture",
    "safety",
    "reset",
    "evaluation",
)


def build_environment_spec(template, source, job, *, stage_builder):
    if source["action_version"] != job.action.version:
        raise ValueError("本适配仅接收同动作版本的完整恢复")
    if source["critic_privileged_version"] != job.model.critic_privileged_version:
        raise ValueError("来源Critic语义与声明不同")
    config = validate_configuration(
        job.reward.fields["scale_corrections"], wheel_scale_rad_s=job.action.wheel_scale_rad_s
    )
    spec = {name: copy.deepcopy(template[name]) for name in ENVIRONMENT_FIELDS if name in template}
    control = template["control"]
    for name in ("physics_dt_s", "pd_dt_s", "policy_dt_s", "leg_kp", "leg_kd", "wheel_kv"):
        if getattr(job.control, name) != control[name]:
            raise ValueError("本次尺度修复不改变物理／PD: " + name)
    if tuple(job.model.actor_hidden_dims) != tuple(
        source["network_architecture"]["actor_hidden_dims"]
    ):
        raise ValueError("来源Actor架构不能根据形状静默转换")
    state = copy.deepcopy(source["motion_course_state"])
    stage = stage_builder(state["stage"], job.runtime.envs, state=state)
    spec.update(
        experiment=job.name,
        seed=job.runtime.seed,
        source_checkpoint=str(job.initialization.checkpoint),
        source_sha256=job.initialization.sha256,
        ppo={"envs": job.runtime.envs, "steps_per_env": job.runtime.steps_per_env},
        supported_scale_corrections=dict(config),
    )
    spec["motion_height_dr"]["state"] = state
    spec["motion_training"].update(stage=state["stage"], resume=None)
    spec["student_training"].update(
        height_dr_state=copy.deepcopy(state),
        course_version=job.curriculum.version,
        stage=state["stage"],
        course_stage=state["stage"],
        source_actor_sha256=job.initialization.sha256,
        warmup_updates_per_phase=0,
        no_independent_evaluation=True,
        critic=f"full same-semantic{job.model.critic_input} resume",
        optimizer="full split Adam resume",
        scale_corrections=dict(config),
        scale_correction_version=VERSION,
    )
    spec["motion_identified_response"].update(
        handover_source_checkpoint=job.assets["prefix_onnx"],
        handover_source_sha256=job.assets["prefix_sha256"],
        handover_source_kind=job.assets.get("prefix_identity", "explicit Teacher output contract"),
    )
    spec["curriculum"]["stages"] = [stage]
    # 新作业沿用相同输入／输出契约，物理引用来源于完整checkpoint本身。
    spec["action_output"] = copy.deepcopy(source["output_contract"])
    spec["network_architecture"] = copy.deepcopy(source["network_architecture"])
    return spec
