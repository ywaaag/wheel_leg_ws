"""监督Actor到新PPO的明确初始化；绝不导入教师Critic、Adam、RNG或旧课程。"""

from __future__ import annotations

import copy

import torch

from .networks import StudentNetwork

ACTION = "motion_multiheight_support_center_015_80_v1"


def validate_actor(payload, job, contract):
    if payload.get("actor_only") is not True or any(
        k in payload for k in ("critic", "optimizer", "training_rng")
    ):
        raise ValueError("actor_only初始化只接受监督Actor，不能拼接旧PPO状态")
    architecture = payload.get("network_architecture", {})
    if (
        architecture.get("actor_input") != job.model.actor_input
        or tuple(architecture.get("actor_hidden_dims", ())) != job.model.actor_hidden_dims
    ):
        raise ValueError("大Actor真实架构与新作业不一致")
    if contract.get("actor_dimensions") != [31, *job.model.actor_hidden_dims, 6]:
        raise ValueError("大Actor合同必须声明实际大网络维数")
    if contract.get("action_version") != ACTION or job.action.version != ACTION:
        raise ValueError("新Actor必须使用明确C0动作版本")
    original = payload.get("output_contract", {})
    if original.get("actor_dimensions") != contract["actor_dimensions"]:
        raise ValueError("大Actor内嵌合同必须声明实际大网络维数；小合同仅作来源记录")
    if {k: v for k, v in original.items() if k != "actor_dimensions"} != {
        k: v for k, v in contract.items() if k != "actor_dimensions"
    }:
        raise ValueError("监督Actor物理接口与C0大合同不一致")
    if (
        contract.get("input_version") != "multiskill_mcu_v2"
        or contract.get("leg_halfwidth_rad") != 0.15
        or contract.get("wheel_halfwidth_rad_s") != 80.0
        or contract.get("input_leg_q_and_previous_scale_rad") != 1.5
        or contract.get("input_wheel_previous_scale_rad_s") != 100.0
    ):
        raise ValueError("C0输入/动作尺度不一致")
    return True


def physical_std(job):
    return torch.tensor(
        [
            job.ppo.exploration_wheel_rad_s / job.action.wheel_scale_rad_s
            if "wheel" in name
            else job.ppo.exploration_leg_rad / job.action.leg_scale_rad
            for name in job.action.order
        ],
        dtype=torch.float32,
    )


def models(job):
    actor = StudentNetwork(
        actor=True,
        input_dim=31,
        hidden_dims=job.model.actor_hidden_dims,
        output_dim=6,
        init_std=physical_std(job),
    )
    critic = StudentNetwork(
        actor=False,
        input_dim=job.model.critic_input,
        hidden_dims=job.model.critic_hidden_dims,
        output_dim=1,
    )
    return actor, critic


def training_source(payload, job, contract, architecture):
    """调用方先设置本次新运行seed；本函数只构造新Critic与明确的环境metadata。"""
    validate_actor(payload, job, contract)
    actor, critic = models(job)
    weights = copy.deepcopy(payload["actor"])
    weights["log_std"] = physical_std(job).log()
    actor.load_state_dict(weights, strict=True)
    if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
        raise ValueError("大Actor包含非有限权重")
    from native_stand.tasks.motion.course_strategy import resolve

    strategy = resolve(job.curriculum.version)
    state = strategy.initial_state(job.assets)
    return dict(
        interface_version=job.model.interface,
        role="MotionZeroStudent",
        actor=actor.state_dict(),
        critic=critic.state_dict(),
        network_architecture=copy.deepcopy(architecture),
        critic_privileged_version=job.model.critic_privileged_version,
        output_contract=copy.deepcopy(contract),
        action_version=ACTION,
        motion_course_state=state,
        reward_refinement_state=[],
        symmetry_mix_state=[],
        warmup_updates_done=0,
        recipe_updates=0,
        online_updates_including_smoke=0,
        update=0,
        dataset_sha256=payload.get("dataset_sha256"),
        manifest_sha256=payload.get("manifest_sha256"),
        asset_sha256=payload.get("asset_sha256"),
        initialization_record=dict(
            mode="actor_only",
            fresh_Critic=True,
            fresh_two_Adam=True,
            fresh_RNG=True,
            fresh_course=True,
            teacher_optimizer_imported=False,
            source_actor_sha256=job.initialization.sha256,
            output_contract_sha256=job.assets["output_contract_sha256"],
            source_C0_contract_sha256=job.assets["prefix_output_contract_sha256"],
            center_definition_canonical_json_sha256=job.assets[
                "center_definition_canonical_json_sha256"
            ],
        ),
    )
