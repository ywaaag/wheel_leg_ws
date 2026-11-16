"""ABI2 Jump任务视图；输出/轨迹身份与完整Critic布局必须显式提供。"""

from dataclasses import dataclass

from .nominal import CONFIG_ID, OUTPUT_ID, VERSION


@dataclass(frozen=True)
class JumpTask:
    job: object
    contract: dict

    def evaluation_protocol(self):
        return dict(
            version=self.job.evaluation.protocol_version,
            seconds=self.job.evaluation.seconds,
            conditions=list(self.job.evaluation.conditions),
            capability_accepted=False,
        )


def build_task(job):
    from .isaac_contract import CRITIC, TRAINING

    if job.action.version == TRAINING:
        import math

        from .unified import contract as unified_contract
        from . import skill_course

        skill_course.validate(job)
        critic_version = skill_course.CRITIC if skill_course.enabled(job) else CRITIC

        contract = job.assets["jump_contract_data"]
        schema = job.assets["jump_critic_schema"]
        if (
            contract != unified_contract()
            or job.model.interface != TRAINING
            or job.model.action_version != TRAINING
        ):
            raise ValueError("Jump Isaac shares the exact ABI4 physical Actor contract")
        if (job.model.actor_input, job.observation.actor_dim) != (37, 37):
            raise ValueError("Jump Isaac DirectActor37 required")
        if (
            schema["version"] != critic_version
            or schema["width"] != job.model.critic_input
            or sum(math.prod(f["shape"]) for f in schema["fields"]) != schema["width"]
        ):
            raise ValueError("Explicit actual Isaac Critic schema required")
        if list(job.control.effort_limits_nm) != contract["control"]["final_motor_caps"]:
            raise ValueError("Jump Isaac motor caps")
        return JumpTask(job, contract)
    if job.action.version == "jump_unified_raw37_wheel48_v1":
        import math

        from .unified import contract as unified_contract

        contract = job.assets["jump_contract_data"]
        schema = job.assets["jump_critic_schema"]
        if contract != unified_contract() or job.model.action_version != contract["version"]:
            raise ValueError("Jump ABI4冻结合同不同")
        if job.model.interface != contract["version"] or (
            job.model.actor_input,
            job.observation.actor_dim,
        ) != (37, 37):
            raise ValueError("Jump ABI4必须是37D DirectActor")
        if (
            schema["version"] != "jump_unified_dense_causal_CPU_critic_v1"
            or schema["width"] != 6370
            or len(schema["fields"]) != 235
        ):
            raise ValueError("Jump MuJoCo完整因果Critic schema不同")
        if (
            sum(math.prod(f["shape"]) for f in schema["fields"]) != schema["width"]
            or job.model.critic_input != schema["width"]
            or job.model.critic_privileged_version != schema["version"]
        ):
            raise ValueError("Jump Critic字段宽度不一致")
        if list(job.control.effort_limits_nm) != contract["control"]["final_motor_caps"]:
            raise ValueError("Jump各任务限矩必须来自自身合同")
        return JumpTask(job, contract)
    contract = dict(job.assets.get("jump_contract_data", {}))
    schema = job.assets.get("jump_critic_schema", {})
    if job.task != "jump" or job.action.version != VERSION or job.model.action_version != VERSION:
        raise ValueError("Jump只接入显式37D nominal/residual ABI2输出")
    if (job.model.actor_input, job.observation.actor_dim) != (37, 37):
        raise ValueError("Jump单帧37D输入")
    if (contract.get("ABI"), contract.get("output_id"), contract.get("config_id")) != (
        2,
        OUTPUT_ID,
        CONFIG_ID,
    ):
        raise ValueError("Jump ABI/输出/配置身份不同")
    if not contract.get("trajectory_id") or len(contract.get("trajectory_digest", "")) != 64:
        raise ValueError("Jump缺少固定轨迹身份")
    if (
        schema.get("version") != job.model.critic_privileged_version
        or schema.get("width") != job.model.critic_input
    ):
        raise ValueError("Jump完整Critic布局必须有独立版本与显式宽度")
    if sum(field["width"] for field in schema.get("fields", [])) != schema["width"]:
        raise ValueError("Jump Critic具名列不完整")
    return JumpTask(job, contract)
