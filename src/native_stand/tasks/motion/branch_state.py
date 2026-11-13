"""选中完整模型的新目标分支视图；权重/优化器/RNG不重建。"""

from __future__ import annotations

import copy

from native_stand.artifacts.io import json_digest

from .course_strategy import resolve


def confirmation_execution_identity(job):
    """与位置无关的实际执行/观测合同，供课程连续确认绑定。"""
    record = job.as_run_record()

    def semantic(value):
        if isinstance(value, dict):
            return {
                k: semantic(v) for k, v in value.items() if not (k == "path" and "sha256" in value)
            }
        if isinstance(value, list):
            return [semantic(v) for v in value]
        return value

    return json_digest(
        dict(
            version="motion_split_confirmation_execution_v1",
            control=record["control"],
            observation=record["observation"],
            action=record["action"],
            physics={
                k: record["assets"][k]
                for k in (
                    "output_contract_sha256",
                    "runtime_template_sha256",
                    "delayed_execution",
                    "motor_limits",
                    "center_definition_canonical_json_sha256",
                )
            },
            **(
                {"terrain": semantic(record["assets"]["terrain_scene"])}
                if record["assets"].get("terrain_scene")
                else {}
            ),
            **(
                {"wheel_contact": record["assets"]["wheel_contact_dr"]}
                if record["assets"].get("wheel_contact_dr")
                else {}
            ),
        )
    )


def prepare_branch_source(source, job, record):
    from native_stand.artifacts.branch_initialization import COURSE_RECORDS

    if record["version"] not in COURSE_RECORDS:
        from .height_extension import extend_source

        source = extend_source(source, job)
        from .contact_randomization import extend_source as extend_contact

        source = extend_contact(source, job)
    configuration = job.assets["branch_course"]
    if configuration != job.curriculum.stages[0] or configuration["branch"] != record["branch"]:
        raise ValueError("分支课程、目标和来源选择不一致")
    if configuration["confirmation_execution_sha256"] != confirmation_execution_identity(job):
        raise ValueError("分支连续确认执行合同与实际作业不一致")
    value = dict(source)
    strategy = resolve(job.curriculum.version)
    state = strategy.implementation.initial_state(configuration, source["motion_course_state"])
    stage = strategy.stage_config(0, job.runtime.envs, state=state)
    state["assessment_keys"] = stage["cohort_keys"]
    value.update(
        motion_course_state=state,
        reward_refinement_state=strategy.implementation.mixtures(stage),
        symmetry_mix_state=strategy.implementation.mixtures(stage, symmetry=True),
        course_updates=0,
        model_online_updates=record.get(
            "source_model_progress_updates", record["source_model_online_updates"]
        ),
        branch_initialization_record=copy.deepcopy(record),
        source_course_history=dict(
            checkpoint_sha256=record["selected"]["sha256"],
            course=copy.deepcopy(source["motion_course_state"]),
            evidence_is_legacy=True,
        ),
    )
    # 新 run 不能承接旧窗口、部分回合、待发布排名或旧确认。
    for key in (
        "completed_episode_statistics",
        "pending_training_candidate",
        "training_window_start_update",
    ):
        value.pop(key, None)
    return value
