"""正式Motion环境工厂：显式课程、控制和时序依赖；不改冻结模块全局。"""

from __future__ import annotations

import copy

from native_stand.artifacts.io import file_sha256, read_json
from native_stand.tasks.motion.action_contract import ActionDecoder
from native_stand.tasks.motion.course_strategy import resolve
from native_stand.tasks.motion.spec import build_task


def control_router(job, source, prefix):
    from .handover import Handover

    task = build_task(job)
    student = ActionDecoder.from_contract(source["output_contract"])
    if student.center != task.center or student.wheel_scale != job.action.wheel_scale_rad_s:
        raise ValueError("冻结Student输出合同与运行配置不同")
    if source["action_version"] != job.action.version:
        raise ValueError("Student状态与动作合同不一致")
    path = job.assets.get("prefix_output_contract")
    if path is None:
        raise ValueError("正式Teacher必须提供独立输出合同")
    if file_sha256(path) != job.assets["prefix_output_contract_sha256"]:
        raise ValueError("Teacher输出合同SHA不一致")
    teacher = ActionDecoder.from_contract(read_json(path))
    return Handover(
        student, teacher, prefix, fixed_prefix=not resolve(job.curriculum.version).legacy
    )


def environment(spec, role, *, envs, job, source, prefix, corrections, measurement=None):
    # Engine imports are strictly after the owner's AppLauncher.
    import onpolicy
    import runtime_support
    import torch
    import transient_events

    from native_stand.command_v6.runtime.configure import apply
    from native_stand.command_v6.runtime.env import CommandEnvCfg
    from native_stand.students.performance import settings
    from native_stand.students.standing_survival_reward import configuration as clock_reward
    from native_stand.tasks.motion.progressive_episodes import episode_class
    from native_stand.tasks.motion.scale_corrections import corrected_environment_class

    from .delayed_execution import dr_factory
    from .fullchain import subclass
    from .motion_quality import QualityMotionStudentEnv
    from .transient_events import event_class

    strategy = resolve(job.curriculum.version)
    maneuver = strategy.maneuver
    stage = spec["curriculum"]["stages"][0]
    cfg = apply(CommandEnvCfg(), spec, envs=envs, mode="evaluation")
    for key in ("effort_limit", "effort_limit_sim"):
        setattr(cfg.robot_cfg.actuators["leg_motors"], key, 67.0)
    if job.assets.get("motor_limits"):
        for key in ("effort_limit", "effort_limit_sim"):
            setattr(cfg.robot_cfg.actuators["wheel_motors"], key, job.control.effort_limits_nm[2])
    performance = settings(spec)
    cfg.motion_hotpath = performance.get("motion_hotpath", False)
    cfg.motion_runtime_reduction = performance.get("motion_runtime_reduction", False)
    if performance.get("contact_history_disabled"):
        cfg.scene.contact_forces.history_length = 0
    cfg.recovery_dr_materials = True
    if spec.get("motion_comprehensive_dr"):
        cfg.scene.env_spacing = 3.0
        cfg.scene.filter_collisions = False
    cls = QualityMotionStudentEnv
    if spec.get("motion_height_dr"):
        from .height import subclass as height

        cls = height(
            cls, spec["motion_height_dr"], reward_owner="maneuver" if maneuver else "fixed_height"
        )
    episodes = onpolicy.Episodes if strategy.legacy else episode_class(onpolicy.Episodes)
    if job.diagnostics.get("movement_components_version") is not None:
        from native_stand.tasks.motion.movement_evidence import episode_class as movement

        episodes = movement(
            episodes, progression_gate=getattr(strategy.implementation, "PROGRESSION", None)
        )
    if maneuver:
        from native_stand.tasks.motion.maneuver_statistics import RESPONSE_VERSION, VERSION
        from native_stand.tasks.motion.maneuver_statistics import episode_class as maneuver_episodes

        episodes = maneuver_episodes(
            episodes,
            version=RESPONSE_VERSION if strategy.response_checked else VERSION,
        )
    timing = runtime_support.DR
    if job.assets.get("delayed_execution") is not None:
        timing = dr_factory(timing, runtime_support.noise, job.assets["delayed_execution"])
    if job.assets.get("motor_limits"):
        from .motion48_execution import dr_factory as motor_dr

        timing = motor_dr(timing, job.assets["motor_limits"])
    if job.assets.get("wheel_contact_dr") is not None:
        from .contact_materials import dr_factory as contact_dr

        timing = contact_dr(timing, job.assets["wheel_contact_dr"], runtime_support.noise)
    events = (
        transient_events.Events
        if strategy.legacy
        else event_class(transient_events.Events, runtime_support.noise)
    )
    router = control_router(job, source, prefix)
    cls = subclass(
        cls,
        {
            **spec["motion_identified_response"],
            "execution_configuration": spec["motion_timing"]["configuration"],
            "armature_level": spec["motion_timing"]["stage"],
            **(
                {"execution_optimization": job.assets["motion_execution_optimization"]}
                if job.assets.get("motion_execution_optimization") is not None
                else {}
            ),
            **(
                {"motor_effort_caps": job.control.effort_limits_nm}
                if job.assets.get("motor_limits")
                else {}
            ),
            **(
                {"terrain_scene": job.assets["terrain_scene"]}
                if job.assets.get("terrain_scene")
                else {}
            ),
        },
        spec["student_training"]["height_dr_state"],
        handover=router,
        dr_factory=timing,
        episodes_factory=episodes,
        events_factory=events,
        terminal_critic_owner=(
            "outer" if maneuver and job.model.critic_input in (877, 881) else "fullchain"
        ),
        measurement=measurement,
    )
    cls = corrected_environment_class(cls, corrections)
    if maneuver:
        from .maneuver_planned import environment_class as planned_env
        from .maneuver_runtime import environment_class as maneuver_env

        cls = maneuver_env(cls, corrections, reward_config=job.reward.fields["maneuver"])
        if job.model.critic_input in (877, 881):
            cls = planned_env(cls)
    recovery = copy.deepcopy(spec["recovery"])
    recovery.pop("support_record_reward", None)
    recovery.pop("drawback_guidance", None)

    class ForbiddenTeacher(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("stand-policy execution forbidden")

    env = cls(
        cfg,
        recovery_config=recovery,
        pose_library=read_json(recovery["pose_library"]),
        stand_policy=ForbiddenTeacher(),
        stage=stage,
        command_height=0.36,
        reward_config={},
        schedule_s=spec["curriculum"]["schedule_s"],
        student_role=role,
        interface_version=spec["interface_version"],
        student_reward_configuration=clock_reward(),
        critic_privileged_version="standing_privileged49_v1",
        student_command_height=spec["student_training"].get("target_height_m", 0.34),
        motion_training_config=spec["student_training"],
        runtime_stage=stage,
    )
    if maneuver:
        from .maneuver_runtime import attach

        attach(
            env,
            corrections,
            reward_config=job.reward.fields["maneuver"],
            command_program_version=stage.get("command_program_version"),
        )
        env.planned_value_ready = job.model.critic_input in (877, 881)
        env.extended_height_quality = (
            job.action.version == "motion_multiheight020_support_center_015_80_v2"
        )
    env.curriculum_training = False
    env.motion_height_range_m = router.student_decoder.center.height_range_m
    env.student_performance = performance
    if spec.get("motion_confirmation") or spec.get("motion_training", {}).get("early_stop_policy"):
        env.motion_absolute_protocol = spec["motion_assessment"]
        env.motion_forward_course = True
        env.motion_course_assessment_version = spec["motion_assessment"]["version"]
    if spec.get("motion_symmetry"):
        from native_stand.students.motion_parallel_assessment import safety_protocol

        env.motion_symmetry_protocol = safety_protocol(spec)
    return env
