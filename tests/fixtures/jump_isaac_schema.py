"""Generate only the frozen Critic layout; no PhysX values are simulated here."""

from pathlib import Path

import numpy as np
import torch

from native_stand.artifacts.io import read_json
from native_stand.simulation.jump_build import build
from native_stand.simulation.jump_isaac_critic import RowCritic
from native_stand.simulation.jump_response import Response
from native_stand.simulation.jump_unified import UnifiedV4Execution
from native_stand.tasks.jump.dense_reward import TaskSpec
from native_stand.tasks.jump.isaac_contract import engine_fields
from native_stand.tasks.jump.unified_reward import UnifiedPacket, UnifiedReward


def schema(job, output):
    refs = job.assets["jump_bindings"]
    V4 = read_json(refs["V4"]["path"])
    joints = len(read_json(refs["mechanics"]["path"])["mechanics"]["joint_names"])
    binding = build(Path(output) / "ABI4.so")
    execution = UnifiedV4Execution(binding["path"], V4, Response, 1, library_sha=binding["sha256"])
    execution.reset(torch.arange(1), torch.zeros(1, 6), torch.zeros(1, 6))
    initial = UnifiedPacket(
        dt_s=0.0,
        phase=9,
        owner=1,
        COM_height_m=0.4,
        COM_velocity_m_s=(0, 0, 0),
        root_height_m=0.34,
        tilt_rad=(0, 0),
        omega_rad_s=(0, 0, 0),
        leg_length_m=(0.28, 0.28),
        leg_swing_rad=(0, 0),
        spring_length_m=(0.18, 0.18),
        contact=(True, True),
        tire_clearance_m=(0, 0),
        wheel_normal_force_N=(100, 100),
        contact_slip_m_s=(0, 0),
        actual_motor_Nm=(0,) * 6,
        actual_dq_rad_s=(0,) * 6,
        adopted_target=(0,) * 6,
        force_valid=True,
    )
    from native_stand.tasks.jump import skill_course

    if skill_course.enabled(job):
        spec = skill_course.SkillTaskSpec("ground_active_jump", 6, 21, objective=0)
        reward = UnifiedReward.start(spec, initial, skill_course.reward_config())
    else:
        reward = UnifiedReward.start(
            TaskSpec(kind="ground_active_jump", horizon_s=16, mass_kg=21), initial
        )
    fields = {name: np.empty(shape) for name, (shape, _, _) in engine_fields(joints).items()}
    # Values here are placeholders solely to declare shapes. They never enter
    # a runtime environment, PPO storage, physical readiness or saved model.
    for value in fields.values():
        value.fill(0.0)
    critic = RowCritic(joints, version=job.model.critic_privileged_version)
    critic.pack(
        execution, 0, np.zeros(37), fields, reward, dict(kind="ground_active_jump", seconds=16.0)
    )
    return critic.schema.contract()
