"""Explicit replacement candidate for the preserved 72D/reference draft."""

from __future__ import annotations

import numpy as np

VERSION = "jump_simple_absolute_37_v1"
INPUT_ID = 0x4A370101
OUTPUT_ID = 0x4A150101
CONFIG_ID = 0x4A050101
SCALE = np.array([1.5, 1.5, 80.0, 1.5, 1.5, 80.0], np.float32)
CENTER = np.zeros(6, np.float32)
PHASES = ("CROUCH", "PUSH", "FLIGHT", "LAND", "SETTLE")


def contract():
    fields = []
    for start, n, name, scale, source in [
        (0, 3, "body_gyro", 0.25, "IMU gyro, model mapping"),
        (3, 3, "body_gravity", 1.0, "gyro propagated EKF; acceleration magnitude gate v1"),
        (6, 4, "leg_q", 1 / 1.5, "four actual encoder reports minus fixed zero input reference"),
        (10, 6, "joint_dq", 0.05, "six actual driver velocity reports"),
        (
            16,
            4,
            "adopted_leg_target",
            1 / 1.5,
            "C local adoption cache minus fixed zero input reference",
        ),
        (20, 2, "adopted_wheel_target", 0.01, "same adopted cache"),
        (22, 1, "vx_command", 0.25, "requested vx, never measured velocity"),
        (23, 1, "yaw_command", 0.25, "requested yaw"),
        (24, 1, "handover_height_command", 2.0, "root height command .34m, not leg length"),
        (25, 6, "Jump_BIG", 1.0, "constant [0,0,1,0,0,1]"),
        (31, 5, "phase_onehot", 1.0, "actual C event FSM"),
        (36, 1, "phase_seconds", 1.0, "uint32 monotonic elapsed ms /1000, no modulo/clip"),
    ]:
        fields.append(dict(start=start, count=n, name=name, multiplier=scale, source=source))
    return dict(
        version=VERSION,
        input_id=INPUT_ID,
        output_id=OUTPUT_ID,
        config_id=CONFIG_ID,
        actor_dim=37,
        fields=fields,
        stale_ms=24,
        phase_time_scale_s=1.0,
        input_reference_rad=[0] * 4,
        actor_network=[37, 2048, 1024, 512, 6],
        activation="ELU",
        output=dict(
            tanh_count=1,
            kind="absolute_position_legs_velocity_wheels",
            center=CENTER.tolist(),
            scale=SCALE.tolist(),
            target_guard=SCALE.tolist(),
            feedforward_Nm=0,
            reference_velocity_D=False,
            reachable_configuration_box=False,
        ),
        control=dict(
            physics_ms=1,
            INS_ms=1,
            deployed_Chassis_FSM_PD_ms=5,
            Actor_ms=10,
            target_transport_ms=5,
            leg_PD=[40, 2],
            wheel_Kv=0.2,
            final_motor_caps=[40, 40, 2.44694786377709, 40, 40, 2.44694786377709],
            spring_compensation="same RLControlCompensateSpring, total motor clamp afterward",
        ),
        phases=list(PHASES),
        support=dict(
            kind="per-side measured-driver residual, not LQR command",
            on_N=20,
            off_N=8,
            on_dwell_ms=5,
            off_dwell_ms=8,
            max_age_ms=20,
            unmeasured_internal_load_model="UNKNOWN; production adapter returns unavailable",
            takeoff="both OFF plus specific-force norm <3m/s2; proxy event, truth scored separately",
        ),
        learning=dict(
            action="latent Normal raw sample -> one tanh -> C decode",
            log_prob="raw latent sample, never protected target",
            full_resume_72D=False,
        ),
        production_model=None,
        hardware_accepted=False,
    )
