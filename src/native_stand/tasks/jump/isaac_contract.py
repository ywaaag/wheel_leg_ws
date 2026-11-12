"""Isaac training identity, sharing the unchanged ABI4 physical Actor contract."""

TRAINING = "jump_isaac_ABI4_raw37_wheel48_v1"
CRITIC = "jump_isaac_full_execution_critic_v1"


def engine_fields(joints):
    """Named actual engine readbacks; schema declarations are never runtime padding."""
    return {
        "root_relative_state": ((13,), 1.0, "m, xyzw, world m/s, world rad/s"),
        "all_joint_q_dq_ddq": ((3, joints), 10.0, "rad,rad/s,finite-difference rad/s2"),
        "COM_height_velocity": ((4,), 1.0, "m,m/s"),
        "leg_length_swing_spring": ((6,), 1.0, "m,rad,m"),
        "closure_residuals": ((4,), 0.005, "m"),
        "wheel_normal_slip_clearance": ((6,), 1.0, "N,m/s,m"),
        "motor_NET_passive": ((3, 6), 40.0, "Nm; distinct actual effort and external loads"),
        "root_mass_COM": ((4,), 1.0, "kg,m; actual native parameter readback"),
        "material_min_max": (
            (2, 3),
            1.0,
            "actual min/max PhysX static/dynamic friction and restitution over shapes",
        ),
        "elapsed_request": ((3,), 10.0, "elapsed_s,request_time_s,active_request_enabled"),
    }
