"""Frozen V4 causal Critic prefix, independent of historical course scripts."""

EXTRA_FIELDS = (
    ("calibration_bias_model_rad", 4),
    ("target_delay_ms", 1),
    ("target_history6_to20ms", 90),
    ("actual_motor_strength", 6),
    ("ground_pitch_roll", 2),
    ("ground_normal_world", 3),
    ("effective_restitution", 2),
    ("push_clock_context", 2),
    ("scheduled_velocity_impulse", 3),
    ("true_joint_q", 6),
    ("true_joint_dq", 6),
    ("actor_leg_packet_age_ms", 1),
    ("effective_true_command", 3),
)

# 562 base + 129 DR/queue + 168 dq context + 4 physical HF filter states.
DIM = 562 + sum(width for _, width in EXTRA_FIELDS) + 168 + 4
