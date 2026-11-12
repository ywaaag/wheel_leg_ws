"""Direct raw Jump action contract; physical guards are separate from gains."""

from __future__ import annotations

import numpy as np

from .simple import contract as observation_contract

VERSION = "jump_direct_raw37_v1"
ABI = 3
INPUT_ID = 0x4A370101
OUTPUT_ID = 0x4A370301
CONFIG_ID = 0x4A050301
GAIN = np.array([0.5, 0.5, 10.0, 0.5, 0.5, 10.0], dtype=np.float32)
CENTER = np.zeros(6, dtype=np.float32)
TARGET_CAP = np.array([1.5, 1.5, 80.0, 1.5, 1.5, 80.0], dtype=np.float32)


def contract():
    c = observation_contract()
    c.update(version=VERSION, ABI=ABI, output_id=OUTPUT_ID, config_id=CONFIG_ID)
    c["output"] = dict(
        kind="direct_raw_position_legs_velocity_wheels",
        Tanh_count=0,
        nominal_trajectory=False,
        phase_dependent_scale=False,
        center=CENTER.tolist(),
        gain=GAIN.tolist(),
        target_guard=TARGET_CAP.tolist(),
        raw_action_bound=None,
        reference_velocity_D=False,
        feedforward_Nm=0.0,
        gain_is_not_range=True,
        reachable_configuration_box=False,
        fixed_center_basis="existing model encoder zero; not SCUT robot joint offsets",
    )
    c["control"].update(
        tau_order="raw PD + existing software spring compensation -> final motor cap40Nm",
        non_Jump_tau_order="preserved inner PD cap40 -> existing compensation -> final cap40",
        target_transport_ms=5,
        phase_bridge=False,
        action_filter=False,
        leg_length_guard_m=[0.126, 0.373],
        crouch_exit=dict(minimum_ms=350, both_leg_length_at_most_m=0.15),
        force_accounting="original V4 aggregate hypothesis unchanged; independent identification pending",
    )
    c["learning"] = dict(
        action="raw Normal sample -> C linear gain -> physical target protection",
        log_prob="original raw sample before protection",
        full_resume_72D=False,
        full_resume_absolute37=False,
        full_resume_nominal_residual37=False,
        initial_std_raw=0.12,
        successful_teacher_required_for_PPO=False,
    )
    return c


def decode_raw(raw):
    a = np.asarray(raw, dtype=np.float32)
    if a.ndim < 1 or a.shape[-1] != 6 or not np.isfinite(a).all():
        raise ValueError("finite six-channel raw action required")
    with np.errstate(over="ignore"):
        requested = CENTER + a * GAIN
    if not np.isfinite(requested).all():
        raise ValueError("physical request overflow")
    return requested, np.clip(requested, -TARGET_CAP, TARGET_CAP)


def labels_from_physical(target):
    a = np.asarray(target, dtype=np.float32)
    if a.ndim < 1 or a.shape[-1] != 6 or not np.isfinite(a).all():
        raise ValueError("finite six-channel physical target required")
    return dict(
        raw=(a - CENTER) / GAIN,
        representable=(abs(a) <= TARGET_CAP).all(-1),
        labels_clipped=False,
        inverse_activation="identity",
    )
