"""Versioned raw37 single-actor contract with an explicit balance encoding."""

from __future__ import annotations

import numpy as np

from .direct import contract as direct_contract
from .direct import decode_raw as decode_raw
from .direct import labels_from_physical as labels_from_physical

VERSION = "jump_unified_raw37_wheel48_v1"
ABI = 4
INPUT_ID = 0x4A370102
OUTPUT_ID = 0x4A370401
CONFIG_ID = 0x4A050401
BALANCE = 9
TASK_FLAGS = np.array([0, 0, 1, 0, 0, 1], np.float32)


def contract():
    c = direct_contract()
    c.update(version=VERSION, ABI=ABI, input_id=INPUT_ID, output_id=OUTPUT_ID, config_id=CONFIG_ID)
    c["control"]["final_motor_caps"] = [40, 40, 4.8, 40, 40, 4.8]
    c["control"]["balance_command_envelope"] = dict(
        vx_m_s=2.0, yaw_rad_s=1.0, height_m=[0.30, 0.38]
    )
    c["control"]["active_request"] = (
        "stationary only; moving jumps require a later explicit extension"
    )
    c["control"]["completion"] = (
        "same Jump actor; own support/posture dwell1s -> BALANCE; no Motion handover"
    )
    c["balance_encoding"] = dict(C_phase=BALANCE, onehot=[0] * 5, seconds=0.0, owner="Jump")
    c["phases"] = ["BALANCE", "CROUCH", "PUSH", "FLIGHT", "LAND", "SETTLE"]
    c["fields"][-2]["source"] = "C event FSM; BALANCE is the explicit all-zero code"
    c["fields"][-1]["source"] = (
        "C phase age seconds; BALANCE has no elapsed-time dependence and encodes0"
    )
    c["learning"]["old_direct_ABI3_full_resume"] = False
    c["learning"]["Motion_source"] = (
        "supervision only; physical targets converted to raw, no runtime teacher"
    )
    return c


def from_motion_observation(x):
    a = np.asarray(x, np.float32)
    if a.shape[-1] != 31 or not np.isfinite(a).all():
        raise ValueError("finite Motion31 observations required")
    out = np.zeros((*a.shape[:-1], 37), np.float32)
    out[..., :31] = a
    out[..., 25:31] = TASK_FLAGS
    return out
