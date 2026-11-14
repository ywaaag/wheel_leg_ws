"""CPU-only audit of the real ABI4 motor boundary, without a physics rollout.

This is an offline diagnostic, not a controller or a training entry point.
The target correction below is an audit hypothesis. It does not alter firmware,
adopted targets, training labels, V4 state or any running worker.
"""

from __future__ import annotations

import argparse
import ctypes as C
from pathlib import Path

import numpy as np

from native_stand.artifacts.io import file_sha256, write_json
from native_stand.simulation.jump_build import build
from native_stand.simulation.jump_shared_c import Frame, array, declared_input
from native_stand.simulation.jump_unified import UnifiedJump

LEGS = [0, 1, 3, 4]
TARGET_GUARD = np.array([1.5, 1.5, 80, 1.5, 1.5, 80], dtype=np.float32)


def feedback(q, dq, length, spread):
    """Explicit synthetic input for a motor unit call; not a physical scene."""
    return declared_input(
        now=100,
        gyro=[0, 0, 0],
        gravity=[0, -1, 0],
        q=q,
        dq=dq,
        command=[0, 0, 0.34],
        adopted=[0] * 6,
        adopted_sequence=1,
        sensor_tick=100,
        imu_tick=100,
        feedback_ticks=[100] * 6,
        accel=[0, -9.81, 0],
        driver_tau=[0] * 4,
        internal_tau=[0] * 4,
        support_J=np.tile(np.eye(2), (2, 1)),
        support_tick=100,
        support_valid=0,
        leg_length=length,
        leg_spread=spread,
        motion_entry=False,
    )


def capture(control, sensor_input, target):
    target = np.ascontiguousarray(target, dtype=np.float32)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("finite six-channel physical target required")
    frame = Frame()
    if control.lib.RLJumpMotorCommand(C.byref(sensor_input), target.ctypes.data, C.byref(frame)):
        raise ValueError("real C motor unit call rejected input")
    return {
        name: array(getattr(frame, name)).tolist()
        for name in (
            "p_nm",
            "d_nm",
            "pd_nm",
            "limited_nm",
            "compensation_motor_nm",
            "final_motor_nm",
            "final_model_nm",
            "adopted",
        )
    }


def compare_same_input(control, sensor_input, target):
    """Test only the instantaneous leg-PD boundary, before V4/target queues."""
    before = bytes(sensor_input), bytes(control.states)
    target = np.asarray(target, dtype=np.float32)
    original = capture(control, sensor_input, target)
    at_q = np.zeros(6, dtype=np.float32)
    at_q[LEGS] = array(sensor_input.q)
    # Remove D from this zero-P input to read only the real C compensation.
    zero_velocity = type(sensor_input).from_buffer_copy(bytes(sensor_input))
    zero_velocity.dq[:] = [0] * 6
    zero_pd = capture(control, zero_velocity, at_q)
    correction = np.asarray(zero_pd["final_model_nm"], dtype=np.float32)[LEGS] / 40.0
    corrected_target = target.copy()
    corrected_target[LEGS] -= correction
    corrected = capture(control, sensor_input, corrected_target)
    motion_leg_boundary = np.clip(np.asarray(original["pd_nm"])[LEGS], -40, 40)
    error = np.asarray(corrected["final_model_nm"])[LEGS] - motion_leg_boundary
    if before != (bytes(sensor_input), bytes(control.states)):
        raise AssertionError("a read-only motor unit call changed input or FSM state")
    return dict(
        q=array(sensor_input.q).tolist(),
        dq=array(sensor_input.dq).tolist(),
        leg_length_m=array(sensor_input.leg_length).tolist(),
        leg_spread_rad=array(sensor_input.leg_spread).tolist(),
        target=target.tolist(),
        original=original,
        zero_PD=zero_pd,
        correction_rad=correction.tolist(),
        corrected_target=corrected_target.tolist(),
        corrected_target_inside_guard=bool((abs(corrected_target) <= TARGET_GUARD).all()),
        corrected=corrected,
        motion_same_input_no_C_leg_boundary_nm=motion_leg_boundary.tolist(),
        corrected_leg_max_error_nm=float(abs(error).max()),
        C_motor_calls=3,
        input_and_FSM_unchanged=True,
    )


def audit(control):
    cases = {}
    q = np.array([-0.13, -0.15, 0.13, -0.15], dtype=np.float32)
    target = np.array([-0.13, -0.15, 0, 0.13, -0.15, 0], dtype=np.float32)
    cases["zero_PD"] = compare_same_input(
        control, feedback(q, [0] * 6, [0.28] * 2, [2.07742691] * 2), target
    )
    cases["asymmetric_with_D"] = compare_same_input(
        control,
        feedback(q, [0.2, -0.3, 0, 0.4, -0.5, 0], [0.24, 0.31], [1.4, 2.2]),
        target + np.array([0.03, -0.02, 0, -0.04, 0.01, 0], dtype=np.float32),
    )
    cases["saturated_representable"] = compare_same_input(
        control,
        feedback([-0.8, 0.8, 0.8, 0.8], [0] * 6, [0.28] * 2, [2.07742691] * 2),
        [0.8, -0.8, 0, -0.8, -0.8, 0],
    )
    cases["outside_target_guard"] = compare_same_input(
        control, feedback(q, [0] * 6, [0.28] * 2, [2.07742691] * 2), [-1.49] * 6
    )
    original = cases["zero_PD"]
    stale = feedback(q, [0] * 6, [0.31, 0.24], [1.4, 2.2])
    stale_frame = capture(control, stale, original["corrected_target"])
    stale_error = float(abs(np.asarray(stale_frame["final_model_nm"])[LEGS]).max())
    for case in cases.values():
        if case["corrected_leg_max_error_nm"] > 1e-4:
            raise AssertionError("same-input leg boundary did not match")
    zero = np.asarray(original["original"]["pd_nm"])
    if zero.any() or stale_error < 0.1:
        raise AssertionError("zero-PD or stale-condition counterexample missing")
    return dict(
        status="CPU_BOUNDARY_VERIFIED",
        ABI=control.ABI_VERSION,
        cases=cases,
        changed_geometry_with_stale_correction=dict(
            final_model_nm=stale_frame["final_model_nm"], max_leg_error_nm=stale_error
        ),
        scope="same-input real C leg boundary only; not queued execution or capability equivalence",
        C_motor_calls=sum(c["C_motor_calls"] for c in cases.values()) + 1,
        new_physics_ticks=0,
        new_GPU_runs=0,
        new_PPO_updates=0,
        new_BC_updates=0,
        V4_steps=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binding = build(args.output / "control.so")
    control = UnifiedJump(binding["path"], 1, library_sha=binding["sha256"])
    report = audit(control)
    report["build"] = binding
    report["script_sha256"] = file_sha256(__file__)
    write_json(args.output / "receipt.json", report)
    print(report["status"], "C_motor_calls=", report["C_motor_calls"])


if __name__ == "__main__":
    main()
