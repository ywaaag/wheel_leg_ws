"""Recovery lifecycle/reward and explicitly versioned causal Critic inputs.

The Motion863 reward/command-phase memories are not valid for Swing. This
Recovery863 prefix is a NEW layout, not a padded Motion Critic or a resume.
"""

from __future__ import annotations

import numpy as np

CRITIC_VERSION = "recovery_causal_execution878_v2"
BASE_FIELDS = (
    ("actor", 31),
    ("true_frame", 16),
    ("q", 6),
    ("dq", 6),
    ("requested", 6),
    ("adopted", 6),
    ("target_history20", 120),
    ("joint_history10", 120),
    ("motor_history24", 144),
    ("NET_history24", 144),
    ("physical_packets3", 48),
    ("fitted_parameters", 26),
    ("fitted_reference", 4),
    ("observer", 16),
    ("native_execution", 20),
    ("physical_profile", 32),
    ("kinematics", 19),
    ("wheel_actual", 2),
    ("sensor_ages", 8),
    ("PD_gains", 10),
    ("motor_caps", 6),
    ("wheel_timing", 4),
    ("previous_policy_target", 6),
    ("previous_policy_motor", 6),
    ("quality_timers", 7),
    ("phase_onehot", 5),
    ("task_onehot", 3),
    ("encoder_bias", 4),
    ("motor_strength", 6),
    ("passive_friction", 6),
    ("armature", 6),
    ("target_history_ages20", 20),
)
assert sum(w for _, w in BASE_FIELDS) == 863
EXTRA_FIELDS = (
    ("initial_root_height", 1),
    ("height_progress", 1),
    ("chassis_clearance", 1),
    ("closure", 1),
    ("restricted_contact", 1),
    ("stage_onehot", 3),
    ("entry_latched", 1),
    ("entry_age", 1),
    ("hold_age", 1),
    ("student_age", 1),
    ("time_remaining", 1),
    ("entry_dwell_progress", 1),
    ("initialization_complete", 1),
)
assert sum(w for _, w in EXTRA_FIELDS) == 15


def critic_tensor(values, extra):
    if set(values) != {k for k, _ in BASE_FIELDS} or set(extra) != {k for k, _ in EXTRA_FIELDS}:
        raise ValueError("All named causal fields required; no zero padding")
    rows = []
    for collection, fields in [(values, BASE_FIELDS), (extra, EXTRA_FIELDS)]:
        for name, width in fields:
            a = np.asarray(collection[name], np.float32)
            if a.shape != (width,) or not np.isfinite(a).all():
                raise ValueError("Invalid Critic field " + name)
            rows.append(a)
    return np.concatenate(rows)


def huber(x):
    x = np.abs(x)
    return np.where(x < 1, 0.5 * x * x, x - 0.5)


class RecoveryLifecycle:
    def __init__(self, initial_height, max_s=23.0, student_start_s=0.0):
        self.h0 = float(initial_height)
        self.max_s = max_s
        self.student_start_s = student_start_s
        self.entry_s = None
        self.dwell = 0.0
        self.hold = 0.0
        self.success = False
        self.last_phi = 0.0
        self.last_target = None
        self.last_delta = np.zeros(6)
        self.elapsed = 0.0
        self.total_reward = 0.0
        self.low = np.zeros(4)

    def observe(
        self,
        *,
        time_s,
        state,
        target,
        motor,
        closure,
        restricted=0.0,
        terminal=False,
        truncated=False,
        student=True,
        hp10=None,
        angular_velocity=None,
        dt_s=0.01,
    ):
        dt = float(dt_s)
        if not 0 <= dt <= 0.010000001:
            raise ValueError("Measured physical interval required")
        s = np.asarray(state)
        target = np.asarray(target)
        motor = np.asarray(motor)
        if not np.isfinite(np.r_[s, target, motor, closure, restricted]).all():
            raise ValueError("Nonfinite Recovery lifecycle")
        height = s[3]
        pitch = s[4]
        roll = s[5]
        dual = bool((s[9:11] > 2).all())
        eligible = (
            0.30 <= height <= 0.40
            and abs(pitch) <= 0.1
            and abs(roll) <= 0.1
            and dual
            and closure <= 0.005
        )
        first = self.entry_s is None
        self.dwell = self.dwell + dt if eligible else 0.0
        entered = first and self.dwell >= 0.2 - 1e-9
        if entered:
            self.entry_s = float(time_s)
        age = 0 if self.entry_s is None else max(0, time_s - self.entry_s)
        hold_quality = (
            0.30 <= height <= 0.40
            and abs(pitch) <= 0.18
            and abs(roll) <= 0.18
            and dual
            and closure <= 0.005
        )
        self.hold = self.hold + dt if self.entry_s is not None and hold_quality and student else 0.0
        succeeded = not self.success and self.hold >= 10.0 - 1e-9
        if succeeded:
            self.success = True
        progress = float(np.clip((height - self.h0) / max(0.34 - self.h0, 0.02), 0, 1))
        broad = np.exp(-((pitch / 0.5) ** 2 + (roll / 0.35) ** 2))
        narrow = np.exp(-((pitch / 0.1) ** 2 + (roll / 0.05) ** 2))
        posture = 0.4 * broad + 0.6 * narrow
        phi = progress * broad
        # True termination absorbs potential; truncation bootstraps successor.
        next_phi = 0.0 if terminal else phi
        potential = 2 * (np.exp(-dt / 5) * next_phi - self.last_phi)
        self.last_phi = phi
        tighten = float(np.clip((age - 0.5) / 1.5, 0, 1)) if self.entry_s is not None else 0.0
        delta = np.zeros(6) if self.last_target is None else target - self.last_target
        dd = delta - self.last_delta
        self.last_target = target.copy()
        self.last_delta = delta
        scale = np.array([0.05, 0.05, 10, 0.05, 0.05, 10])
        caps = np.array([40, 40, 2.44694786377709, 40, 40, 2.44694786377709])
        height_kernel = 0.4 * np.exp(-(((height - 0.34) / 0.1) ** 2)) + 0.6 * np.exp(
            -(((height - 0.34) / 0.025) ** 2)
        )
        hp = np.zeros(4) if hp10 is None else hp10
        rotation = abs(s[2]) if angular_velocity is None else np.linalg.norm(angular_velocity)
        terms = dict(
            supported_posture=3 * posture * (0.25 + 0.75 * dual),
            height_precision=height_kernel * broad,
            mirror_precision=0.5 * np.exp(-((s[6] / 0.05) ** 2 + (s[8] / 0.1) ** 2)) * dual,
            rotation_cost=-0.1 * float(huber(max(rotation - (0.2 if tighten else 2), 0))),
            motor_effort_cost=-0.02 * float(((motor / caps) ** 2).mean()),
            physical_target_delta_cost=-0.002 * float(((delta / scale) ** 2).mean()),
            target_second_difference_cost=-tighten
            * (
                0.005 * float(((dd[[0, 1, 3, 4]] / 0.05) ** 2).mean())
                + 0.001 * float(((dd[[2, 5]] / 10) ** 2).mean())
            ),
            standing_contact_deficit_cost=-0.5 * tighten * (not dual),
            actual_dq_HP10_cost=-0.1 * tighten * float(huber(np.asarray(hp) / 0.2).mean()),
            saturation_cost=-0.1 * tighten * float((abs(motor) >= caps * 0.98).mean()),
            drift_cost=-0.05 * tighten * float(huber(max(abs(s[0]) - 0.3, 0))),
            pre_entry_time_cost=-0.05 * (self.entry_s is None),
        )
        reward = dt * sum(terms.values()) + potential + float(entered) + 10 * float(succeeded)
        self.elapsed = time_s
        self.total_reward += reward
        done = bool(terminal or succeeded or time_s >= self.max_s)
        return dict(
            reward=float(reward),
            density=terms,
            potential=potential,
            entered=entered,
            success=succeeded,
            terminated=bool(terminal or succeeded),
            truncated=bool(not terminal and not succeeded and (truncated or time_s >= self.max_s)),
            done=done,
            entry_s=self.entry_s,
            first_entry_age_s=age,
            student_control=bool(student),
            entry_eligible=eligible,
            hold_quality=hold_quality,
        )

    def extra(self, *, height, clearance, closure, restricted, initialized=True):
        phase = 0 if self.entry_s is None and height < 0.27 else 1 if self.entry_s is None else 2
        return dict(
            initial_root_height=np.array([self.h0 / 0.5]),
            height_progress=np.array(
                [np.clip((height - self.h0) / max(0.34 - self.h0, 0.02), 0, 1)]
            ),
            chassis_clearance=np.array([clearance / 0.5]),
            closure=np.array([closure / 0.005]),
            restricted_contact=np.array([restricted / 100]),
            stage_onehot=np.eye(3)[phase],
            entry_latched=np.array([self.entry_s is not None]),
            entry_age=np.array([0 if self.entry_s is None else (self.elapsed - self.entry_s) / 10]),
            hold_age=np.array([self.hold / 10]),
            student_age=np.array([max(0, self.elapsed - self.student_start_s) / 23]),
            time_remaining=np.array([max(0, self.max_s - self.elapsed) / 23]),
            entry_dwell_progress=np.array([min(1, self.dwell / 0.2)]),
            initialization_complete=np.array([initialized]),
        )


def qualified_label_source(case_quality, state, *, student_state=False):
    if case_quality is not True:
        return "UNKNOWN"
    s = np.asarray(state)
    return (
        "C0_NEIGHBORHOOD"
        if (0.30 <= s[3] <= 0.40 and abs(s[4]) <= 0.1 and abs(s[5]) <= 0.1 and (s[9:11] > 2).all())
        else "UNKNOWN"
    )
