"""Normalized causal fields, separate from byte-exact full checkpoints.

C time fields become relative ages; transport/response queues retain order and
all entries. The final critic width freezes from actual plant state shapes,
not from Motion863/Recovery908 or the preserved raw5850 draft.
"""

from dataclasses import dataclass

import numpy as np

VERSION = "jump_simple_causal_normalized_v1"


@dataclass(frozen=True)
class Field:
    name: str
    shape: tuple
    scale: np.ndarray
    unit: str


class NormalizedCritic:
    def __init__(self, fields):
        self.fields = tuple(fields)
        if len({f.name for f in fields}) != len(fields):
            raise ValueError("duplicate critic field")
        for f in fields:
            if not np.isfinite(f.scale).all() or (np.asarray(f.scale) <= 0).any():
                raise ValueError("finite positive field scale " + f.name)

    @classmethod
    def reuse(cls, fields, previous=None):
        """Reuse immutable field layout without rebuilding JSON contracts each tick."""
        if previous is None:
            return cls(fields)
        if type(previous) is not cls or len(previous.fields) != len(fields):
            raise ValueError("Critic schema drift")
        for a, b in zip(previous.fields, fields, strict=True):
            if (a.name, a.shape, a.unit) != (b.name, b.shape, b.unit) or not np.array_equal(
                a.scale, b.scale
            ):
                raise ValueError("Critic schema drift: " + b.name)
        return previous

    @property
    def width(self):
        return sum(int(np.prod(f.shape)) for f in self.fields)

    def encode(self, values):
        if set(values) != set(f.name for f in self.fields):
            raise ValueError("causal field missing/extra; reject schema drift")
        out = []
        for f in self.fields:
            a = np.asarray(values[f.name], float)
            if a.shape != f.shape or not np.isfinite(a).all():
                raise ValueError("critic shape/finite " + f.name)
            out.append((a / np.asarray(f.scale)).reshape(-1))
        return np.concatenate(out).astype("float32")

    def contract(self):
        return dict(
            version=VERSION,
            width=self.width,
            fields=[
                dict(name=f.name, shape=f.shape, scale=np.asarray(f.scale).tolist(), unit=f.unit)
                for f in self.fields
            ],
            complete_checkpoint_not_replaced=True,
            absolute_ticks=False,
        )


def controller_fields(state, inputs, *, fault_count=13, phase_count=9):
    """All C values that can change next control/event; no privileged Actor data."""
    now = int(inputs.now)
    values = {}
    defs = []

    def add(name, value, scale=1.0, unit="dimensionless"):
        a = np.asarray(value, float)
        values[name] = a
        defs.append(Field(name, a.shape, np.asarray(scale), unit))

    def elapsed(stamp):
        return ((now - int(stamp)) & 0xFFFFFFFF) * 0.001

    for name in ["phase", "owner", "fault", "status"]:
        sizes = {"phase": phase_count, "owner": 5, "fault": fault_count, "status": 8}
        add("C." + name, np.eye(sizes[name])[int(getattr(state, name))])
    for name in [
        "initialized",
        "last_key",
        "entry_valid",
        "adopted_valid",
        "command_valid",
        "first_hold",
        "airborne",
        "success",
        "stable_valid",
    ]:
        add("C." + name, int(getattr(state, name)))
    enabled_age = {
        "phase_tick": True,
        "jump_tick": 1 <= state.phase <= 6,
        "last_tick": bool(state.initialized),
        "wait_tick": state.status == 1,
        "entry_tick": bool(state.entry_valid),
        "adopted_tick": bool(state.adopted_valid),
        "command_tick": bool(state.command_valid),
        "command_snapshot_tick": bool(state.command_valid),
        "stable_tick": bool(state.stable_valid),
        "event_tick": True,
    }
    for name, enabled in enabled_age.items():
        add("C.age." + name, elapsed(getattr(state, name)) if enabled else 0, 1.0, "seconds")
    for name in ["contact", "candidate", "contact_valid", "off_valid"]:
        add("C." + name, list(getattr(state, name)))
    add(
        "C.candidate_age",
        [elapsed(state.candidate_tick[k]) if state.candidate[k] < 2 else 0 for k in range(2)],
        0.1,
        "seconds",
    )
    add(
        "C.off_age",
        [elapsed(state.off_tick[k]) if state.off_valid[k] else 0 for k in range(2)],
        0.1,
        "seconds",
    )
    # Only equality/order of serials affects control; raw absolute IDs do not.
    add(
        "C.command_identity",
        [
            state.command_epoch == state.epoch,
            state.command_phase_sequence == state.phase_sequence,
            state.command_previous_sequence == state.adopted_sequence,
            ((int(state.command_sequence) - int(state.adopted_sequence)) & 0xFFFFFFFF) < 0x80000000,
        ],
    )
    target_scale = [1.5, 1.5, 100, 1.5, 1.5, 100]
    for name in ["adopted", "mean", "requested", "protected_target", "pending"]:
        add(
            "C." + name,
            list(getattr(state, name)),
            1 if name == "mean" else target_scale,
            "normalized_mean" if name == "mean" else "mixed_rad_rad_s",
        )
    add("C.support_N", list(state.support_N), 250.0, "N")
    flags = [
        "enabled_mode",
        "key_down",
        "upstream_owner",
        "legacy_jump",
        "motion_entry",
        "motion_success",
        "support_mask",
        "support_model_valid",
        "feedback_mask",
        "imu_valid",
        "adopted_valid",
    ]
    for name in flags:
        add("input." + name, getattr(inputs, name))
    scales = {
        "gyro": 4.0,
        "gravity": 1.0,
        "q": 1.5,
        "dq": 100.0,
        "command": [4, 4, 0.5],
        "upstream_target": target_scale,
        "accel": 9.81,
        "driver_tau": 40.0,
        "internal_tau": 40.0,
        "support_J": 0.5,
        "leg_length": 0.5,
        "leg_spread": np.pi,
    }
    for name, scale in scales.items():
        add("input." + name, list(getattr(inputs, name)), scale, "explicit SI physical field")
    for name in ["sensor_tick", "imu_tick", "support_tick"]:
        add("input.age." + name, elapsed(getattr(inputs, name)), 0.1, "seconds")
    add("input.feedback_ages", [elapsed(t) for t in inputs.feedback_tick], 0.1, "seconds")
    return values, defs


def extend_fields(values, definitions, extras):
    """Explicit plant/estimator/reward/FIFO field records; never implicit scale1.

    extras[name]=(array,scale,SI unit). Caller MUST include response actual/modal,
    all chronological queues, filter/noise state and DR parameters, physical
    state, reward once/event memory. A final environment binds this exact list.
    """
    out = dict(values)
    defs = list(definitions)
    for name, (value, scale, unit) in extras.items():
        if name in out or not unit:
            raise ValueError("unique named unit required")
        a = np.asarray(value, float)
        out[name] = a
        defs.append(Field(name, a.shape, np.asarray(scale), unit))
    return out, defs


COUNTEREXAMPLES = {
    "pending_command_and_age": "same q/dq and adopted target, different queued target/age -> different next PD adoption",
    "contact_candidate_and_dwell": "same current force, prior ON vs OFF or different candidate tick -> different phase event",
    "reward_once_memory": "same flight state, paid vs unpaid apex -> different reward",
    "response_actual_and_FIFO": "same last PD command, different modal motor response / oldest queued command -> different next torque",
    "held_filtered_feedback": "same true q/dq, different held quantized feedback or velocity filter -> different next PD",
    "plant_DR_and_RNG": "same controller state, different mass/NET/noise state -> different next dynamics",
    "last_key_and_cooldown_age": "same key-down, different prior edge/cooldown -> new request vs no request",
}
