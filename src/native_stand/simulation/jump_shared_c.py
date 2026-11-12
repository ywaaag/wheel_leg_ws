"""Host ABI to actual Jump firmware C; no alternative Python FSM/PD.

Batch calls execute the identical scalar C core for every environment. This is
CPU C even when an upstream policy/physics engine uses a GPU. Compilation and
source/SHA preflight belong to experiments.jump_shared_prepare, never runtime.
"""

from __future__ import annotations

import base64
import ctypes as C
import sys

import numpy as np

ABI = 1
INPUT_ID = 0x4A370101
OUTPUT_ID = 0x4A150101
CONFIG_ID = 0x4A050101
U = C.c_uint32
F = C.c_float
P = C.c_void_p


def fields(words="", floats=()):
    out = []
    for word in words.split():
        name, _, size = word.partition(":")
        out.append((name, U * int(size) if size else U))
    for word in floats:
        name, _, size = word.partition(":")
        out.append((name, F * int(size) if size else F))
    return out


class Config(C.Structure):
    _fields_ = fields(
        "abi input_id output_id config_id enabled model_available model_id model_input_id model_output_id"
    )


class Input(C.Structure):
    _fields_ = fields(
        "now sensor_tick imu_tick feedback_mask imu_valid feedback_tick:6 enabled_mode key_down upstream_owner legacy_jump motion_entry motion_success support_tick support_mask support_model_valid adopted_valid adopted_sequence",
        [
            "gyro:3",
            "gravity:3",
            "q:4",
            "dq:6",
            "command:3",
            "upstream_target:6",
            "accel:3",
            "driver_tau:4",
            "internal_tau:4",
            "support_J:8",
            "leg_length:2",
            "leg_spread:2",
        ],
    )


class State(C.Structure):
    _fields_ = fields(
        "abi phase owner fault status epoch phase_tick last_tick jump_tick initialized last_key request_id wait_tick entry_tick entry_valid phase_sequence adopted_sequence adopted_tick adopted_valid command_sequence command_tick command_snapshot_tick command_epoch command_phase_sequence command_previous_sequence command_valid first_hold airborne success stable_tick stable_valid contact:2 candidate:2 candidate_tick:2 contact_valid:2 off_tick:2 off_valid:2 event_tick event_id",
        ["adopted:6", "mean:6", "requested:6", "protected_target:6", "pending:6", "support_N:2"],
    )


class Frame(C.Structure):
    _fields_ = fields(
        "owner phase fault adopted_sequence adopted_tick clipped_mask",
        [
            "requested:6",
            "protected_target:6",
            "adopted:6",
            "p_nm:6",
            "d_nm:6",
            "pd_nm:6",
            "limited_nm:6",
            "compensation_motor_nm:4",
            "final_motor_nm:4",
            "wheel_current_raw:2",
            "final_model_nm:6",
        ],
    )


def copy_struct(value):
    return type(value).from_buffer_copy(bytes(value))


def array(value):
    return np.ctypeslib.as_array(value).copy()


def declared_input(
    *,
    now,
    gyro,
    gravity,
    q,
    dq,
    command,
    adopted,
    adopted_sequence,
    sensor_tick,
    imu_tick,
    feedback_ticks,
    accel,
    driver_tau,
    internal_tau,
    support_J,
    support_tick,
    support_valid,
    leg_length,
    leg_spread,
    motion_entry,
    mode=True,
    key=False,
    upstream_owner=1,
    motion_success=True,
):
    """Explicit deployable values. No body velocity/contact/clearance argument.

    support_valid declares the internal-load hypothesis and effort availability;
    callers must not substitute simulated contact forces in driver_tau.
    """
    i = Input()
    i.now = int(now) & 0xFFFFFFFF
    i.sensor_tick = int(sensor_tick) & 0xFFFFFFFF
    i.imu_tick = int(imu_tick) & 0xFFFFFFFF
    i.feedback_mask = 63
    i.imu_valid = 1
    i.enabled_mode = bool(mode)
    i.key_down = bool(key)
    i.upstream_owner = upstream_owner
    i.motion_entry = motion_entry
    i.motion_success = motion_success
    i.adopted_valid = 1
    i.adopted_sequence = adopted_sequence
    i.support_tick = int(support_tick) & 0xFFFFFFFF
    i.support_mask = int(support_valid)
    i.support_model_valid = bool(support_valid)
    for name, value in [
        ("gyro", gyro),
        ("gravity", gravity),
        ("q", q),
        ("dq", dq),
        ("command", command),
        ("upstream_target", adopted),
        ("feedback_tick", feedback_ticks),
        ("accel", accel),
        ("driver_tau", driver_tau),
        ("internal_tau", internal_tau),
        ("support_J", np.asarray(support_J).reshape(-1)),
        ("leg_length", leg_length),
        ("leg_spread", leg_spread),
    ]:
        target = getattr(i, name)
        a = np.asarray(value)
        if a.shape != (len(target),) or not np.isfinite(a).all():
            raise ValueError("finite explicit " + name)
        target[:] = a
    return i


class SharedJump:
    VALID_PHASES = tuple(range(9))
    ABI_VERSION = ABI
    INPUT_CONTRACT = INPUT_ID
    OUTPUT_CONTRACT = OUTPUT_ID
    CONFIG_CONTRACT = CONFIG_ID
    ConfigType = Config
    InputType = Input
    StateType = State
    FrameType = Frame

    def __init__(self, library, n, *, config=None, library_sha=None):
        if not 1 <= n <= 65536:
            raise ValueError("explicit batch capacity")
        from native_stand.artifacts.io import file_sha256

        if library_sha is None or file_sha256(library) != library_sha:
            raise ValueError("Jump库需要实际SHA身份")
        self.n = n
        self.library = str(library)
        self.library_sha = library_sha
        self.measurement = None
        lib = self.lib = C.CDLL(str(library))
        lib.RLJumpSize.argtypes = [U]
        lib.RLJumpSize.restype = U
        for k, size in enumerate(
            [
                self.ABI_VERSION,
                C.sizeof(self.ConfigType),
                C.sizeof(self.InputType),
                C.sizeof(self.StateType),
                C.sizeof(self.FrameType),
            ]
        ):
            if lib.RLJumpSize(k) != size:
                raise ValueError("Jump ABI layout mismatch " + str(k))
        specs = {
            "RLJumpDefaultConfig": [P],
            "RLJumpInit": [P, U],
            "RLJumpReset": [P, U, C.c_int],
            "RLJumpIdentity": [P],
            "RLJumpStep": [P, P, P],
            "RLJumpEncode": [P, P, P],
            "RLJumpDecodeMean": [P, P, P],
            "RLJumpPublishMean": [P, P, P, U, P],
            "RLJumpPd": [P, P, P],
            "RLJumpBatch": [P, P, P, P, U, C.c_int],
            "RLJumpEncodeBatch": [P, P, P, P, U],
            "RLJumpPublishBatch": [P, P, P, P, P, P, P, U],
            "RLJumpPdBatch": [P, P, P, P, U],
            "RLJumpEstimateSupport": [P, P, P],
            "RLJumpAccelReliable": [P],
            "RLJumpMotorCommand": [P, P, P],
        }
        for name, args in specs.items():
            getattr(lib, name).argtypes = args
        self.config = self.ConfigType()
        lib.RLJumpDefaultConfig(C.byref(self.config))
        if config is not None:
            self.config = copy_struct(config)
        self.states = (self.StateType * n)()
        self.inputs = (self.InputType * n)()
        self.frames = (self.FrameType * n)()
        self.last_snapshot = None
        for s in self.states:
            lib.RLJumpInit(C.byref(s), 0)

    def step(self, inputs, *, pd=False):
        if len(inputs) != self.n:
            raise ValueError("batch size")
        for k, i in enumerate(inputs):
            self.inputs[k] = i
        if self.measurement is not None:
            self.measurement.count("jump_C_FSM_calls")
        return self.lib.RLJumpBatch(
            self.states, self.inputs, C.byref(self.config), self.frames, self.n, pd
        )

    def observe(self):
        if self.measurement is not None:
            self.measurement.count("jump_C_encode_calls")
        out = np.empty((self.n, 37), np.float32)
        valid = np.empty(self.n, np.uint32)
        self.lib.RLJumpEncodeBatch(
            self.states, self.inputs, out.ctypes.data, valid.ctypes.data, self.n
        )
        self.last_snapshot = (self.StateType * self.n).from_buffer_copy(bytes(self.states))
        return out, valid.astype(bool)

    def publish(self, mean, mask=None, sequences=None, *, snapshot=None):
        if self.measurement is not None:
            self.measurement.count("jump_C_publish_calls")
        a = np.ascontiguousarray(mean, dtype=np.float32)
        if a.shape != (self.n, 6) or not np.isfinite(a).all():
            raise ValueError("finite batch mean")
        old = snapshot if snapshot is not None else self.last_snapshot
        if old is None:
            raise ValueError("observation token required")
        m = np.ascontiguousarray(
            np.ones(self.n, np.uint32) if mask is None else mask, dtype=np.uint32
        )
        seq = np.ascontiguousarray(
            [s.adopted_sequence + 1 for s in self.states] if sequences is None else sequences,
            dtype=np.uint32,
        )
        if m.shape != (self.n,) or seq.shape != m.shape:
            raise ValueError("mask/sequence size")
        rc = np.empty(self.n, np.int32)
        self.lib.RLJumpPublishBatch(
            self.states,
            self.inputs,
            old,
            seq.ctypes.data,
            a.ctypes.data,
            m.ctypes.data,
            rc.ctypes.data,
            self.n,
        )
        return rc

    def frame_records(self, frames=None):
        """Owned diagnostic snapshots, requested only by an actual consumer."""
        if self.measurement is not None:
            self.measurement.count("jump_C_diagnostic_snapshots")
        return [
            {
                name: array(getattr(f, name))
                if isinstance(getattr(f, name), C.Array)
                else int(getattr(f, name))
                for name, _ in self.FrameType._fields_
            }
            for f in (self.frames if frames is None else frames)
        ]

    def pd(self, *, as_dict=True):
        if self.measurement is not None:
            self.measurement.count("jump_C_PD_calls")
        rc = np.empty(self.n, np.int32)
        self.lib.RLJumpPdBatch(self.states, self.inputs, self.frames, rc.ctypes.data, self.n)
        return rc, self.frame_records() if as_dict else self.frames

    def state_dict(self):
        return dict(
            version="jump_shared_c_state_v1",
            abi=self.ABI_VERSION,
            input_id=self.INPUT_CONTRACT,
            output_id=self.OUTPUT_CONTRACT,
            config_id=self.CONFIG_CONTRACT,
            endian=sys.byteorder,
            n=self.n,
            library_sha=self.library_sha,
            config=base64.b64encode(bytes(self.config)).decode(),
            states=base64.b64encode(bytes(self.states)).decode(),
            inputs=base64.b64encode(bytes(self.inputs)).decode(),
            frames=base64.b64encode(bytes(self.frames)).decode(),
        )

    def load_state_dict(self, value):
        identity = (
            "jump_shared_c_state_v1",
            self.ABI_VERSION,
            self.INPUT_CONTRACT,
            self.OUTPUT_CONTRACT,
            self.CONFIG_CONTRACT,
            sys.byteorder,
            self.n,
            self.library_sha,
        )
        if (
            tuple(
                value.get(k)
                for k in [
                    "version",
                    "abi",
                    "input_id",
                    "output_id",
                    "config_id",
                    "endian",
                    "n",
                    "library_sha",
                ]
            )
            != identity
        ):
            raise ValueError("full-state contract/library identity")
        pending = {}
        for name, typ in [
            ("config", self.ConfigType),
            ("states", self.StateType * self.n),
            ("inputs", self.InputType * self.n),
            ("frames", self.FrameType * self.n),
        ]:
            data = base64.b64decode(value[name], validate=True)
            if len(data) != C.sizeof(typ):
                raise ValueError("full-state ABI bytes " + name)
            pending[name] = typ.from_buffer_copy(data)
        if bytes(pending["config"]) != bytes(self.config):
            raise ValueError("configuration identity")
        for s in pending["states"]:
            if (
                s.abi != self.ABI_VERSION
                or s.phase not in self.VALID_PHASES
                or s.owner > 4
                or not np.isfinite(np.r_[array(s.adopted), array(s.pending)]).all()
            ):
                raise ValueError("invalid C state")
        for name, value in pending.items():
            setattr(self, name, value)
        self.last_snapshot = None
