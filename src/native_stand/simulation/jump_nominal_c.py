"""37D trajectory plus bounded residual, actual firmware C ABI2.

Input schema remains37D. Output/config/trajectory/ABI change explicitly; old
absolute or72D checkpoint cannot be silently interpreted as residual control.
"""

import base64
import ctypes as C

import numpy as np

from native_stand.simulation.jump_shared_c import Config, Frame, SharedJump, State, fields


class NominalConfig(C.Structure):
    _fields_ = Config._fields_ + [("trajectory_id", C.c_uint32)]


class NominalState(C.Structure):
    _fields_ = State._fields_ + fields(
        "reference_initialized reference_phase_sequence reference_tick",
        [
            "nominal:6",
            "nominal_rate:6",
            "nominal_acc:6",
            "geometric:6",
            "bridge_delta:6",
            "bridge_rate:6",
            "bridge_acc:6",
            "sampled_nominal:6",
        ],
    )


class NominalFrame(C.Structure):
    _fields_ = Frame._fields_ + fields("", ["nominal:6", "residual:6"])


class NominalJump(SharedJump):
    ABI_VERSION = 2
    OUTPUT_CONTRACT = 0x4A370201
    CONFIG_CONTRACT = 0x4A050201
    ConfigType = NominalConfig
    StateType = NominalState
    FrameType = NominalFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lib.RLJumpNominalObserveBatch.argtypes = [C.c_void_p] * 5 + [C.c_uint32]
        self.lib.RLJumpNominalSample.argtypes = [C.c_uint32, C.c_uint32] + [C.c_void_p] * 4
        self.lib.RLJumpNominalUpdate.argtypes = [C.c_void_p, C.c_uint32]
        self.lib.RLJumpNominalDecode.argtypes = [C.c_void_p] * 4
        self.lib.RLJumpUpstreamMotorCommand.argtypes = [C.c_void_p] * 3

    def observe(self):
        out = np.empty((self.n, 37), np.float32)
        valid = np.empty(self.n, np.uint32)
        self.last_snapshot = (self.StateType * self.n)()
        self.lib.RLJumpNominalObserveBatch(
            self.states, self.inputs, self.last_snapshot, out.ctypes.data, valid.ctypes.data, self.n
        )
        return out, valid.astype(bool)

    def load_state_dict(self, value):
        raw = base64.b64decode(value["states"], validate=True)
        pending = (self.StateType * self.n).from_buffer_copy(raw)
        for s in pending:
            for name, _ in NominalState._fields_[len(State._fields_) + 3 :]:
                if not np.isfinite(np.asarray(getattr(s, name))).all():
                    raise ValueError("nonfinite nominal reference state")
        return super().load_state_dict(value)
