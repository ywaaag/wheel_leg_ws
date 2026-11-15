"""Optional exact firmware spring-command injection; no passive spring model."""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import torch

from native_stand.artifacts.io import file_sha256


class FirmwareSpring:
    def __init__(self, sources, output):
        for ref in sources.values():
            if file_sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("Firmware spring dependency changed: " + ref["path"])
        self.sources = sources
        library = Path(output) / "firmware_spring.so"
        subprocess.run(
            [
                "cc",
                "-O2",
                "-std=c11",
                "-shared",
                "-fPIC",
                "-DRL_KINEMATICS_HOST",
                *[sources[k]["path"] for k in ("control", "motion", "gate", "geometry")],
                "-lm",
                "-o",
                str(library),
            ],
            check=True,
        )
        self.lib = ctypes.CDLL(str(library))
        ar = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        self.lib.RLControlCompensateSpring.argtypes = [ar, ar, ar, ar]
        self.lib.RLMotion31MapPolicyTargets.argtypes = [ar, ar, ar]
        self.lib.RLMotion31MapPolicyLegTorqueToMotorOrder.argtypes = [ar, ar]
        self.lib.getPhi.argtypes = [ctypes.c_float] * 7 + [ctypes.POINTER(ctypes.c_float)] * 4
        self.held = None
        self.delta = []
        self.saturation = 0

    def __call__(self, q, dq, motor, tick, bias):
        del dq
        if self.held is None or tick % 5 == 0:
            final = torch.zeros_like(motor)
            for row in range(len(q)):
                reported = q[row].numpy().copy()
                reported[[0, 1, 3, 4]] += bias[row].numpy()
                angles, wheels = np.zeros(4, np.float32), np.zeros(2, np.float32)
                if self.lib.RLMotion31MapPolicyTargets(reported, angles, wheels):
                    raise ValueError("Spring geometry mapping failed")
                lengths = np.zeros(2, np.float32)
                for side in range(2):
                    result = [ctypes.c_float() for _ in range(4)]
                    self.lib.getPhi(
                        float(angles[side * 2]),
                        float(angles[side * 2 + 1]),
                        0.21,
                        0.25,
                        0.25,
                        0.21,
                        0.0,
                        *[ctypes.byref(x) for x in result],
                    )
                    lengths[side] = result[2].value
                spread = np.asarray([angles[0] - angles[1], angles[2] - angles[3]], np.float32)
                before, after = np.zeros(4, np.float32), np.zeros(4, np.float32)
                if self.lib.RLControlCompensateSpring(before, lengths, spread, after):
                    raise ValueError("Firmware spring rejected geometry")
                # Invert the exact motor-order/sign mapping: lf=-ch4,lb=-ch3,rf=ch1,rb=-ch0.
                final[row, [0, 1, 3, 4]] = torch.tensor([-after[3], after[2], -after[1], -after[0]])
                self.delta.append((after - before).tolist())
            self.held = final
        result = motor.clone()
        summed = motor[:, [0, 1, 3, 4]] + self.held[:, [0, 1, 3, 4]]
        self.saturation += int((summed.abs() >= 40).sum())
        result[:, [0, 1, 3, 4]] = summed.clamp(-40, 40)
        return result

    def report(self):
        a = np.asarray(self.delta)
        return dict(
            kind="firmware_015_spring_command",
            passive_spring_added=False,
            PD_period_ms=5,
            motor_limit_Nm=40,
            post_add_saturation_samples=self.saturation,
            added_motor_torque_max_abs_Nm=float(abs(a).max()) if a.size else 0.0,
            source_sha_bound=True,
        )
