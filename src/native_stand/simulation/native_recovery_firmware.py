"""Original Recovery C pipeline on the existing V4 plant, without extra spring.

This diagnoses controller migration. Physical springs, bus latency and MCU
inference duration are not identified by a firmware source checkout.
"""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import torch

from native_stand.artifacts.io import file_sha256

LEGS = [0, 1, 3, 4]
WHEELS = [2, 5]

BRIDGE = r"""
#include "rl_control.h"
#include <math.h>
#include <string.h>
static void sample(const float *f, RLMotion31Snapshot *s) {
    memset(s,0,sizeof(*s));
    for(unsigned i=0;i<3;i++) {
        s->gyro_model_rad_s[i]=f[i]*4;
        s->gravity_model[i]=f[i+3];
    }
    for(unsigned i=0;i<4;i++) s->joint_position_rad[i]=f[i+6]*1.5f;
    for(unsigned i=0;i<6;i++) s->joint_velocity_rad_s[i]=f[i+10]*20;
    s->command_height_m=.34f;s->recovery_mode=1;
}
int NativeObservation(const float *f,const float *previous,float *out) {
    RLMotion31Snapshot s;sample(f,&s);
    memcpy(s.previous_target,previous,6*sizeof(float));
    return RLMotion31BuildObservation(&s,out);
}
int NativeDecode(const float *mean,float *target) {
    return RLMotion31DecodeOutput(mean,target);
}
int NativeTorque(const float *f,const float *target,int current_gains,
                 float *raw,float *limited,float *physical) {
    RLMotion31Snapshot s;sample(f,&s);
    if(RLMotion31ComputePdTorque(&s,target,raw,limited))return -1;
    if(current_gains)for(unsigned i=0;i<6;i++) {
        if(i==2 || i==5)raw[i]=.2f*(target[i]-s.joint_velocity_rad_s[i]);
        else raw[i]=40*(target[i]-s.joint_position_rad[i<2?i:i-1])-2*s.joint_velocity_rad_s[i];
        float cap=(i==2 || i==5)?5.5f:40.f;
        limited[i]=fminf(fmaxf(raw[i],-cap),cap);
    }
    RLMotion31Preview p={0};float encoder[4]={0},angles[4],currents[2],wheel[2];
    if(RLMotion31MapPolicyTargets(target,p.hardware_joint_target_rad,p.motor_wheel_target_rad_s) ||
       RLMotion31MapPolicyWheelTorqueToForwardOrder(limited,wheel) ||
       RLControlMapOutput(&p,wheel,encoder,encoder,angles,currents))return -1;
    memcpy(physical,limited,6*sizeof(float));
    physical[2]=-currents[1]/RL_MOTION31_WHEEL_CURRENT_PER_NM;
    physical[5]=currents[0]/RL_MOTION31_WHEEL_CURRENT_PER_NM;
    return 0;
}
"""


class NativeFirmware:
    def __init__(self, sources, output):
        for ref in sources.values():
            if file_sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("Native firmware source SHA changed")
        output = Path(output)
        bridge = output / "native_recovery_bridge.c"
        bridge.write_text(BRIDGE)
        library = output / "native_recovery.so"
        subprocess.run(
            [
                "cc",
                "-O2",
                "-std=c11",
                "-shared",
                "-fPIC",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(bridge),
                sources["rl_motion31"]["path"],
                sources["rl_control"]["path"],
                "-I" + str(Path(sources["rl_motion31_header"]["path"]).parent),
                "-lm",
                "-o",
                str(library),
            ],
            check=True,
        )
        self.lib = ctypes.CDLL(str(library))
        ar = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        self.lib.NativeObservation.argtypes = [ar, ar, ar]
        self.lib.NativeDecode.argtypes = [ar, ar]
        self.lib.NativeTorque.argtypes = [ar, ar, ctypes.c_int, ar, ar, ar]

    def observation(self, frame, previous):
        out = np.zeros(31, np.float32)
        if self.lib.NativeObservation(frame, previous, out):
            raise ValueError("Native observation rejected")
        return out

    def torque(self, frame, target, current_gains=False):
        raw, limited, physical = [np.zeros(6, np.float32) for _ in range(3)]
        if self.lib.NativeTorque(frame, target, current_gains, raw, limited, physical):
            raise ValueError("Native PD/output rejected")
        return raw, limited, physical

    def decode(self, mean):
        target = np.zeros(6, np.float32)
        if self.lib.NativeDecode(mean, target):
            raise ValueError("Native decoder rejected")
        return target


def execution_factory(base, firmware, profile):
    """Reuse the frozen execution histories and V4 NET; inject original C torque.

    V4 motor transport/modal time constants remain. Firmware has no explicit
    target5/wheel7/11 queues; unknown bus/inference age is an explicit sweep.
    The discarded ordinary PD calculation never reaches the physics engine.
    """

    class NativeExecution(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.native_previous = torch.zeros_like(self.requested)
            self.native_command = torch.zeros_like(self.requested)
            self.native_frame = torch.zeros(self.n, 16, dtype=self.dtype)
            self.native_frames = torch.zeros(16, self.n, 16, dtype=self.dtype)
            self.native_queue = torch.zeros(128, self.n * 2, 2, dtype=self.dtype)
            self.native_actual = torch.zeros(self.n * 2, 2, dtype=self.dtype)

        def reset(self, ids, q, dq):
            super().reset(ids, q, dq)
            self.native_previous[ids] = 0
            self.native_command[ids] = 0
            self.native_queue[:, (ids[:, None] * 2 + torch.arange(2)).flatten()] = 0
            self.native_actual[(ids[:, None] * 2 + torch.arange(2)).flatten()] = 0
            self.packet_age[ids] = profile["actor_age_ms"]
            self.target_delay[ids] = 0
            self.wheel_delay[ids] = 0

        def report(self, true_frame, q, dq):
            super().report(true_frame, q, dq)
            frame = true_frame.clone()
            frame[:, 6:10] += self.bias / 1.5
            first = self.tick == 0
            if first:
                self.native_frames[:] = frame
            self.native_frames[(self.tick // 10) % 16] = frame
            age = min(profile["actor_age_ms"], self.tick // 10 * 10)
            self.native_frame.copy_(self.native_frames[((self.tick - age) // 10) % 16])
            self.effective_packet_age.fill_(age)
            return torch.from_numpy(
                np.stack(
                    [
                        firmware.observation(f.numpy().copy(), p.numpy().copy())
                        for f, p in zip(self.native_frame, self.native_previous, strict=True)
                    ]
                )
            )

        def submit(self, raw):
            super().submit(raw)
            # Original C has no joint-range projection after the linear decoder.
            self.requested.copy_(raw)
            for i in range(self.n):
                _, _, physical = firmware.torque(
                    self.native_frame[i].numpy().copy(),
                    raw[i].numpy().copy(),
                    profile["gains"] == "current",
                )
                self.native_command[i] = torch.from_numpy(physical)
            self.native_previous.copy_(raw)
            self.adopted.copy_(raw)
            return self.requested

        def step(self, q, dq):
            t = self.tick
            _, net = super().step(q, dq)
            pair = self.pair(self.native_command)
            desired = torch.stack(
                ((pair[:, 0] + pair[:, 1]) * 0.5, (pair[:, 1] - pair[:, 0]) * 0.5), -1
            )
            self.native_queue[t % 128] = desired
            transported = self.native_queue[(t - self.leg.delay) % 128, self.leg.ids]
            self.native_actual += self.leg.alpha * (transported - self.native_actual)
            motor_pair = torch.stack(
                (
                    self.native_actual[:, 0] - self.native_actual[:, 1],
                    self.native_actual[:, 0] + self.native_actual[:, 1],
                ),
                -1,
            )
            self.motor.copy_(self.unpair(motor_pair.clamp(-40, 40)))
            self.motor[:, WHEELS] = self.native_command[:, WHEELS]
            self.motors[t % 128] = self.motor
            return self.motor, net

    return NativeExecution
