"""Same C controller feeding the existing full V4 motor/NET plant kernel.

This adapter is shared by training and playback. No physical integrator lives
here. One batch C crossing per 5ms PD, not one call per environment per ms.
The original V4 sampling, quantization, post-PD transport, response and NET
body are reused by replacing ONLY its two internal PD assignments in memory.
A source drift raises; the original frozen module/file is never changed.
"""

from __future__ import annotations

import ctypes as C
import random

import numpy as np
import torch

from native_stand.simulation.jump_shared_c import SharedJump, array, copy_struct
from native_stand.simulation.recovery_isaac.execution import Execution


class SharedV4Execution:
    def __init__(
        self,
        library,
        V4,
        response_class,
        n,
        *,
        device="cpu",
        dtype=torch.float32,
        config=None,
        library_sha=None,
        control_class=SharedJump,
        capture_frames=False,
    ):
        self.c = control_class(library, n, config=config, library_sha=library_sha)
        self.v4 = Execution(V4, response_class, n, device=device, dtype=dtype)
        self.v4.leg.external_pd = self._pd
        self.tick = 0
        self.last_frames = None
        self.capture_frames = capture_frames
        self.C_PD_calls = 0
        self.timing_version = "jump_same_C_target5_V4_sensor_postPD_v1"
        self.input_rows = None

    def reset(self, ids, q, dq):
        """Episode reset ONLY. Ownership transitions never call this method."""
        self.v4.reset(ids, q, dq)
        for k in ids.detach().cpu().tolist():
            self.c.lib.RLJumpInit(C.byref(self.c.states[k]), self.tick & 0xFFFFFFFF)

    def update(self, rows):
        if len(rows) != self.c.n:
            raise ValueError("feedback batch")
        self.input_rows = [copy_struct(i) for i in rows]
        # Deployed RobotTask/FSM cadence is 5ms; INS and plant remain1ms.
        if self.tick % 5 == 0:
            self.c.step(self.input_rows, pd=False)

    def observe(self, actor_rows=None):
        if self.input_rows is None:
            raise RuntimeError("explicit sensor rows required")
        if actor_rows is not None:
            if len(actor_rows) != self.c.n:
                raise ValueError("Actor sampled feedback batch")
            for k, i in enumerate(actor_rows):
                self.c.inputs[k] = i
        else:
            # Reuse the public V4 packet sample/hold/age exactly once. Only its
            # first16 physical fields are consumed: task flags, commands and
            # actual previous are always encoded by the real Jump C core.
            rows = self.input_rows
            v = self.v4
            frame = torch.tensor(
                np.stack(
                    [
                        np.r_[
                            np.asarray(i.gyro) * 0.25,
                            np.asarray(i.gravity),
                            np.asarray(i.q) / 1.5,
                            np.asarray(i.dq) * 0.05,
                        ]
                        for i in rows
                    ]
                ),
                device=v.device,
                dtype=v.dtype,
            )
            sensed = v.report(frame, v.last_q, v.last_dq)[:, :16].detach().cpu().numpy()
            ages = v.effective_packet_age.detach().cpu().tolist()
            for k, old in enumerate(rows):
                i = copy_struct(old)
                i.gyro[:] = sensed[k, :3] / 0.25
                i.gravity[:] = sensed[k, 3:6]
                i.q[:] = sensed[k, 6:10] * 1.5
                i.dq[:] = sensed[k, 10:16] / 0.05
                i.sensor_tick = (int(i.sensor_tick) - ages[k]) & 0xFFFFFFFF
                i.imu_tick = (int(i.imu_tick) - ages[k]) & 0xFFFFFFFF
                i.feedback_tick[:] = [(int(x) - ages[k]) & 0xFFFFFFFF for x in i.feedback_tick]
                self.c.inputs[k] = i
        return self.c.observe()

    def publish_latent(self, raw, mask=None):
        """PPO retains raw and its Gaussian probability; this is the ONLY tanh."""
        a = raw.detach().tanh().cpu().numpy() if torch.is_tensor(raw) else np.tanh(raw)
        return self.c.publish(a, mask)

    def publish_mean(self, mean, mask=None):
        a = mean.detach().cpu().numpy() if torch.is_tensor(mean) else mean
        return self.c.publish(a, mask)

    def submit(self, target):
        # Compatibility boundary with the existing physical plant: this checks
        # the already published request, and never decodes or delays it again.
        a = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
        expected = np.stack(
            [
                array(s.protected_target) if s.owner == 2 else array(i.upstream_target)
                for s, i in zip(self.c.states, self.c.inputs, strict=False)
            ]
        )
        if a.shape != expected.shape or not np.allclose(a, expected, rtol=0, atol=1e-6):
            raise ValueError("physical plant cannot bypass C request/adoption")
        return expected

    def _pd(self, heldq, heldv, ignored_target):
        v = self.v4
        if self.input_rows is None:
            raise RuntimeError("causal deployable input required before PD")
        # V4 is the sole source of sensor age/hold/quantization for the legs.
        q = (v.unpair(heldq)[:, [0, 1, 3, 4]] + v.bias).detach().cpu().numpy()
        dq = v.unpair(heldv).detach().cpu().numpy()
        for k, i0 in enumerate(self.input_rows):
            i = copy_struct(i0)
            i.q[:] = q[k]
            i.dq[0] = dq[k, 0]
            i.dq[1] = dq[k, 1]
            i.dq[3] = dq[k, 3]
            i.dq[4] = dq[k, 4]
            self.c.inputs[k] = i
        rc, frames = self.c.pd(as_dict=False)
        # Upstream fallback is an execution result, not a mutation of the C
        # checkpoint frame. Preserve that distinction in the Critic as before.
        frames = list(frames)
        # Non-Jump owners retain their original C PD/compensation/cap contract,
        # supplied only with their actually adopted target. Actor is external.
        for k, code in enumerate(rc):
            if code < 0:
                raise RuntimeError(
                    f"Jump C safety owner {self.c.states[k].fault}; no Motion fallback"
                )
            if code == 1:
                i = self.c.inputs[k]
                f = self.c.FrameType()
                target = np.asarray(i.upstream_target, np.float32)
                upstream = (
                    self.c.lib.RLJumpUpstreamMotorCommand
                    if self.c.ABI_VERSION in (2, 3, 4)
                    else self.c.lib.RLJumpMotorCommand
                )
                if upstream(C.byref(i), target.ctypes.data, C.byref(f)):
                    raise RuntimeError("upstream C PD input")
                frames[k] = f
        self.last_frames = self.c.frame_records(frames) if self.capture_frames else None
        self.C_PD_calls += 1
        final = torch.as_tensor(
            np.stack([np.asarray(f.final_model_nm) for f in frames]), device=v.device, dtype=v.dtype
        )
        adopted = torch.as_tensor(
            np.stack([np.asarray(f.adopted) for f in frames]), device=v.device, dtype=v.dtype
        )
        v.adopted.copy_(adopted)
        v.wheel_raw.copy_(
            torch.as_tensor(
                np.stack([np.asarray(f.pd_nm)[[2, 5]] for f in frames]),
                device=v.device,
                dtype=v.dtype,
            )
        )
        v.wheel_PD.copy_(final[:, [2, 5]])
        return v.pair(final)

    def step(self, q, dq):
        """1ms plant command/NET. No simulator truth becomes an Actor phase."""
        v = self.v4
        t = self.tick
        if v.leg.tick != t or self.input_rows is None:
            raise ValueError("continuous V4/shared C clocks")
        motor, net = v.leg.step(v.pair(q), v.pair(dq), v.pair(v.adopted))
        v.motor.copy_(v.unpair(motor))
        v.NET.copy_(v.unpair(net))
        v.wheel_queue[t % 128] = v.wheel_PD
        v.motor[:, [2, 5]] = v.wheel_queue[
            (t - v.wheel_delay) % 128, v.ids[:, None], torch.arange(2, device=v.device)[None]
        ]
        v.motors[t % 128] = v.motor
        v.NETs[t % 128] = v.NET
        v.theta.copy_(v.pair(q))
        v.omega.copy_(v.pair(dq))
        v.last_q.copy_(q)
        v.last_dq.copy_(dq)
        v.tick += 1
        self.tick += 1
        return v.motor, v.NET

    def state_dict(self):
        return dict(
            version=self.timing_version,
            tick=self.tick,
            C=self.c.state_dict(),
            V4=self.v4.state_dict(),
            C_PD_calls=self.C_PD_calls,
            rows=[bytes(i).hex() for i in self.input_rows] if self.input_rows is not None else None,
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            CUDA_rng=torch.cuda.get_rng_state(self.v4.device)
            if self.v4.device.type == "cuda"
            else None,
        )

    def load_state_dict(self, value):
        if value["version"] != self.timing_version or value["tick"] != value["V4"]["tick"]:
            raise ValueError("C/V4 full-state identity")
        self.c.load_state_dict(value["C"])
        self.v4.load_state_dict(value["V4"])
        self.tick = value["tick"]
        self.C_PD_calls = value["C_PD_calls"]
        from native_stand.simulation.jump_shared_c import Input

        self.input_rows = (
            None
            if value["rows"] is None
            else [Input.from_buffer_copy(bytes.fromhex(i)) for i in value["rows"]]
        )
        torch.set_rng_state(value["torch_rng"])
        np.random.set_state(value["numpy_rng"])
        random.setstate(value["python_rng"])
        if self.v4.device.type == "cuda":
            if value["CUDA_rng"] is None:
                raise ValueError("CUDA RNG cannot be omitted")
            torch.cuda.set_rng_state(value["CUDA_rng"], self.v4.device)
        elif value["CUDA_rng"] is not None:
            raise ValueError("cannot discard CUDA RNG")
