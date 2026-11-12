"""Explicit V4 acceleration, preserving causal queues and fitted equations.

The reference class is supplied by the environment boundary. No imported
module/class is replaced, and the frozen dependency remains read-only.
"""

import torch

from .observability import count, section

VERSION = "motion_v4_tensor_execution_v2"


def response_tick(
    q,
    v,
    target,
    slot,
    ids,
    qhist,
    vhist,
    queue,
    filtered,
    heldq,
    heldv,
    des,
    des_unclipped,
    actual,
    av,
    alpha,
    qage,
    vage,
    delay,
    qbias,
    vbias,
    vgain,
    color,
    qnoise,
    vnoise,
    white,
    strength,
    kp,
    kd,
    cap,
    ref,
    parameters,
    load_scale,
    *,
    pd: bool,
    quantized: bool,
):
    """Same operation order as the reference, including the 5ms held PD input."""
    filtered.add_(av * (v - filtered))
    qhist.index_copy_(0, slot, q[None])
    vhist.index_copy_(0, slot, filtered[None])
    if pd:
        heldq.copy_(qhist[(slot - qage + 256) % 128, ids])
        heldv.copy_(vhist[(slot - vage + 256) % 128, ids])
        heldq.add_(qbias).add_(white[:, :2] * qnoise)
        heldv.mul_(vgain).add_(vbias).add_(color).add_(white[:, 2:] * vnoise)
        if quantized:
            heldq.copy_(torch.round(heldq / (12.566 / 65535)) * (12.566 / 65535))
            heldv.copy_(torch.round(heldv / (90 / 4095)) * (90 / 4095))
        des_unclipped.copy_(strength * (kp * (target - heldq) - kd * heldv))
        des.copy_(des_unclipped.clamp(-cap, cap))
    demand = torch.stack(((des[:, 0] + des[:, 1]) * 0.5, (des[:, 1] - des[:, 0]) * 0.5), -1)
    queue.index_copy_(0, slot, demand[None])
    actual.add_(alpha * (queue[(slot - delay + 256) % 128, ids] - actual))
    motor = torch.stack((actual[:, 0] - actual[:, 1], actual[:, 0] + actual[:, 1]), -1)
    motor = motor.clamp(-cap, cap)
    shifted = q - ref
    z = torch.stack(
        ((shifted[:, 0] + shifted[:, 1]) * 0.5, (shifted[:, 1] - shifted[:, 0]) * 0.5), -1
    )
    dz = torch.stack(((v[:, 0] + v[:, 1]) * 0.5, (v[:, 1] - v[:, 0]) * 0.5), -1)
    x = parameters
    external = torch.stack(
        (
            x[:, 5] - x[:, 7] * z[:, 0] - x[:, 9] * z[:, 1] - x[:, 1] * dz[:, 0],
            x[:, 6] - x[:, 8] * z[:, 1] - x[:, 9] * z[:, 0] - x[:, 2] * dz[:, 1],
        ),
        -1,
    )
    if x.shape[1] == 13:
        external = external - x[:, 10:12] * z.square()
    return motor, torch.stack(
        (external[:, 0] - external[:, 1], external[:, 0] + external[:, 1]), -1
    ) * load_scale


def accelerate(original, *, compile_tick=False, measurement=None):
    """Adopt the existing state without reinitializing a single causal buffer."""
    base = type(original)

    class AcceleratedResponse(base):
        def reset(self, q, v, target, ids=None):
            count(
                self.measurement,
                "motion.response_reset_rows",
                len(q),
                reason="all_rows" if ids is None else "partial_rows",
            )
            with section(self.measurement, "motion.response_history_reset"):
                return self._reset_buffers(q, v, target, ids)

        def _reset_buffers(self, q, v, target, ids=None):
            if ids is None:
                ids = self.ids
                self.tick = 0
                for name in ("filtered", "heldq", "heldv", "des", "actual"):
                    setattr(self, name, torch.zeros_like(q))
            self.filtered[ids] = v
            self.color[ids] = 0
            self.heldq[ids] = q + self.qbias[ids]
            self.heldv[ids] = v * self.vgain[ids] + self.vbias[ids]
            requested = self.strength[ids] * (
                self.kp[ids] * (target - self.heldq[ids]) - self.kd[ids] * self.heldv[ids]
            )
            self.des[ids] = requested.clamp(-self.cap[ids], self.cap[ids])
            self.actual[ids] = torch.stack(
                (
                    (self.des[ids, 0] + self.des[ids, 1]) * 0.5,
                    (self.des[ids, 1] - self.des[ids, 0]) * 0.5,
                ),
                -1,
            )
            if not hasattr(self, "des_unclipped"):
                self.des_unclipped = self.des.clone()
            self.des_unclipped[ids] = requested
            ages = ((self._history_slots - self.tick) % 128 - 128).to(q.dtype)
            self.qhist[:, ids] = q[None] + v[None] * ages[:, None, None] * 0.001
            self.vhist[:, ids] = v
            self.queue[:, ids] = self.actual[ids]

        def step(self, q, v, target, quantized=True):
            tick = self.tick
            if self.sample_color is not None:
                self.color.mul_(self.color_rho).add_(
                    self.sample_color(tick) * self.color_std * (1 - self.color_rho.square()).sqrt()
                )
            white = self._zero_white
            if tick % 5 == 0 and self.sample_noise is not None:
                white = self.sample_noise(tick)
            arguments = (
                q,
                v,
                target,
                self._history_slots[tick % 128 : tick % 128 + 1],
                self.ids,
                self.qhist,
                self.vhist,
                self.queue,
                self.filtered,
                self.heldq,
                self.heldv,
                self.des,
                self.des_unclipped,
                self.actual,
                self.av,
                self.alpha,
                self.qage,
                self.vage,
                self.delay,
                self.qbias,
                self.vbias,
                self.vgain,
                self.color,
                self.qnoise,
                self.vnoise,
                white,
                self.strength,
                self.kp,
                self.kd,
                self.cap,
                self.ref,
                self.x,
                self.load_scale,
            )
            phase = (tick % 5 == 0, quantized)
            if self._compiled_tick and phase not in self._compiled_phases:
                # Lazy compiler/autotuner activity must not consume policy RNG.
                devices = [q.device.index] if q.is_cuda else []
                with (
                    section(
                        self.measurement,
                        "compile_prepare",
                        parent="rollout_seconds",
                        reason="PD" if phase[0] else "held_PD",
                    ),
                    torch.random.fork_rng(devices=devices),
                ):
                    result = self._tick_function(*arguments, pd=phase[0], quantized=quantized)
                self._compiled_phases.add(phase)
            else:
                result = self._tick_function(*arguments, pd=phase[0], quantized=quantized)
            self.tick += 1
            return result

    result = object.__new__(AcceleratedResponse)
    result.__dict__.update(original.__dict__)
    result._history_slots = torch.arange(128, device=original.x.device)
    result.measurement = measurement
    result._zero_white = original.x.new_zeros(original.n, 4)
    result._tick_function = response_tick
    result._compiled_tick = compile_tick
    result._compiled_phases = set()
    if compile_tick:
        result._tick_function = torch.compile(
            response_tick,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
    result.acceleration_readback = dict(
        version=VERSION,
        vectorized_reset=True,
        compiled_tensor_tick=compile_tick,
        frozen_reference_module=base.__module__,
        changes_physics_equations=False,
    )
    return result
