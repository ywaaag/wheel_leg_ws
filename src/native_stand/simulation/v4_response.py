"""Full v4 causal response, complete tuple and declared sample-time noise.

Default deterministic path is checked against original v4 NumPy/goldens.
No physical state writes, future data or independently mixed fitted endpoints.
"""

import torch

PARAMETERS = (
    "leg_mass_inertia_scale",
    "extra_modal_B_common",
    "extra_modal_B_difference",
    "motor_tau_common_ms",
    "motor_tau_difference_ms",
    "net_load_common_Nm",
    "net_load_difference_Nm",
    "net_K_common",
    "net_K_difference",
    "net_K_cross",
)


def modal(v):
    return torch.stack(((v[..., 0] + v[..., 1]) * 0.5, (v[..., 1] - v[..., 0]) * 0.5), -1)


def physical(v):
    return torch.stack((v[..., 0] - v[..., 1], v[..., 0] + v[..., 1]), -1)


class Response:
    def __init__(self, parameters, profiles, *, pd=None):
        self.pd = pd
        if parameters.shape[1] not in (10, 13):
            raise ValueError("explicit v3(10) or v4(13) tuple required")
        self.x = parameters
        self.profiles = profiles
        self.n = len(parameters)
        self.tick = 0
        self.kp = parameters.new_full((self.n, 2), 40.0)
        self.kd = parameters.new_full((self.n, 2), 2.0)
        self.cap = parameters.new_full((self.n, 2), 40.0)
        self.strength = parameters.new_ones(self.n, 2)
        self.vgain = parameters.new_ones(self.n, 2)
        self.color = parameters.new_zeros(self.n, 2)
        self.color_std = parameters.new_zeros(self.n, 1)
        self.color_rho = parameters.new_ones(self.n, 1)
        self.sample_color = None
        self.qbias = parameters.new_zeros(self.n, 2)
        self.vbias = self.qbias.clone()
        self.qnoise = self.qbias.clone()
        self.vnoise = self.qbias.clone()
        self.sample_noise = None
        self.load_scale = parameters.new_ones(self.n, 1)
        self.qhist = parameters.new_zeros(128, self.n, 2)
        self.vhist = self.qhist.clone()
        self.queue = self.qhist.clone()
        self.delay = torch.tensor([p["transport_ms"] for p in profiles], device=parameters.device)
        self.qage = torch.tensor([p["q_age_ms"] for p in profiles], device=parameters.device)
        self.vage = torch.tensor([p["v_age_ms"] for p in profiles], device=parameters.device)
        self.ref = parameters.new_tensor([p["q_reference"] for p in profiles])
        tau = parameters[:, 3:5]
        self.alpha = torch.where(
            tau > 0, -torch.expm1(-1 / tau.clamp_min(1e-12)), torch.ones_like(tau)
        )
        vt = parameters.new_tensor([p["velocity_tau_ms"] for p in profiles])[:, None]
        self.av = torch.where(vt > 0, -torch.expm1(-1 / vt.clamp_min(1e-12)), torch.ones_like(vt))
        self.ids = torch.arange(self.n, device=parameters.device)

    def reset(self, q, v, target, ids=None):
        # q/v/target contain just the reset rows. A partial simulator reset must
        # not erase another environment's history or restart the shared clock.
        if ids is None:
            ids = self.ids
            self.tick = 0
            self.filtered = torch.zeros_like(q)
            self.heldq = torch.zeros_like(q)
            self.heldv = torch.zeros_like(q)
            self.des = torch.zeros_like(q)
            self.actual = torch.zeros_like(q)
        self.filtered[ids] = v
        self.color[ids] = 0
        self.heldq[ids] = q + self.qbias[ids]
        self.heldv[ids] = v * self.vgain[ids] + self.vbias[ids]
        requested = (
            self.strength[ids]
            * (self.kp[ids] * (target - self.heldq[ids]) - self.kd[ids] * self.heldv[ids])
            if self.pd is None
            else self.pd(self.heldq[ids], self.heldv[ids], target)
        )
        self.des[ids] = requested.clamp(-self.cap[ids], self.cap[ids])
        self.actual[ids] = modal(self.des[ids])
        if not hasattr(self, "des_unclipped"):
            self.des_unclipped = self.des.clone()
        self.des_unclipped[ids] = requested
        for h in range(128):
            age = (h - self.tick) % 128 - 128
            self.qhist[h, ids] = q + v * age * 0.001
        self.vhist[:, ids] = v
        self.queue[:, ids] = self.actual[ids]

    def step(self, q, v, target, quantized=True):
        t = self.tick
        self.filtered += self.av * (v - self.filtered)
        if self.sample_color is not None:
            self.color.mul_(self.color_rho).add_(
                self.sample_color(t) * self.color_std * (1 - self.color_rho.square()).sqrt()
            )
        self.qhist[t % 128] = q
        self.vhist[t % 128] = self.filtered
        if t % 5 == 0:
            self.heldq.copy_(self.qhist[(t - self.qage + 256) % 128, self.ids])
            self.heldv.copy_(self.vhist[(t - self.vage + 256) % 128, self.ids])
            self.heldq += self.qbias
            self.heldv.mul_(self.vgain).add_(self.vbias).add_(self.color)
            if self.sample_noise is not None:
                u = self.sample_noise(t)
                self.heldq += u[:, :2] * self.qnoise
                self.heldv += u[:, 2:] * self.vnoise
            if quantized:
                self.heldq.copy_(torch.round(self.heldq / (12.566 / 65535)) * (12.566 / 65535))
                self.heldv.copy_(torch.round(self.heldv / (90 / 4095)) * (90 / 4095))
            demand = (
                self.strength * (self.kp * (target - self.heldq) - self.kd * self.heldv)
                if self.pd is None
                else self.pd(self.heldq, self.heldv, target)
            )
            self.des_unclipped.copy_(demand)
            self.des.copy_(self.des_unclipped.clamp(-self.cap, self.cap))
        self.queue[t % 128] = modal(self.des)
        self.actual += self.alpha * (
            self.queue[(t - self.delay + 256) % 128, self.ids] - self.actual
        )
        motor = physical(self.actual).clamp(-self.cap, self.cap)
        z = modal(q - self.ref)
        dz = modal(v)
        x = self.x
        extmodal = torch.stack(
            (
                x[:, 5] - x[:, 7] * z[:, 0] - x[:, 9] * z[:, 1] - x[:, 1] * dz[:, 0],
                x[:, 6] - x[:, 8] * z[:, 1] - x[:, 9] * z[:, 0] - x[:, 2] * dz[:, 1],
            ),
            -1,
        )
        if x.shape[1] == 13:
            extmodal -= x[:, 10:12] * z.square()
        self.tick += 1
        return motor, physical(extmodal) * self.load_scale
