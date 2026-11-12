"""Reset-invalidated constants and precomputed causal history gather indices."""

import torch


class CriticCache:
    def __init__(self, actor):
        self.scales = actor.new_tensor([1.5, 1.5, 100, 1.5, 1.5, 100])
        self.parameter_scales = actor.new_tensor([2, 1, 1, 40, 40, 10, 10, 20, 20, 20, 10, 10, 10])
        self.target_scales = self.scales.repeat(5)
        self.indices = {}
        self.epoch = None
        self.static = None

    def history(self, ring, tick, ages):
        key = len(ring), tuple(ages)
        if key not in self.indices:
            slots = torch.arange(len(ring), device=ring.device)[:, None]
            offsets = torch.tensor(ages, device=ring.device)[None]
            self.indices[key] = (slots - offsets) % len(ring)
        return ring[self.indices[key][tick % len(ring)]].permute(1, 0, 2).flatten(1)

    def reset_fields(self, env):
        epoch = getattr(env, "_motion_reset_epoch", 0)
        if self.static is None or self.epoch != epoch:
            t = env.motion_timing
            d, b = t.dr, t.base
            self.static = dict(
                parameters=t.leg.x.reshape(env.num_envs, 2, 13) / self.parameter_scales,
                reference=t.leg.ref.reshape(env.num_envs, 4) / 1.5,
                gains=torch.cat((d.kp / 40, d.kd / 2, d.kv / 0.2), -1),
                caps=d.caps / 40,
                wheel_timing=torch.cat(
                    (b.motor_ring.delay[:, [2, 5]].float() / 30, b.motor_ring.tau[:, [2, 5]] / 10),
                    -1,
                ),
                context=d.parameter_tensor(),
                wheel_B_Tc=torch.cat((d.wheel_B, d.wheel_Tc), -1),
            )
            self.epoch = epoch
        return self.static

    def invalidate(self):
        self.static = None
