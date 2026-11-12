"""因果Actor传感包；四槽历史，部分reset保留其他环境历史。"""

import torch


class Packets:
    def __init__(self, timing, *, noise):
        self.noise = noise
        self.t = timing
        self.delay_ms = 20
        self.age = torch.zeros(timing.n, device=timing.device)
        self.values = torch.zeros(4, timing.n, 16, device=timing.device)
        self.true_frame = self.values[0].clone()
        self.result = self.true_frame.clone()
        self.ready = torch.zeros(timing.n, device=timing.device, dtype=torch.bool)
        self.birth = torch.zeros(timing.n, device=timing.device, dtype=torch.long)
        self.leg_age = self.age.clone()
        self.last_tick = None
        self.has_fresh = True

    def reset(self, ids):
        self.ready[ids] = False
        self.age[ids] = 0
        self.leg_age[ids] = 0
        self.has_fresh = True

    def _initialize_history(self, sample, tick):
        fresh = ~self.ready
        self.values.copy_(torch.where(fresh[None, :, None], sample[None], self.values))
        self.birth.copy_(torch.where(fresh, self.birth.new_tensor(tick), self.birth))

    def read(self, current, tick):
        d = self.t.dr
        self.true_frame.copy_(current)
        if tick != self.last_tick or self.has_fresh:
            u = self.noise(d.episodes + tick // 10, self.t.ids, 16, 311)
            sample = current.clone()
            sample[:, :3] += (d.bias[:, :3] + u[:, :3] * d.scales[:, 0:1]) * 0.25
            rotation = d.bias[:, 3:6] + u[:, 3:6] * d.scales[:, 1:2]
            angle = rotation.norm(dim=-1, keepdim=True)
            axis = rotation / angle.clamp_min(1e-12)
            g = sample[:, 3:6].clone()
            sample[:, 3:6] = (
                g * angle.cos()
                + torch.cross(axis, g, dim=-1) * angle.sin()
                + axis * (axis * g).sum(-1, keepdim=True) * (1 - angle.cos())
            )
            sample[:, [12, 15]] += (d.bias[:, [12, 15]] + u[:, [12, 15]] * d.scales[:, 4:5]) * 0.05
            q, v = self.t.measurement()
            sample[:, 6:10] = q[:, [0, 1, 3, 4]] / 1.5
            sample[:, [10, 11, 13, 14]] = v[:, [0, 1, 3, 4]] * 0.05
            # reset() already supplies a Python dirty flag. No per-step GPU
            # reduction or four-slot warm-fill is needed when every row is ready.
            if self.has_fresh:
                self._initialize_history(sample, tick)
            if tick != self.last_tick:
                self.values[(tick // 10) % 4] = sample
            self.ready.fill_(True)
            self.last_tick = tick
            self.has_fresh = False
            age = torch.minimum(d.packet_age, (tick - self.birth).clamp_min(0) // 10 * 10)
            self.result.copy_(self.values[((tick - age) // 10) % 4, self.t.ids])
            self.age.copy_(age.float())
            la = torch.minimum(d.leg_packet_age, (tick - self.birth).clamp_min(0) // 10 * 10)
            lv = self.values[((tick - la) // 10) % 4, self.t.ids]
            self.result[:, 6:12] = lv[:, 6:12]
            self.result[:, 13:15] = lv[:, 13:15]
            self.leg_age.copy_(la.float())
        return d.dq.read(self.result, tick)

    def pending(self):
        k = self.t.tick // 10
        return torch.stack([self.values[(k - j) % 4] for j in (0, 1, 2)], 1)
