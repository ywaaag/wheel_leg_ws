"""Lightweight episode extrema from real geometry; diagnostics, never gates."""

import copy

import torch


class PhaseReadback:
    def __init__(self, n, device):
        z = torch.zeros(n, device=device)
        self.initial_geometry = z[:, None].expand(-1, 4).clone()
        self.minimum_length = z[:, None].expand(-1, 2).clone()
        self.maximum_swing_change = self.minimum_length.clone()
        self.maximum_height = z.clone()
        self.first_entry = z - 1
        self.maximum_hold = z.clone()

    def reset(self, ids, geometry, height):
        self.initial_geometry[ids] = geometry
        self.minimum_length[ids] = geometry[:, :2]
        self.maximum_swing_change[ids] = 0
        self.maximum_height[ids] = height
        self.first_entry[ids] = -1
        self.maximum_hold[ids] = 0

    def update(self, geometry, height, entry, hold):
        self.minimum_length = torch.minimum(self.minimum_length, geometry[:, :2])
        delta = geometry[:, 2:] - self.initial_geometry[:, 2:]
        change = torch.atan2(torch.sin(delta), torch.cos(delta)).abs()
        self.maximum_swing_change = torch.maximum(self.maximum_swing_change, change)
        self.maximum_height = torch.maximum(self.maximum_height, height)
        self.first_entry = torch.where(
            (self.first_entry < 0) & (entry >= 0), entry, self.first_entry
        )
        self.maximum_hold = torch.maximum(self.maximum_hold, hold)

    def record(self, i):
        return dict(
            initial_leg_length_m=self.initial_geometry[i, :2].tolist(),
            minimum_leg_length_m=self.minimum_length[i].tolist(),
            maximum_shortening_m=(self.initial_geometry[i, :2] - self.minimum_length[i]).tolist(),
            maximum_swing_angle_change_rad=self.maximum_swing_change[i].tolist(),
            maximum_root_height_m=float(self.maximum_height[i]),
            first_entry_s=float(self.first_entry[i]) if self.first_entry[i] >= 0 else None,
            maximum_continuous_hold_s=float(self.maximum_hold[i]),
            scope="actual geometry extrema, not a contraction/rotation success certificate",
        )

    def state_dict(self):
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state):
        if set(state) != set(vars(self)):
            raise ValueError("Incomplete phase readback state")
        for key, value in state.items():
            if key not in vars(self) or value.shape != getattr(self, key).shape:
                raise ValueError("Phase readback state shape")
            getattr(self, key).copy_(value)
