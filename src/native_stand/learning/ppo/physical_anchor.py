"""Rotate through ALL qualified anchors; balance command height and dq activity.

No episode/phase labels are inferred from a flat NPZ. torch's checkpointed
device RNG makes sampled indices reproducible on complete resume.
"""

import torch

VERSION = "physical_anchor_full_pool_height_dq_stratified_v1"


class PhysicalAnchor:
    def __init__(self, x, y, coefficient, batch_size=512):
        if len(x) == 0 or x.shape != (len(x), 31) or y.shape != (len(x), 6):
            raise ValueError("Nonempty paired 31D/6D physical anchor")
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()) or coefficient < 0:
            raise ValueError("Finite qualified anchor and nonnegative coefficient")
        self.x = x
        self.y = y
        self.coefficient = coefficient
        self.batch_size = batch_size
        levels = x.new_tensor([0.30, 0.32, 0.34, 0.36, 0.38])
        height = (x[:, 24:25] * 0.5 - levels).abs().argmin(-1)
        active = (torch.linalg.vector_norm(x[:, 10:16] / 0.05, dim=-1) > 1.0).long()
        group = 2 * height + active
        _, inverse, counts = torch.unique(group, return_inverse=True, return_counts=True)
        self.weights = counts[inverse].to(x.dtype).reciprocal()
        self.last_indices = None

    def __call__(self):
        self.last_indices = torch.multinomial(self.weights, self.batch_size, replacement=True)
        return self.x[self.last_indices], self.y[self.last_indices], self.coefficient
