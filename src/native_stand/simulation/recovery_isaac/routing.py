"""Read-only expert query; submitting/adopting remains a separate causal step."""

import torch

from .execution import LEGS


def physical_control(policy_raw, actor_observation, student_mask, C0, C0_center):
    if (
        policy_raw.ndim != 2
        or policy_raw.shape[-1] != 6
        or not bool(torch.isfinite(policy_raw).all())
    ):
        raise ValueError("Finite raw physical policy request")
    actual = policy_raw.clone()
    ids = torch.where(~student_mask)[0]
    if len(ids):
        x = actor_observation[ids].clone()
        x[:, 25:31] = x.new_tensor([1, 0, 0, 1, 0, 0])
        with torch.no_grad():
            mean = C0.mean(x)
        if mean.shape != (len(ids), 6) or not bool(torch.isfinite(mean).all()):
            raise ValueError("Finite frozen C0 mean")
        target = mean * 80
        target[:, LEGS] = C0_center + 0.15 * mean[:, LEGS]
        actual[ids] = target
    return actual
