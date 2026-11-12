"""Weak old-Recovery physical-action prior on CURRENT student Swing states.

Never controls the plant. No successful-V4 label, clipping or teacher averaging.
"""

import torch

from native_stand.tasks.recovery.cubic_task import BASE_FIELDS


def offsets():
    start = 0
    out = {}
    for k, w in BASE_FIELDS:
        out[k] = start
        start += w
    return out


class ActionPrior:
    def __init__(self, teacher, coefficient=0.0025, max_rows=64):
        self.teacher = teacher.eval().requires_grad_(False)
        self.coefficient = coefficient
        self.max_rows = max_rows
        self.last = {}
        self.total_queried_rows = 0
        self.kinematics_offset = offsets()["kinematics"]
        self._constants = None

    def __call__(self, actor, batch, valid):
        if self.coefficient == 0:
            self.last = dict(rows=0, coefficient=0, total_queried_rows=self.total_queried_rows)
            return None
        x = batch.observations["student_actor"]
        c = batch.observations["student_critic"]
        kin = self.kinematics_offset
        initial = c[:, 863] * 0.5
        height = c[:, kin + 3]
        use = valid & (initial < 0.27) & (height < 0.30)
        if c.shape[-1] == 908:
            use &= c[:, 904] < 0.5  # 909 column904 is height, not Motion owner
        ids = torch.where(use)[0][: self.max_rows]
        self.total_queried_rows += len(ids)
        self.last = dict(
            total_queried_rows=self.total_queried_rows,
            source="oldRecovery1500; oldPD60/2 Kv1 success; V4 quality UNKNOWN",
            rows=len(ids),
            coefficient=self.coefficient,
        )
        if not len(ids):
            return None
        if (
            self._constants is None
            or self._constants[0].device != x.device
            or self._constants[0].dtype != x.dtype
        ):
            self._constants = (
                x.new_tensor([0, 1, 0, 1, 0, 0]),
                x.new_tensor([1.5, 1.5, 100.0, 1.5, 1.5, 100.0]),
                x.new_tensor([0.05, 0.05, 10.0, 0.05, 0.05, 10.0]),
            )
        flags, output_scale, loss_scale = self._constants
        inp = x[ids].detach().clone()
        inp[:, 25:31] = flags
        with torch.no_grad():
            target = self.teacher.mean(inp) * output_scale
        if (
            not torch.isfinite(target).all()
            or bool((target[:, [0, 1, 3, 4]].abs() > 1.5).any())
            or bool((target[:, [2, 5]].abs() > 100).any())
        ):
            raise ValueError("Unrepresentable action prior; no clipping")
        pred = actor.physical_mean(inp)
        error = pred - target
        self.last.update(
            leg_MAE_rad=error[:, [0, 1, 3, 4]].detach().abs().mean(),
            wheel_MAE_rad_s=error[:, [2, 5]].detach().abs().mean(),
        )
        return self.coefficient * torch.nn.functional.huber_loss(
            error / loss_scale, torch.zeros_like(error)
        )
