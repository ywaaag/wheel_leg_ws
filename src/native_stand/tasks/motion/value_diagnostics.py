"""按显式cohort计算稳定价值诊断；纯计算、不参与loss或晋级。"""

from __future__ import annotations

import torch

VERSION = "motion_cohort_centered_value_diagnostics_v2"


def grouped(returns, values, advantages, valid, cohorts, commands):
    r, v, a = (t.reshape(-1).double() for t in (returns, values, advantages))
    mask, index = valid.reshape(-1), cohorts.reshape(-1).long()
    if mask.dtype != torch.bool or any(t.shape != r.shape for t in (v, a, mask, index)):
        raise ValueError("value diagnostics require aligned tensors and boolean valid mask")
    size = len(commands)
    if index.min() < 0 or index.max() >= size:
        raise ValueError("value diagnostics cohort outside declared commands")
    count = torch.zeros(size, device=r.device, dtype=torch.long).scatter_add_(0, index, mask.long())
    den = count.clamp_min(1).double()
    raw = r - v

    def summed(x):
        return torch.zeros(size, device=r.device, dtype=torch.float64).scatter_add_(
            0, index, torch.where(mask, x, 0.0)
        )

    means = torch.stack(
        [summed(x) / den for x in (r, raw, a, raw.square(), ((raw > 0) != (a > 0)).double())], -1
    )
    # Center before summing squared residuals, avoiding E[x²]-E[x]² cancellation.
    variance = summed((r - means[index, 0]).square()) / den
    residual_variance = summed((raw - means[index, 1]).square()) / den
    ev = 1 - residual_variance / variance.clamp_min(1e-16)
    packed = torch.cat((count[:, None], means, variance[:, None], ev[:, None]), -1).cpu().tolist()
    result = []
    for command, row in zip(commands, packed, strict=False):
        n = int(row[0])
        result.append(
            dict(
                command=command,
                samples=n,
                return_mean=row[1] if n else None,
                raw_advantage_mean=row[2] if n else None,
                normalized_advantage_mean=row[3] if n else None,
                value_error_MSE=row[4] if n else None,
                sign_flip_fraction=row[5] if n else None,
                return_variance=row[6] if n else None,
                explained_variance=row[7] if n and row[6] > 1e-12 else None,
            )
        )
    return dict(
        version=VERSION,
        rows=result,
        training_objective_changed=False,
        variance="centered population moments, float64",
        no_sample_status="UNKNOWN/null",
    )


def from_storage(storage, *, commands_override):
    from native_stand.recovery.controlled_ppo import KEY

    return grouped(
        storage.returns,
        storage.values,
        storage.advantages,
        storage.observations[KEY],
        storage.observations["motion_diagnostic_cohort"],
        commands_override,
    )
