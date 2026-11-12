"""Reset-only Recovery DR subset from the frozen Motion parameter ranges.

No claim that every original Motion DR field is connected. The exact enabled
subset is serialized; zero strength does not consume the DR random stream.
"""

from __future__ import annotations

import numpy as np

VERSION = "recovery_nominal_anchor_DR_subset_v1"


def draw(strength, rng):
    if strength not in [0.0, 0.25, 0.5, 1.0]:
        raise ValueError("Declared DR strengths only")
    values = dict(
        strength=strength,
        root_mass_scale=1.0,
        root_inertia_scale=1.0,
        root_COM_m=np.zeros(3),
        friction=0.5,
        restitution=0.0,
        encoder_bias_rad=np.zeros(4),
        Actor_age_ms=20,
        target_delay_ms=5,
        wheel_delay_ms=np.array([7, 11]),
    )
    if strength == 0:
        return values
    u = rng.uniform(-1, 1, 16)
    # Residual zero bias starts ±1deg per user design; other magnitude bounds
    # reuse original .20 mass, .02 COM, .5..1.25 mu and 0..20 target delay.
    values.update(
        root_mass_scale=1 + u[0] * 0.2 * strength,
        root_inertia_scale=1 + u[0] * 0.2 * strength,
        root_COM_m=u[1:4] * 0.02 * strength,
        friction=0.5 + (u[4] + 1) * 0.375 * strength,
        encoder_bias_rad=u[5:9] * np.deg2rad(1 if strength == 0.25 else 2),
        target_delay_ms=int(round(5 + (((u[9] + 1) * 10) - 5) * strength)),
        wheel_delay_ms=np.rint(np.array([7, 11]) + u[10:12] * 4 * strength).astype(int),
    )
    if (u[13] + 1) * 0.5 < strength:
        values["Actor_age_ms"] = int(np.clip((u[14] + 1) * 1.5, 0, 2)) * 10
    return values


def coordinate_report_q(model_q, bias):
    result = np.array(model_q, copy=True)
    result[[0, 1, 3, 4]] += bias
    return result
