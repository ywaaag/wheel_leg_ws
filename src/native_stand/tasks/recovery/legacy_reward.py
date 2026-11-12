"""Old successful contact-quality reward, explicitly adapted to V4 units.

Pure tensor algebra. Old reference implementation is read-only and used by
CPU parity tests. No controller, target modification or learned quantities.
"""

import torch

VERSION = "recovery_legacy_contact_quality_V4_physical_v7"
FIELDS = (
    "lift_trend",
    "supported_level",
    "height_precision",
    "mirror_precision",
    "posture_huber",
    "pre_entry_rotation",
    "post_entry_rotation",
    "post_entry_velocity",
    "mirror_huber",
    "effort",
    "target_change",
    "post_entry_zero_wide",
    "post_entry_zero_narrow",
    "post_entry_wheel_smooth",
    "post_entry_contact_deficit",
)
WEIGHTS = (2.0, 3.0, 1.0, 1.0, -0.15, -0.1, -0.1, -0.5, -0.1, -0.02, -0.002, 0.5, 1.5, -0.2, -0.5)
MAX_DENSITY = 9.0  # Exact sum of positive maxima, including both stop kernels.


def huber(x):
    a = x.abs()
    return torch.where(a <= 1, 0.5 * a.square(), a - 0.5)


def kernel(x, width):
    return torch.exp(-(x / x.new_tensor(width)).square().mean(-1))


def density(
    *,
    state,
    h0,
    command_height,
    leg_q,
    angular_pitch_roll,
    motor,
    previous_target,
    adopted_target,
    entered,
    entry_age,
    clearance,
    controlled,
    valid,
    fault,
    effort_caps,
):
    n = len(h0)
    for name, x, width in [
        ("state", state, 13),
        ("leg_q", leg_q, 4),
        ("angular", angular_pitch_roll, 2),
        ("motor", motor, 6),
        ("previous", previous_target, 6),
        ("adopted", adopted_target, 6),
    ]:
        if x.shape != (n, width) or not bool(torch.isfinite(x).all()):
            raise ValueError("Finite explicit reward " + name)
    for name, x in [
        ("h0", h0),
        ("command", command_height),
        ("entry_age", entry_age),
        ("clearance", clearance),
    ]:
        if x.shape != (n,) or not bool(torch.isfinite(x).all()):
            raise ValueError("Finite reward " + name)
    for x in [entered, controlled, valid, fault]:
        if x.shape != (n,) or x.dtype != torch.bool:
            raise ValueError("Explicit reward masks")
    if bool((effort_caps <= 0).any()):
        raise ValueError("Positive physical caps")
    height = state[:, 3]
    angles = state[:, 4:6]
    contacts = state[:, 9:11]
    H = ((height - h0) / (0.30 - h0).clamp_min(0.05)).clamp(0, 1)
    C = ((contacts > 2).all(-1) & (clearance > 0)).to(height.dtype)
    pb = kernel(angles, [0.5, 0.35])
    pn = kernel(angles, [0.1, 0.05])
    mirror = torch.stack((leg_q[:, 0] + leg_q[:, 2], leg_q[:, 1] - leg_q[:, 3]), -1)
    mb = kernel(mirror, [0.30, 0.30])
    mn = kernel(mirror, [0.06, 0.06])
    hb = torch.exp(-((height - command_height) / 0.1).square())
    hn = torch.exp(-((height - command_height) / 0.025).square())
    G = H * C * pb
    entered_mask = entered.to(height.dtype)
    a = entry_age.clamp(0, 1)
    alpha = a.square() * (3 - 2 * a)
    gate = entered_mask * alpha
    velocity = state[:, :3]
    delta = adopted_target - previous_target
    raw = torch.stack(
        (
            H * pb,
            H * C * (0.4 * pb + 0.6 * pn),
            G * (0.4 * hb + 0.6 * hn),
            G * (0.4 * mb + 0.6 * mn),
            huber(angles / angles.new_tensor([0.5, 0.35])).mean(-1),
            (1 - entered_mask)
            * ((angular_pitch_roll.abs() - 2).clamp_min(0) / 2).square().mean(-1),
            entered_mask * angular_pitch_roll.square().mean(-1),
            entered_mask
            * (
                0.2 * alpha * huber(velocity[:, 0] / 0.2)
                + huber(velocity[:, 1] / 0.2)
                + huber(velocity[:, 2] / 0.3)
            ),
            G * huber(mirror / 0.2).mean(-1),
            (motor / effort_caps).square().mean(-1),
            (delta / delta.new_tensor([0.1, 0.1, 10.0, 0.1, 0.1, 10.0])).square().mean(-1),
            gate * torch.exp(-(velocity[:, 0] / 0.5).square()),
            gate * torch.exp(-(velocity[:, 0] / 0.15).square()),
            gate * (delta[:, [2, 5]] / 10).square().mean(-1),
            gate * ((2.0 - contacts) / 2).clamp(0, 1).square().mean(-1),
        ),
        -1,
    )
    eligible = controlled & valid & ~fault
    weighted = raw * raw.new_tensor(WEIGHTS) * eligible[:, None]
    if not bool(torch.isfinite(weighted).all()):
        raise ValueError("Nonfinite legacy reward")
    return dict(
        density={k: weighted[:, i] for i, k in enumerate(FIELDS)},
        height_progress=H,
        precision_gate=G,
        mirror_error_rad=mirror,
        stop_alpha=alpha,
        weighted_density=weighted,
        eligible=eligible,
    )
