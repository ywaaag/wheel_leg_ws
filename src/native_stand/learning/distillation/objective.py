"""Explicit physical-target BC objective and episode/phase sampling."""

import torch


def validate_objective(config, student_kind):
    if config is None:
        return
    if student_kind != "recovery_physical":
        raise ValueError("Physical Huber requires the Recovery physical decoder")
    if set(config) != {"kind", "scales", "phase_weights", "updates_per_round"}:
        raise ValueError("Explicit physical objective fields required")
    if config["kind"] != "physical_huber" or config["scales"] != [
        0.05,
        0.05,
        10.0,
        0.05,
        0.05,
        10.0,
    ]:
        raise ValueError("Recovery physical Huber scales differ")
    if type(config["updates_per_round"]) is not int or config["updates_per_round"] < 1:
        raise ValueError("Positive fixed supervised update count required")
    weights = config["phase_weights"]
    if not weights or any(
        not torch.isfinite(torch.tensor(float(v))) or float(v) <= 0 for v in weights.values()
    ):
        raise ValueError("Positive finite phase weights required")


def per_row_loss(prediction, target, config=None):
    if config is None:
        return (prediction - target).square().mean(-1)
    residual = (prediction - target) / prediction.new_tensor(config["scales"])
    return torch.nn.functional.huber_loss(
        residual, torch.zeros_like(residual), reduction="none"
    ).mean(-1)


def sampling_weights(episodes, phases, phase_weights):
    """Each episode gets equal mass; its observed phases get declared relative mass."""
    episodes = torch.as_tensor(episodes)
    phases = torch.as_tensor(phases)
    if phases.shape != episodes.shape or not len(episodes):
        raise ValueError("Actual per-row Recovery phase evidence required")
    result = torch.zeros(len(episodes), dtype=torch.float64)
    for episode in episodes.unique():
        mask = episodes == episode
        present = phases[mask].unique().tolist()
        if any(str(p) not in phase_weights for p in present):
            raise ValueError("Unknown Recovery sampling phase")
        mass = sum(phase_weights[str(p)] for p in present)
        for phase in present:
            rows = mask & (phases == phase)
            result[rows] = phase_weights[str(phase)] / mass / int(rows.sum())
    return result / result.sum()
