"""Reset-only contact hypotheses; physical stiffness/damping, not motor PD."""

from __future__ import annotations

import math

import torch

VERSION = "motion_wheel_contact_reset_DR_v1"
CRITIC_VERSION = "motion48_height_bank_contact_causal881_v1"
DIM = 881


def validate(config):
    if config.get("version") != VERSION:
        raise ValueError("Unknown wheel contact DR version")
    for name in ("stiffness_n_m", "damping_ns_m"):
        bounds = config[name]
        if len(bounds) != 2 or not all(math.isfinite(x) and x > 0 for x in bounds):
            raise ValueError("Contact bounds must be positive and finite: " + name)
        if bounds[0] > bounds[1]:
            raise ValueError("Contact bounds reversed: " + name)
    if not 0 < config["compliant_probability"] <= 1:
        raise ValueError("Contact mixture probability must lie in (0,1]")
    if config.get("profiles") != [0] or config.get("sampling") != "reset_only":
        raise ValueError("First contact recipe applies only to the combined reset cohort")
    return config


def sample(uniform, amplitude, config):
    """Five independent U[-1,1] values per active row; zero means rigid."""
    validate(config)
    if uniform.shape != (len(amplitude), 5):
        raise ValueError("Contact sampling needs five columns per row")
    a = amplitude.reshape(-1, 1)
    if not bool(torch.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()):
        raise ValueError("Contact strength outside [0,1]")
    enabled = ((uniform[:, :1] + 1) / 2 < config["compliant_probability"]) & (a > 0)
    # Strength widens a log-range from the geometric centre. The rigid
    # nominal cohort is separate; finite stiffness is an engineering hypothesis.
    result = []
    for j, name in enumerate(("stiffness_n_m", "damping_ns_m")):
        low, high = map(math.log, config[name])
        u = uniform[:, 1 + j * 2 : 3 + j * 2]
        result.append(torch.exp((low + high) / 2 + u * a * (high - low) / 2))
    return torch.cat(result, -1) * enabled.to(uniform)


def append_context(prefix, actual, config):
    if prefix.shape[-1] != 877 or actual.shape != (len(prefix), 4):
        raise ValueError("Contact Critic requires real877 prefix and four actual material values")
    scales = prefix.new_tensor([config["stiffness_n_m"][1]] * 2 + [config["damping_ns_m"][1]] * 2)
    return torch.cat((prefix, actual.to(prefix) / scales), -1)


def extend_source(source, job):
    """Explicit877->881 first-layer/Adam padding at the NEW branch boundary."""
    config = job.assets.get("wheel_contact_dr")
    if config is None:
        return source
    validate(config)
    if job.model.critic_input != DIM or job.model.critic_privileged_version != CRITIC_VERSION:
        raise ValueError("Contact DR requires the declared causal881 Critic")
    declaration = job.assets.get("contact_critic_extension")
    if declaration != {"from": "motion48_height_bank_planned_causal877_v2", "to": CRITIC_VERSION}:
        raise ValueError("Missing explicit contact Critic migration")
    if source["critic_privileged_version"] != declaration["from"]:
        raise ValueError("Contact Critic migration accepts877 source only; never repeat on resume")
    weight = source["critic"]["mlp.0.weight"]
    if weight.ndim != 2 or weight.shape[1] != 877:
        raise ValueError("Contact source first layer is not877")
    result = dict(source)
    result["critic"] = dict(source["critic"])
    result["critic"]["mlp.0.weight"] = torch.nn.functional.pad(weight, (0, 4))
    optim = source["optimizer"]
    from native_stand.learning.ppo.dual_adam import DualAdam

    if optim.get("version") != DualAdam.VERSION:
        raise ValueError("Contact migration requires explicit DualAdam")
    result["optimizer"] = dict(optim)
    critic_optim = dict(optim["critic"])
    ids = [p for group in critic_optim["param_groups"] for p in group["params"]]
    names = list(source["critic"])
    if len(ids) != len(names):
        raise ValueError("Critic Adam parameter identities differ")
    pid = ids[names.index("mlp.0.weight")]
    states = dict(critic_optim["state"])
    state = dict(states.get(pid, {}))
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if key in state:
            if state[key].shape != weight.shape:
                raise ValueError("Critic first-layer Adam moment shape differs")
            state[key] = torch.nn.functional.pad(state[key], (0, 4))
    if state:
        states[pid] = state
    critic_optim["state"] = states
    result["optimizer"]["critic"] = critic_optim
    result["network_architecture"] = {**source["network_architecture"], "critic_input": DIM}
    result["critic_privileged_version"] = CRITIC_VERSION
    result["contact_critic_extension_receipt"] = dict(
        version=VERSION,
        from_dim=877,
        to_dim=DIM,
        new_columns="actual wheel K[L,R]/Kmax, C[L,R]/Cmax; rigid=zero",
        first_layer_and_Adam_new_columns_zero=True,
        old_columns_Actor_Adam_std_RNG_and_warmup_preserved=True,
    )
    return result
