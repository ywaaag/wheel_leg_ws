"""Single new output definition: nominal control demand plus small residual."""

import numpy as np

VERSION = "jump_nominal_residual37_totalcap_v1"
OUTPUT_ID = 0x4A370201
CONFIG_ID = 0x4A050201
RESIDUAL = np.array(
    [
        [0.10, 0.10, 10, 0.10, 0.10, 10],
        [0.05, 0.05, 10, 0.05, 0.05, 10],
        [0.10, 0.10, 10, 0.10, 0.10, 10],
        [0.10, 0.10, 10, 0.10, 0.10, 10],
        [0.10, 0.10, 10, 0.10, 0.10, 10],
    ],
    np.float32,
)


def labels_from_physical(target, nominal, phase, activation):
    t = np.asarray(target, float)
    q = np.asarray(nominal, float)
    if t.shape != q.shape or t.shape[-1] != 6 or not np.isfinite(np.r_[t.ravel(), q.ravel()]).all():
        raise ValueError("same finite physical target/nominal shapes")
    phase = np.asarray(phase, int)
    act = np.asarray(activation, float)
    if (phase < 1).any() or (phase > 5).any() or (act < 0).any() or (act > 1).any():
        raise ValueError("valid Jump controlling phase and activation")
    scale = RESIDUAL[phase - 1] * np.expand_dims(act, -1)
    delta = t - q
    mean = np.divide(delta, scale, out=np.full_like(delta, np.nan), where=scale > 0)
    zero = (scale == 0) & (abs(delta) < 1e-8)
    mean[zero] = 0
    valid = np.isfinite(mean).all(-1) & (abs(mean) < 1).all(-1)
    latent = np.full_like(mean, np.nan)
    latent[valid] = np.arctanh(mean[valid])
    return dict(mean=mean, latent=latent, representable=valid, labels_clipped=False)


def critic_controller_fields(state, inputs):
    from .simple_critic import controller_fields, extend_fields

    values, definitions = controller_fields(state, inputs, fault_count=14)
    extras = {
        "nominal.initialized": (state.reference_initialized, 1.0, "bool"),
        "nominal.phase_matches": (
            state.reference_phase_sequence == state.phase_sequence,
            1.0,
            "bool",
        ),
        "nominal.reference_age": (
            ((int(inputs.now) - int(state.reference_tick)) & 0xFFFFFFFF) * 0.001
            if state.reference_initialized
            else 0.0,
            0.1,
            "seconds",
        ),
    }
    for name, unit in [
        ("nominal", "rad_rad_s"),
        ("geometric", "rad_rad_s"),
        ("sampled_nominal", "rad_rad_s"),
        ("nominal_rate", "rad_per_s"),
        ("nominal_acc", "rad_per_s2"),
        ("bridge_delta", "rad_rad_s"),
        ("bridge_rate", "rad_per_s"),
        ("bridge_acc", "rad_per_s2"),
    ]:
        units = (
            np.array([1.5, 1.5, 100, 1.5, 1.5, 100])
            if "rate" not in name and "acc" not in name
            else (
                np.array([20, 20, 1000, 20, 20, 1000])
                if "rate" in name
                else np.array([1000, 1000, 10000, 1000, 1000, 10000])
            )
        )
        extras["nominal." + name] = (list(getattr(state, name)), units, unit)
    return extend_fields(values, definitions, extras)
