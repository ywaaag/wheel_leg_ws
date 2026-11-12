"""Factory for a new Motion environment, never mutates an active DR class."""

from __future__ import annotations

from native_stand.tasks.motion.execution_limits import validate


def apply_limits(timing, config, ids=None):
    """PD, driver and Critic share actual caps. No clock, RNG or FIFO reset."""
    import torch

    limits = validate(config)
    base, dr = timing.base, timing.dr
    cap = dr.caps.new_tensor(limits.caps)
    if ids is None:
        ids = torch.arange(len(dr.caps), device=dr.caps.device)
    dr.caps[ids] = cap
    # The legacy PD uses a shared six-vector; all new-job rows have this cap.
    base.limits.copy_(cap.expand_as(base.limits))
    return limits


def dr_factory(base_class, config):
    validate(config)

    class Motion48DR(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Timing.dr is assigned by the caller after DR construction.
            limits = validate(config)
            self.caps[:] = self.caps.new_tensor(limits.caps)
            self.t.base.limits.copy_(self.caps[0].expand_as(self.t.base.limits))

        def reset(self, ids):
            super().reset(ids)
            apply_limits(self.t, config, ids)

        def summary(self):
            result = super().summary()
            result["motor_limits"] = validate(config).as_dict()
            result["motor_limits"]["effective_caps_min"] = self.caps.amin(0).cpu().tolist()
            result["motor_limits"]["effective_caps_max"] = self.caps.amax(0).cpu().tolist()
            return result

    return Motion48DR


def readback(timing, config, observations=None):
    import torch

    limits = validate(config)
    actual = timing.dr.caps
    cap = actual.new_tensor(limits.caps)
    if not torch.equal(actual, cap.expand_as(actual)):
        raise RuntimeError("Actual driver / Critic caps differ from Motion4.8 contract")
    if not torch.equal(timing.base.limits, cap.expand_as(timing.base.limits)):
        raise RuntimeError("Actual PD cap differs from Motion4.8 contract")
    if observations is not None:
        from critic_tensor import FIELDS

        offset = sum(
            width
            for _, width in FIELDS[
                : next(i for i, x in enumerate(FIELDS) if x[0] == "effective_motor_caps")
            ]
        )
        critic = observations["student_critic"]
        if not torch.allclose(critic[:, offset : offset + 6], actual / 40, atol=1e-7, rtol=0):
            raise RuntimeError("Critic actual motor-cap columns differ from physical driver")
    return {
        **limits.as_dict(),
        "actual_PD_and_driver_caps_verified": True,
        "actual_Critic_caps_verified": observations is not None,
        "Critic_source": "timing.dr.caps; reward effort uses the same tensor",
    }


def environment_class(base_class, config):
    """Cold-start extension: every partial reset reapplies the new limits."""
    validate(config)

    class Motion48Environment(base_class):
        def _reset_idx(self, ids):
            result = super()._reset_idx(ids)
            timing = getattr(self, "motion_timing", None)
            if timing is not None and timing.dr is not None:
                apply_limits(timing, config, ids)
            return result

    return Motion48Environment
