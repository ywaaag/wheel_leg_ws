"""Full GPU checkpoints must survive exp rounding, while rejecting changed distributions."""

import pytest
import torch

from native_stand.learning.ppo.jump_unified import validate_saved_std

FROZEN_TEST_SCOPE = "PORTABLE"


def test_device_exp_rounding_does_not_mutate_authoritative_log_std_or_saved_std():
    log_std = torch.tensor([-2.1261, -2.1259, -2.1233, -2.1245, -2.1269, -2.1245])
    sigma = log_std.exp()
    saved = torch.nextafter(sigma, torch.full_like(sigma, float("inf")))
    saved[::2] = torch.nextafter(sigma[::2], torch.zeros_like(sigma[::2]))
    original, original_log = saved.clone(), log_std.clone()
    assert not torch.equal(saved, sigma)
    validate_saved_std(saved, log_std)
    assert torch.equal(saved, original) and torch.equal(log_std, original_log)


@pytest.mark.parametrize("bad", ["changed", "nan", "zero", "inf", "shape", "dtype"])
def test_rounding_tolerance_does_not_accept_a_different_sampling_distribution(bad):
    log_std = torch.full((6,), -2.125)
    saved = log_std.exp()
    if bad == "changed":
        saved[0] += 0.00001
    elif bad == "nan":
        saved[0] = float("nan")
    elif bad == "zero":
        saved[0] = 0
    elif bad == "inf":
        saved[0] = float("inf")
    elif bad == "shape":
        saved = saved[:3]
    else:
        saved = saved.double()
    with pytest.raises(ValueError, match="std与Actor状态冲突"):
        validate_saved_std(saved, log_std)
