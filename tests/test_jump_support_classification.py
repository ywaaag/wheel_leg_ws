"""Real C: hysteresis uncertainty must not masquerade as unavailable support."""

import pytest
from test_jump_unified import feedback

from native_stand.simulation.jump_build import build
from native_stand.simulation.jump_unified import UnifiedJump

FROZEN_TEST_SCOPE = "PORTABLE"


@pytest.fixture
def bridge(tmp_path):
    binding = build(tmp_path / "control.so")
    c = UnifiedJump(binding["path"], 1, library_sha=binding["sha256"])
    c.config.enabled = c.config.model_available = 1
    c.config.model_id = 1
    c.config.model_input_id = c.INPUT_CONTRACT
    c.config.model_output_id = c.OUTPUT_CONTRACT
    return c


def support_input(tick, forces, *, available=True, key=False):
    row = feedback(tick)
    row.internal_tau[:] = [forces[0] * 0.1, 0, forces[1] * 0.1, 0]
    row.support_mask = 3 if available else 0
    row.key_down = key
    return row


def test_fresh_deadband_and_debounce_do_not_fault_or_accept_an_unconfirmed_jump(bridge):
    # Actual failing trace: right side ~19N, then >20N for one sample, then OFF.
    sequence = [
        (11.57, 14.88),
        (0.13, 19.60),
        (0.31, 19.05),
        (0.47, 21.83),
        (0.57, 0.33),
        (0.65, 0.42),
        (0.65, 0.38),
    ]
    for tick, forces in zip(range(0, 35, 5), sequence, strict=True):
        bridge.step([support_input(tick, forces)])
        assert bridge.states[0].fault == 0
    for tick in range(35, 130, 5):
        bridge.step([support_input(tick, [12, 12], key=tick == 40)])
        assert bridge.states[0].phase == 9 and bridge.states[0].fault == 0
    assert not bridge.states[0].entry_valid
    # Entry still needs confirmed support plus 100ms posture dwell.
    for tick in range(130, 240, 5):
        bridge.step([support_input(tick, [100, 100])])
    assert bridge.states[0].phase == 1 and bridge.states[0].fault == 0


@pytest.mark.parametrize("reason", ["unavailable", "stale", "singular"])
def test_unavailable_support_retains_real_C_fault_and_rejects_actor_samples(bridge, reason):
    for tick in range(0, 30, 5):
        row = support_input(tick, [100, 100], available=reason != "unavailable")
        if reason == "stale":
            row.support_tick = 0
        elif reason == "singular":
            row.support_J[:] = [0] * 8
        bridge.step([row])
    assert bridge.states[0].phase == 8 and bridge.states[0].fault == 5
    _, valid = bridge.observe()
    assert not valid.any()
