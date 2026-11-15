"""Large Critic packing preserves columns, numeric values and rejection gates."""

import json
from pathlib import Path

import numpy as np
import pytest

from native_stand.tasks.jump.controller_critic import Field, NormalizedCritic, PackedCritic

FROZEN_TEST_SCOPE = "PORTABLE"


def encoders():
    path = Path(__file__).parents[1] / "configs/contracts/jump_isaac_full_execution_critic_v1.json"
    fields = [
        Field(x["name"], tuple(x["shape"]), np.asarray(x["scale"]), x["unit"])
        for x in json.loads(path.read_text())["fields"]
    ]
    rng = np.random.default_rng(41)
    values = {f.name: rng.normal(size=f.shape) for f in fields}
    return NormalizedCritic(fields), PackedCritic(fields), values


def test_full_5591_columns_are_bitwise_equal_and_outputs_are_owned():
    old, packed, values = encoders()
    assert old.contract() == packed.contract()
    for multiplier in (0.0, 1.0, -3.5, 1e6):
        current = {k: v * multiplier for k, v in values.items()}
        actual = packed.encode(current)
        np.testing.assert_array_equal(actual, old.encode(current))
        assert actual.shape == (5591,) and actual.dtype == np.float32
        actual[:] = 123
        np.testing.assert_array_equal(packed.encode(current), old.encode(current))


@pytest.mark.parametrize("bad", ["missing", "extra", "shape", "nan", "inf"])
def test_packed_critic_retains_dynamic_field_shape_and_finite_checks(bad):
    old, packed, values = encoders()
    name = next(k for k, v in values.items() if v.size > 1)
    if bad == "missing":
        del values[name]
    elif bad == "extra":
        values["unknown"] = np.zeros(1)
    elif bad == "shape":
        values[name] = np.zeros((123,))
    else:
        values[name].flat[0] = float(bad)
    for encoder in (old, packed):
        with pytest.raises(ValueError, match="missing/extra|shape/finite"):
            encoder.encode(values)


def test_performance_receipt_reuses_existing_mask_and_allocator_metadata(monkeypatch):
    from types import SimpleNamespace

    import torch

    from native_stand.learning.ppo.performance import Measurement
    from native_stand.simulation.jump_adapter import JumpAdapter

    adapter = JumpAdapter.__new__(JumpAdapter)
    adapter.env = SimpleNamespace(completed=[], serial=0)
    adapter.metrics_cursor = 0
    adapter.measurement = Measurement(True, "cpu")
    adapter.ppo = SimpleNamespace(
        mask_summary=dict(valid_actor_samples=17, invalid_actor_samples=3)
    )
    adapter.job = SimpleNamespace(runtime=SimpleNamespace(device="cuda:0"))
    calls = []

    def metadata(device):
        calls.append(device)
        return 1234

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", metadata)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", metadata)
    result = dict(reward_mean=0.25)
    adapter.after_update(SimpleNamespace(updates_used=1), 0, result)
    assert calls == ["cuda:0", "cuda:0"]
    assert result == dict(
        reward_mean=0.25,
        peak_memory_allocated_bytes=1234,
        peak_memory_reserved_bytes=1234,
        jump_episode_window=result["jump_episode_window"],
    )
    assert result["jump_episode_window"]["ground_active_jump"]["episodes"] == 0
    host_peak = adapter.measurement.counters.pop("jump_host_peak_rss_bytes")
    assert host_peak > 0
    assert adapter.measurement.counters == dict(
        jump_valid_actor_samples=17, jump_invalid_actor_samples=3
    )
    # Production writer filters scalars; use its existing public field names.
    import tempfile

    from native_stand.artifacts.metrics import MetricsWriter

    target = Path(tempfile.mkdtemp()) / "metrics.jsonl"
    result["update"] = 1
    adapter.measurement.counters["jump_host_peak_rss_bytes"] = host_peak
    result["performance_counters"] = adapter.measurement.counters
    writer = MetricsWriter(target, every_updates=1, tensorboard=False)
    assert writer.write(result)
    saved = json.loads(target.read_text())
    assert saved["peak_memory_allocated_bytes"] == 1234
    assert saved["peak_memory_reserved_bytes"] == 1234
    assert saved["jump_episode_window"] == result["jump_episode_window"]
    assert saved["performance_counters"]["jump_host_peak_rss_bytes"] == host_peak
    assert saved["performance_counters"]["jump_valid_actor_samples"] == 17
