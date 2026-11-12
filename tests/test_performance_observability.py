"""Shared inclusive timings and counters through the real public driver; no engine."""

import copy
import json

import pytest
import torch

from native_stand.learning.ppo.performance import Measurement
from native_stand.simulation.driver import run_stage

FROZEN_TEST_SCOPE = "PORTABLE"


def test_disabled_measurement_is_empty_without_cuda_or_counter_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled measurement accessed CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    timer = Measurement(False, "cuda:0")
    with timer.section("rollout"), timer.section("reset", parent="rollout"):
        timer.count("reset_rows", object())
    assert timer.summary() == dict(
        seconds={},
        counters={},
        parents={},
        clocks={},
        aggregation="inclusive; do not add child sections to their parents",
    )


def test_nested_sections_do_not_sync_substeps_or_double_count(monkeypatch):
    from native_stand.learning.ppo import performance

    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(str(device)))
    clock = iter([0.0, 1.0, 3.0, 5.0])
    monkeypatch.setattr(performance, "perf_counter", lambda: next(clock))
    timer = Measurement(True, "cuda:0")
    with timer.section("rollout"), timer.section("reset", parent="rollout", synchronize=False):
        timer.count("reset_calls")
        timer.count("reset_rows", 4)
        timer.count("cache_invalidations", reason="partial_reset")
    assert len(calls) == 2
    assert timer.seconds == {"reset": 2.0, "rollout": 5.0}
    assert timer.parents == {"rollout": None, "reset": "rollout"}
    assert timer.clocks == {"rollout": "synchronized_stage_wall", "reset": "cpu_wall"}
    summary = timer.summary()
    timer.reset()
    assert summary["counters"]["reset_rows"] == 4 and not timer.counters


def test_timing_failure_does_not_hide_original_failure(monkeypatch):
    calls = []

    def sync(device):
        calls.append(device)
        if len(calls) == 2:
            raise RuntimeError("timing sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    timer = Measurement(True, "cuda:0")
    with (
        pytest.raises(ValueError, match="original fault") as raised,
        timer.section("rollout"),
    ):
        raise ValueError("original fault")
    assert "timing sync failed" in raised.value.__notes__[0]
    assert not timer._stack


@pytest.mark.parametrize("kind", ["recovery", "jump"])
@pytest.mark.parametrize("warmup", [False, True])
def test_shared_nested_measurement_preserves_rng_dual_adam_and_distribution(kind, warmup):
    from test_ppo_efficiency import equal, fill, observation, policy

    obs = observation(kind, [True, False, True, True])
    torch.manual_seed(333)
    original = policy(kind, obs, auxiliary=kind == "recovery")
    measured = copy.deepcopy(original)
    original.actor_update_enabled = measured.actor_update_enabled = not warmup
    rng = torch.get_rng_state()
    fill(original, obs, inference=True)
    original.update()
    end_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    timer = measured.measurement = Measurement(True)
    with timer.section("rollout"):
        fill(measured, obs, inference=True)
    with timer.section("learning"):
        measured.update()
    equal(end_rng, torch.get_rng_state())
    equal(original.actor.state_dict(), measured.actor.state_dict())
    equal(original.critic.state_dict(), measured.critic.state_dict())
    equal(original.optimizer.state_dict(), measured.optimizer.state_dict())
    equal(original.actor.output_distribution_params, measured.actor.output_distribution_params)
    assert timer.parents["actor_forward"] == "learning"
    assert timer.parents["minibatch_data"] == "learning"


def test_public_driver_shares_measurement_and_preserves_publish_rebuild_cleanup_order(tmp_path):
    from test_phase3_fixes import _cpu_job, _production_context, _sentinel_full_state

    job = _cpu_job(tmp_path, diagnostics={"performance_measurement": True})
    context = _production_context(tmp_path, job)
    events = []
    save = context.save
    prune = context.prune_checkpoints

    def saving(*args, **kwargs):
        events.append("save_SHA")
        return save(*args, **kwargs)

    def pruning():
        events.append("retention")
        return prune()

    context.save = saving
    context.prune_checkpoints = pruning

    class Adapter:
        def build(self, context, *args):
            assert self.measurement is context.measurement
            self.measurement.count("compile_initialization")

        def update_once(self, context, index):
            events.append("optimize")
            assert context.updates_used == index - 1
            with self.measurement.section("rollout"):
                self.measurement.count("reset_rows", index)
            return dict(actor_steps=1)

        def after_update(self, context, index, metrics):
            assert context.updates_used == index
            events.append("course")

        def capture(self, context, counters):
            events.append("capture")
            return _sentinel_full_state(job, sentinel=0.1)

        def after_save(self, context, reference):
            with self.measurement.section("candidate_references_views"):
                events.append("publish")
            with self.measurement.section("environment_close"):
                events.append("close_old")
            with self.measurement.section("environment_construction"):
                self.measurement.count("environment_rebuilds", reason="course_keys")
                events.append("construct_new")

        def close(self):
            events.append("close_final")

    run_stage(context, Adapter(), stage="smoke")
    assert events == [
        "optimize",
        "course",
        "capture",
        "save_SHA",
        "publish",
        "close_old",
        "construct_new",
        "retention",
    ] * 2 + ["close_final"]
    rows = [json.loads(x) for x in (context.run_dir / "performance.jsonl").read_text().splitlines()]
    assert [r["counters"]["reset_rows"] for r in rows] == [1, 2]
    assert rows[0]["initialization_measurement"]["counters"] == {"compile_initialization": 1}
    assert rows[1]["initialization_measurement"] is None
    for row in rows:
        assert row["parents"]["candidate_references_views"] == "after_save_total"
        assert row["parents"]["environment_close"] == "after_save_total"
        assert row["phase_seconds"]["after_save_total"] >= sum(
            row["phase_seconds"][key]
            for key in [
                "candidate_references_views",
                "environment_close",
                "environment_construction",
            ]
        )


def test_generic_adapter_passes_same_timer_to_environment_and_policy(tmp_path):
    from test_phase3_fixes import _cpu_job, _production_context

    from native_stand.simulation.cpu import CpuEnvFixture
    from native_stand.simulation.driver import Backend
    from native_stand.simulation.engine import GenericPPOAdapter

    job = _cpu_job(tmp_path)
    context = _production_context(tmp_path, job)
    timer = context.measurement = Measurement(True)
    constructed = []

    def factory(job, measurement):
        constructed.append(measurement)
        env = CpuEnvFixture(job)
        env.set_measurement = lambda actual: constructed.append(actual)
        return env

    adapter = GenericPPOAdapter(Backend("cpu", lambda job: None, make_measured_env=factory))
    adapter.build(context, "smoke", None)
    try:
        assert constructed == [timer, timer]
        assert adapter.measurement is timer and adapter.ppo.measurement is timer
    finally:
        adapter.close()
