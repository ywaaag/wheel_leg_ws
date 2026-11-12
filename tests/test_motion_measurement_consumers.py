"""Actual shared Measurement through Motion consumers; scripted native I/O only."""

import json
from types import SimpleNamespace as NS

import pytest
import torch

from native_stand.learning.ppo.performance import Measurement
from native_stand.simulation.isaac.contact_materials import write_native
from native_stand.simulation.isaac.critic_cache import CriticCache
from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter, _build_environment
from native_stand.students.resume import restore_rng, rng_state

FROZEN_TEST_SCOPE = "PORTABLE"


@pytest.mark.parametrize("enabled", [False, True])
def test_contact_counts_actual_full_pool_vs_selected_rows_without_extra_sync(monkeypatch, enabled):
    from test_motion_contact_terrain import contact_fixture

    def forbidden(*args, **kwargs):
        pytest.fail("Motion child measurement accessed CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    measured, _, measured_io, _ = contact_fixture()
    reference, _, reference_io, _ = contact_fixture()
    timer = measured.measurement = Measurement(enabled, "cuda:0")
    rng = torch.get_rng_state()
    ids = torch.tensor([0])  # Rigid single-row reset, full native buffers still have4 rows.
    reference.reset(ids)
    measured.reset(ids)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(measured.actual, reference.actual)
    assert torch.equal(measured_io.mat, reference_io.mat)
    assert torch.equal(measured_io.sd, reference_io.sd)
    summary = timer.summary()
    if not enabled:
        assert not summary["seconds"] and not summary["counters"]
        return
    counts = summary["counters"]

    def getter(prefix):
        return sum(v for k, v in counts.items() if k.startswith(prefix + ":"))

    assert counts["motion.contact_reset_rows"] == 1
    assert getter("motion.material_getter_calls") == 12
    assert getter("motion.material_getter_pool_rows") == 48
    assert getter("motion.material_getter_used_rows") == 8
    assert getter("motion.material_setter_calls") == 4
    assert getter("motion.material_setter_buffer_rows") == 16
    assert getter("motion.material_setter_used_rows") == 4
    assert getter("motion.material_setter_buffer_bytes") > 0
    assert not any(k.startswith("motion.explicit_transfer_bytes") for k in counts)
    assert all(clock == "cpu_wall" for clock in summary["clocks"].values())


def test_native_setter_failure_is_counted_without_hiding_the_original():
    timer = Measurement(True)

    class Native:
        def get_material_properties(self):
            return torch.zeros(4, 1, 3)

        def set_material_properties(self, *args):
            raise RuntimeError("native setter failed")

    with pytest.raises(RuntimeError, match="native setter failed"):
        write_native(
            Native(),
            "set_material_properties",
            (torch.zeros(4, 1, 3), torch.tensor([2], dtype=torch.int32)),
            measurement=timer,
        )
    assert timer.counters["motion.material_setter_calls:set_material_properties"] == 1
    assert timer.counters["motion.material_setter_used_rows:set_material_properties"] == 1
    assert "motion.material_setters" in timer.seconds and not timer._stack


def test_cache_counters_measure_hits_and_real_invalidation_without_scalar_readbacks():
    from test_motion_rollout_hotpath import ScalarReads

    timer = Measurement(True)
    cache = CriticCache(torch.zeros(2, 31), timer)
    ring = torch.zeros(128, 2, 4)
    with ScalarReads() as scalars:
        cache.history(ring, 0, range(1, 5))
        cache.history(ring, 1, range(1, 5))
        cache.invalidate(reason="partial_reset")
    assert scalars.count == 0
    assert timer.counters == {
        "motion.critic_history_index_cache:miss": 1,
        "motion.critic_history_index_cache:hit": 1,
        "motion.critic_parameter_cache_invalidation:partial_reset": 1,
    }


def test_formal_environment_factory_injects_exact_shared_measurement_before_reset(tmp_path):
    from test_motion_production_integration import make_job

    job, _ = make_job(tmp_path, retention=False)
    timer = Measurement(True)
    events = []

    class NativeIO:
        device = "cpu"
        command_index = torch.zeros(job.runtime.envs, dtype=torch.long)
        motion_cohort_mix = torch.zeros(job.runtime.envs)

        def __init__(self, *args, measurement=None, **kwargs):
            self.measurement = measurement
            assert measurement is timer
            events.append("construct")

        def set_measurement(self, measurement):
            assert measurement is self.measurement is timer
            events.append("inject")

        def select_initial_family(self, *args):
            events.append("family")

        def reset(self):
            assert self.measurement is timer
            timer.count("fixture.actual_reset_rows", len(self.command_index))
            events.append("reset")
            return {}, {}

        def close(self):
            events.append("close")

    source = dict(symmetry_mix_state=[{"mix": 0.25}], reward_refinement_state=[{"mix": 0.25}])
    env, _ = _build_environment(
        job,
        {},
        {},
        None,
        source,
        NativeIO,
        lambda e, o: o,
        train=True,
        measurement=timer,
        construction_reason="course_revision",
    )
    assert env.measurement is timer and events == ["construct", "inject", "family", "reset"]
    assert timer.parents["environment_construction"] == "after_save_total"
    assert timer.parents["first_reset_readback"] == "after_save_total"
    assert timer.counters["fixture.actual_reset_rows"] == job.runtime.envs
    assert env._measurement_reset_parent == "rollout_seconds"


def test_candidate_and_rebuild_sections_use_public_after_save_parent_and_keep_counts(tmp_path):
    from test_motion_spin_branches import fixture
    from test_motion_window_accounting import environment, step, window

    job, _, _ = fixture(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"logical_run_id": "fixture"}))
    old = environment()
    old.onpolicy_episodes.bind_sample_window(window())
    step(old)
    old.onpolicy_episodes.export(100, window=window())
    order = []
    old.close = lambda: order.append("close")
    adapter = MotionProductionAdapter()
    adapter.job = job
    adapter.env = old
    adapter.context = NS(run_dir=run, updates_used=100)
    adapter.state = {"stage": 1, "consumed_window": {"end_update": 100}}
    adapter.modules = {"rng_state": rng_state, "restore_rng": restore_rng}
    adapter._needs_rebuild = True
    adapter.measurement = timer = Measurement(True)

    def construct():
        order.append("construct")
        new = environment()
        new.stage_spec["id"] = 1
        # Deliberate construction randomness must not consume training RNG.
        torch.rand(2)
        return new, {}

    adapter._make_env = construct
    rng = torch.get_rng_state()
    with timer.section("after_save_total", synchronize=False):
        adapter.after_save(adapter.context, {"update": 100})
    assert order == ["close", "construct"] and torch.equal(rng, torch.get_rng_state())
    assert not adapter._needs_rebuild
    assert timer.parents["candidate_references_views"] == "after_save_total"
    assert timer.parents["environment_close"] == "after_save_total"
    assert timer.counters["motion.environment_rebuild:course_revision"] == 1
    current = adapter.env.onpolicy_episodes.sample_accounting.summary()
    assert current["student_policy_samples"] == current["all_transitions"] == 0
    assert current["cumulative_student_policy_samples"] == 1
    assert current["cumulative_all_transitions"] == 2


def test_rebuild_construction_failure_keeps_original_exception_and_rng(tmp_path):
    adapter = MotionProductionAdapter()
    adapter.env = NS(close=lambda: None)
    adapter.modules = {"rng_state": rng_state, "restore_rng": restore_rng}
    adapter.measurement = Measurement(True)

    def fail():
        torch.rand(2)
        raise RuntimeError("construction failed")

    adapter._make_env = fail
    rng = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="construction failed"):
        adapter._rebuild_environment()
    assert torch.equal(rng, torch.get_rng_state()) and adapter.env is None
