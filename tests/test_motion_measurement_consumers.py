"""Actual shared Measurement through Motion consumers; scripted native I/O only."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from native_stand.learning.ppo.performance import Measurement
from native_stand.simulation.isaac.contact_materials import write_native
from native_stand.simulation.isaac.critic_cache import CriticCache
from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter, _build_environment
from native_stand.simulation.isaac.observability import rebuild_attribution, rebuild_inputs
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
    assert timer.counters["motion.environment_rebuild"] == 1
    assert timer.counters["motion.environment_rebuild:attribution_unavailable"] == 1
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


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("cohort_keys", "cohort_keys"),
        ("slot_counts", "slots"),
        ("sampling_slots", "slots"),
        ("terrain_families", "scene"),
        ("identified_amplitudes", "physics"),
        ("commands", "task"),
    ],
)
def test_rebuild_reasons_compare_actual_inputs_and_keep_key_slot_evidence(tmp_path, field, reason):
    from test_motion_window_accounting import environment

    job = NS(assets={})
    spec = {"curriculum": {"stages": [environment().stage_spec]}}
    before = rebuild_inputs(spec, job)
    after = copy.deepcopy(spec)
    stage = after["curriculum"]["stages"][0]
    stage["id"] += 1
    stage[field] = ["changed"]
    rng = torch.get_rng_state()
    result = rebuild_attribution(before, rebuild_inputs(after, job))
    assert result["reasons"] == [reason]
    assert result["old"]["cohort_keys"] == spec["curriculum"]["stages"][0]["cohort_keys"]
    assert result["new"]["cohort_keys"] == stage["cohort_keys"]
    assert result["old"]["slot_counts"] == [1, 1]
    assert result["new"]["slot_counts"] == stage["slot_counts"]
    assert torch.equal(rng, torch.get_rng_state())


def test_statistics_revision_and_multi_reason_rebuild_keep_existing_lifecycle(tmp_path):
    from test_motion_spin_branches import fixture
    from test_motion_window_accounting import environment

    job, _, _ = fixture(tmp_path)
    old = environment()
    spec = {"curriculum": {"stages": [old.stage_spec]}}
    previous = rebuild_inputs(spec, job)
    revised = copy.deepcopy(spec)
    stage = revised["curriculum"]["stages"][0]
    stage["id"] = 1
    assert rebuild_attribution(previous, rebuild_inputs(revised, job))["reasons"] == [
        "statistics_revision_only"
    ]
    stage.update(cohort_keys=[["new"]], slot_counts=[2], terrain_families=["H1"])
    revised["motion_timing"] = {"configuration": {"target_ms": 5}}
    detail = rebuild_attribution(previous, rebuild_inputs(revised, job))
    assert detail["reasons"] == ["cohort_keys", "slots", "scene", "physics"]
    order = []
    old.close = lambda: order.append("close")
    adapter = MotionProductionAdapter()
    adapter.env = old
    adapter._rebuild_attribution = detail
    adapter.measurement = timer = Measurement(True)
    adapter._make_env = lambda: (order.append("construct"), {})
    adapter._bind_sample_window = lambda: None
    # No sample state to import into this engine-I/O-only fixture.
    del old.onpolicy_episodes
    with timer.section("after_save_total", synchronize=False):
        adapter._rebuild_environment()
    assert order == ["close", "construct"]
    assert timer.counters["motion.environment_rebuild"] == 1
    for reason in detail["reasons"]:
        assert timer.counters["motion.environment_rebuild:" + reason] == 1
    assert "motion.environment_rebuild:statistics_revision_only" not in timer.counters


def test_real_course_advance_reports_rebuild_inputs_before_save_without_early_rebuild(tmp_path):
    from test_motion_spin_branches import fixture

    from native_stand.simulation.isaac.motion_backend import _resolve_source
    from native_stand.tasks.motion.course_strategy import resolve
    from native_stand.tasks.motion.legacy_spec_adapter import build_environment_spec
    from native_stand.tasks.motion.maneuver_statistics import RESPONSE_VERSION, episode_class

    job, _, _ = fixture(tmp_path, envs=128)
    source = _resolve_source(job, None).payload
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"logical_run_id": "fixture"}))
    adapter = MotionProductionAdapter()
    adapter.job, adapter.source = job, source
    adapter.course_strategy = resolve(job.curriculum.version)
    adapter.state = adapter.course_strategy.prepare(source, job.runtime.envs)
    adapter.template = json.loads(Path(job.assets["runtime_template"]).read_text())
    adapter.spec = build_environment_spec(
        adapter.template, source, job, stage_builder=adapter.course_strategy.stage_config
    )
    old_stage = adapter.spec["curriculum"]["stages"][0]
    adapter.env = env = NS(
        num_envs=128,
        device="cpu",
        stage_spec=old_stage,
        identified_dr=NS(episodes=torch.zeros(128)),
        transient_events=NS(summary=lambda: {}),
    )
    env.onpolicy_episodes = episode_class(object, version=RESPONSE_VERSION)(env)
    adapter.context = context = NS(run_dir=run, updates_used=100)
    adapter.modules = {"build_environment_spec": build_environment_spec}
    adapter._update_scope = dict(
        stats=NS(export=lambda: {}),
        execution_stats=NS(export=lambda: {"scope": "scripted fixture"}),
        diagnostics={},
        detail={},
        channel_clips=torch.zeros(6),
        valid_samples=torch.tensor(128.0),
    )
    adapter.physical_readback = lambda: {"physics_steps": 0}
    original = copy.deepcopy(adapter.spec)
    for update in (100, 200):
        stats = env.onpolicy_episodes
        for name in ("count", "complete", "passed", "quality"):
            getattr(stats, name)[:] = 16
        stats.group_sums[:, [0, 12, 14]] = 16
        stats.group_sums[:, 1:3] = torch.tensor(old_stage["commands"]) * 16
        context.updates_used = update
        metrics = {}
        adapter.after_update(context, update, metrics)
    assert adapter._needs_rebuild and adapter.env is env
    assert env.stage_spec == original["curriculum"]["stages"][0]
    actual = metrics["environment_rebuild"]
    expected = rebuild_attribution(rebuild_inputs(original, job), rebuild_inputs(adapter.spec, job))
    assert actual == expected
    assert "cohort_keys" in actual["reasons"] and "physics" in actual["reasons"]
    assert "statistics_revision_only" not in actual["reasons"]
    assert json.loads(json.dumps(metrics))["environment_rebuild"] == actual
