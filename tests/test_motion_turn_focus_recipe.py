"""Turn-focused recipe through real course/statistics/source/adapter CPU consumers."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from test_motion_spin_branches import equal_tree, fixture, report
from test_motion_window_accounting import environment

from native_stand.artifacts.branch_initialization import PREPARATION, require_allocated
from native_stand.artifacts.checkpoint import load_complete_state, save_full
from native_stand.artifacts.io import file_sha256, json_digest, read_json, write_json
from native_stand.learning.ppo.supported import assemble_from_full
from native_stand.simulation.isaac.motion_backend import (
    MotionProductionAdapter,
    _check_source_progress,
    _resolve_source,
    validate_job,
)
from native_stand.tasks.motion.branch_state import confirmation_execution_identity
from native_stand.tasks.motion.course_strategy import resolve
from native_stand.tasks.motion.maneuver_statistics import RESPONSE_VERSION, episode_class

FROZEN_TEST_SCOPE = "PORTABLE"
ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "configs/jobs/motion_u1500_turn_focus_v3_prepared.json"


def configuration():
    return read_json(JOB)["assets"]["branch_course"]


def initial():
    cfg = configuration()
    strategy = resolve(cfg["version"])
    return strategy, strategy.initial_state({"branch_course": cfg})


def review(state, update=100):
    result = report(state, "motion")
    result["through_update"] = update
    for row in result["rows"]:
        row["profile"] = row["key"][4]
    return result


def test_weighted_layout_keeps_base_quota_and_rough_frontiers_pauses_height():
    strategy, state = initial()
    rows = strategy.implementation.task_rows(state, 4096)
    assert sum(r["count"] for r in rows) == 4096
    turn = sum(r["count"] for r in rows if r["axis"].startswith("turn_"))
    rough = sum(r["count"] for r in rows if r["axis"].startswith("rough_"))
    assert turn / 4096 == pytest.approx(0.36, abs=0.002)
    assert rough / 4096 == pytest.approx(0.09, abs=0.002)
    assert sum(r["count"] for r in rows if not r["purpose"].startswith("focus_")) == pytest.approx(
        4096 * 0.55, abs=1
    )
    assert all(r["goal_height"] == 0.34 for r in rows)
    assert state["levels"]["rough_broken_rubble_H1"] == 1
    assert state["levels"]["rough_pitted_road_H1"] == 1
    assert not any(state["streaks"].values())
    assert all(r["prefix"] == 0 for r in rows)
    turns = state["configuration"]["targets"]["turn_fp"]
    assert [t[1] for t in turns[:4]] == [0.5, 0.65, 0.8, 1.0]
    assert turns[-1] == [3.6, 0.75, 0.34]


@pytest.mark.parametrize(
    "change", ["weight_missing", "weight_negative", "weight_sum", "gain", "level", "gate"]
)
def test_invalid_focus_recipe_rejected_at_configuration_boundary(change):
    cfg = configuration()
    if change == "weight_missing":
        cfg["focus_weights"].pop("turn_fp")
    elif change == "weight_negative":
        cfg["focus_weights"]["turn_fp"] = -0.2
    elif change == "weight_sum":
        cfg["focus_weights"]["turn_fp"] = 0.4
    elif change == "gain":
        cfg["minimum_signed_yaw_gain"] = float("nan")
    elif change == "level":
        cfg["initial_levels"]["turn_fp"] = len(cfg["targets"]["turn_fp"])
    else:
        cfg["retention_gate"]["nominal_pass"] = True
    with pytest.raises(ValueError):
        resolve(cfg["version"]).initial_state({"branch_course": cfg})


@pytest.mark.parametrize("kind", ["nominal", "DR", "unknown"])
def test_base_failure_or_unknown_blocks_focus_promotion_without_fake_regression(kind):
    strategy, state = initial()
    for update in (100, 200):
        r = review(state, update)
        row = next(
            x
            for x in r["rows"]
            if x["purpose"] == ("anchor_backward_0" if kind == "DR" else "clean_standing")
        )
        if kind == "unknown":
            row.update(ended_episodes=0, completed_episodes=0, pass_fraction=None)
        else:
            row["pass_fraction"] = 0.49 if kind == "DR" else 0.94
        state, decision = strategy.advance(state, r, update, envs=2560)
        assert not any(x["promoted"] for x in decision["decisions"].values())
        assert not any(x["regressed"] for x in decision["decisions"].values())
        assert decision["retention_gate"]["status"] == (
            "UNKNOWN" if kind == "unknown" else "FAILED"
        )
        assert not any(state["streaks"][a] for a in state["levels"])


def test_focus_promotes_with_base_retained_and_serialized_confirmation():
    strategy, state = initial()
    first = review(state)
    # DR is stable enough for retention but not for expanding shared DR strength.
    for row in first["rows"]:
        if row["profile"] == 0 and not row["purpose"].startswith("focus_"):
            row["pass_fraction"] = 0.6
    one, decision = strategy.advance(state, first, 100, envs=2560)
    restored = copy.deepcopy(one)
    second = review(restored, 200)
    for row in second["rows"]:
        if row["profile"] == 0 and not row["purpose"].startswith("focus_"):
            row["pass_fraction"] = 0.6
    two, decision = strategy.advance(restored, second, 200, envs=2560)
    assert decision["retention_gate"]["status"] == "PASSED"
    assert decision["decisions"]["turn_fp"]["promoted"]
    assert not decision["DR_promoted"] and two["general_DR_frontier"] == 1
    assert two["configuration"]["targets"]["turn_fp"][two["levels"]["turn_fp"]][1] == 0.65


@pytest.mark.parametrize(
    "gain,minimum,passed",
    [(0.6, None, True), (0.6, 0.75, False), (0.8, 0.75, True), (-0.8, 0.75, False)],
)
def test_actual_episode_scoring_rejects_underturn_and_preserves_old_default(gain, minimum, passed):
    e = environment()
    e.student_actor_valid[:] = True
    e.stage_spec["commands"] = [[2, 0.5], [2, 0.5]]
    if minimum is not None:
        e.stage_spec["minimum_signed_yaw_gain"] = minimum
    e.motion_action_command[:] = torch.tensor([2, 0.5, 0.34])
    e.recovery_snapshot[:, 7] = 2
    e.recovery_snapshot[:, 9] = 0.5 * gain
    s = e.onpolicy_episodes
    # Scripted sums for the preceding 1499 steady samples, then the real consumer
    # processes the timeout sample. This is no physical integration or rollout.
    s.n[:] = 2999
    s.hold[:] = 1499
    values = torch.tensor([1, 2, 0.5 * gain, 0, 0, abs(0.5 * gain - 0.5), 0, 0, 0, 0, 0, 0, 0])
    s.sums[:] = values * 1499
    s.observe_final((torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)))
    result = s.export(100)
    assert result["rows"][0]["quality_pass_fraction"] == 1
    assert result["rows"][0]["pass_fraction"] == float(passed)
    assert result["rows"][0]["minimum_signed_yaw_gain"] == (minimum or 0.5)


def prepared_fixture(tmp_path):
    job, source, obs = fixture(tmp_path)
    source["model_online_updates"] = 5011
    cp = tmp_path / "full.pt"
    save_full(cp, source)

    def ref(p):
        return dict(path=str(p), sha256=file_sha256(p))

    original = job.as_run_record()
    original["assets"].update(historical_online_updates=2097, original_online_limit=10097)
    run, state, budget = (tmp_path / n for n in ("run.json", "state.json", "budget.json"))
    write_json(run, dict(run_id="previous", config=original))
    write_json(
        state,
        dict(
            run_id="previous",
            status="RUNNING",
            updates_used=3055,
            latest_model=dict(checkpoint=str(cp), sha256=file_sha256(cp), update=3000),
        ),
    )
    write_json(budget, dict(limits={"smoke": 0, "formal": 8000}, used={"smoke": 0, "formal": 3055}))
    cfg = configuration()
    job = replace(
        job,
        curriculum=replace(job.curriculum, version=cfg["version"], stages=(cfg,)),
        initialization=replace(job.initialization, checkpoint=cp, sha256=file_sha256(cp)),
        assets={
            **job.assets,
            "branch_course": cfg,
            "historical_online_updates": 5152,
            "original_online_limit": 10097,
            "preparation_only": True,
        },
    )
    cfg["confirmation_execution_sha256"] = confirmation_execution_identity(job)
    record = dict(
        version=PREPARATION,
        branch="motion",
        branch_updates=0,
        budget_status="PREPARATION_ONLY",
        source={**ref(cp), "model_local_update": 3000, "full_checkpoint": True},
        source_run=ref(run),
        observed_state=ref(state),
        observed_budget=ref(budget),
        output_contract=dict(
            path=job.assets["output_contract"], sha256=job.assets["output_contract_sha256"]
        ),
        center_binding=job.assets["support_center_policy_binding"],
        source_model_online_updates=5097,
        source_model_progress_updates=5011,
        authorization_spent_updates=5152,
        shared_remaining_updates=4945,
        rolled_back_updates_not_refunded=True,
        target_course_sha256=json_digest(cfg),
    )
    prep = tmp_path / "preparation.json"
    write_json(prep, record)
    return replace(job, assets={**job.assets, "branch_initialization": ref(prep)}), source, obs


def test_prepared_full_source_keeps_two_Adam_RNG_std_and_55_charged_updates(tmp_path):
    job, source, obs = prepared_fixture(tmp_path)
    validate_job(job)
    loaded = _resolve_source(job, None)
    assert _check_source_progress(job, loaded.payload, None) == 0
    assert (
        job.assets["historical_online_updates"] - loaded.payload["online_updates_including_smoke"]
        == 55
    )
    assert loaded.payload["model_online_updates"] == 5011
    assert loaded.payload["warmup_updates_done"] == 50
    for key in ("actor", "critic", "optimizer", "std", "training_rng"):
        equal_tree(source[key], loaded.payload[key])
    algorithm = assemble_from_full(loaded.payload, obs, job)
    equal_tree(source["optimizer"], algorithm.optimizer.state_dict())
    assert algorithm.actor_update_enabled
    with pytest.raises(ValueError, match="课程准备不是训练分配"):
        require_allocated(job)
    with pytest.raises(ValueError, match="只允许零额度"):
        validate_job(replace(job, limits=replace(job.limits, formal_updates=1)))


@pytest.mark.parametrize("change", ["reward", "KL", "target", "refund", "snapshot"])
def test_preparation_rejects_contract_changes_and_accounting_tampering(tmp_path, change):
    job, _, _ = prepared_fixture(tmp_path)
    assets = copy.deepcopy(job.assets)
    if change == "reward":
        job = replace(job, reward=replace(job.reward, dt=0.02))
    elif change == "KL":
        assets["source_policy_retention"]["coefficient"] = 0.03
    elif change == "target":
        assets["branch_course"]["minimum_signed_yaw_gain"] = 0.9
    else:
        p = Path(assets["branch_initialization"]["path"])
        record = read_json(p)
        if change == "refund":
            record["authorization_spent_updates"] -= 55
        else:
            Path(record["observed_state"]["path"]).write_text("{}")
        write_json(p, record)
        assets["branch_initialization"]["sha256"] = file_sha256(p)
    with pytest.raises(ValueError):
        _resolve_source(replace(job, assets=assets), None)


def test_formal_adapter_capture_and_same_run_restore_preserve_preparation(tmp_path, monkeypatch):
    from test_phase3_fixes import _production_context

    job, original, obs = prepared_fixture(tmp_path)
    ctx = _production_context(tmp_path / "adapter", job, smoke=1)

    def allocate(adapter):
        adapter.env = NS(
            num_envs=4,
            device="cpu",
            stage_spec=adapter.spec["curriculum"]["stages"][0],
            identified_dr=NS(episodes=torch.full((4,), 17)),
            transient_events=NS(summary=lambda: {}),
        )
        adapter.env.onpolicy_episodes = episode_class(object, version=RESPONSE_VERSION)(adapter.env)
        adapter.obs = obs
        from native_stand.students.resume import rng_state

        adapter.modules = {"rng_state": rng_state}

    monkeypatch.setattr(MotionProductionAdapter, "_open_environment", allocate)
    monkeypatch.setattr(MotionProductionAdapter, "_write_restoration", lambda self: None)
    adapter = MotionProductionAdapter()
    try:
        adapter.build(ctx, "formal", None)
        assert adapter.env.stage_spec["minimum_signed_yaw_gain"] == 0.75
        saved = adapter.capture(ctx, {})
        assert saved["counters"]["online_updates"] == 5152
        assert saved["model_online_updates"] == 5011
        equal_tree(original["optimizer"], saved["optimizer"])
        cp = ctx.save(saved, update=0, stage="formal")
        loaded = load_complete_state(cp["checkpoint"])
        restored = MotionProductionAdapter()
        restored.build(ctx, "formal", loaded)
        assert restored.state == adapter.state
        equal_tree(saved["optimizer"], restored.algorithm.optimizer.state_dict())
    finally:
        ctx.metrics.close()
