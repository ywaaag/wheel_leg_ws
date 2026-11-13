"""Height sampling and remaining-budget continuation through real CPU consumers."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_motion_cloud_allocation import allocated
from test_motion_turn_focus_recipe import equal_tree

from native_stand.artifacts.branch_initialization import CONTINUED_COURSE, require_allocated
from native_stand.artifacts.io import file_sha256, read_json, write_json
from native_stand.simulation.isaac.motion_backend import _resolve_source, validate_job
from native_stand.tasks.motion.course_strategy import resolve
from native_stand.tasks.motion.maneuver_program import extended_tensor_program

FROZEN_TEST_SCOPE = "PORTABLE"
RECIPE = Path(__file__).resolve().parents[1] / "configs/jobs/motion_height_small_dose_prepared.json"


def test_height_quota_and_height_first_motion_cover_both_directions():
    cfg = read_json(RECIPE)["assets"]["branch_course"]
    strategy = resolve(cfg["version"])
    state = strategy.initial_state({"branch_course": cfg})
    rows = strategy.implementation.task_rows(state, 4096)
    height_rows = [r for r in rows if r["axis"].startswith("height_")]
    assert sum(r["count"] for r in height_rows) / 4096 == pytest.approx(0.10, abs=0.006)
    assert {r["goal_height"] for r in height_rows} == {0.2, 0.24, 0.28, 0.32, 0.36, 0.4}
    assert {r["target"][0] for r in height_rows} == {0, 0.25, -0.25}
    assert all(r["count"] >= 10 for r in height_rows)
    assert len(rows) <= 128
    assert all(r["target"][1] == 0 and r["prefix"] == 0 for r in height_rows)
    assert not any(r["axis"].startswith("height_") and "rough" in r["purpose"] for r in rows)
    for goal in (0.2, 0.24, 0.28, 0.32, 0.36, 0.4):
        ages = torch.tensor(
            [
                1.0,
                1 + abs(goal - 0.34) / 0.02,
                2 + abs(goal - 0.34) / 0.02,
                4 + abs(goal - 0.34) / 0.02,
            ]
        )
        cmd, _ = extended_tensor_program(
            ages, torch.tensor([[-0.25, 0.0]]).expand(4, -1), torch.full((4,), goal)
        )
        assert cmd[:3, 0].tolist() == [0, 0, 0]
        assert cmd[-1, 0].item() == -0.25
        assert torch.allclose(cmd[1:, 2], torch.full((3,), goal))


def continued(tmp_path):
    job, source, obs = allocated(tmp_path)
    path = Path(job.assets["branch_initialization"]["path"])
    record = read_json(path)
    record.update(version=CONTINUED_COURSE, retired_remaining_updates=0)
    record.pop("new_authorization")
    settlementpath = Path(record["settlement"]["path"])
    settlement = read_json(settlementpath)
    settlement["retired_remaining_updates"] = 0
    write_json(settlementpath, settlement)
    record["settlement"]["sha256"] = file_sha256(settlementpath)
    planpath = Path(record["allocation"]["path"])
    plan = read_json(planpath)
    plan.update(settlement_sha256=file_sha256(settlementpath), remaining_before_allocation=4945)
    plan["allocations"][0]["formal_updates"] = 4945
    write_json(planpath, plan)
    record["allocation"]["sha256"] = file_sha256(planpath)
    write_json(path, record)
    assets = copy.deepcopy(job.assets)
    assets.update(
        branch_initialization={"path": str(path), "sha256": file_sha256(path)},
        new_independent_authorization=False,
        original_online_limit=10097,
    )
    return replace(job, assets=assets, limits=replace(job.limits, formal_updates=4945)), source, obs


def test_remaining_only_keeps_full_learning_state_and_no_refund(tmp_path, monkeypatch):
    job, source, _ = continued(tmp_path)
    from native_stand.tasks.motion import contact_randomization, height_extension

    def forbidden(*args):
        raise AssertionError("complete 881 continuation must not repeat any extension")

    monkeypatch.setattr(contact_randomization, "extend_source", forbidden)
    monkeypatch.setattr(height_extension, "extend_source", forbidden)
    validate_job(job)
    assert require_allocated(job)["formal_updates"] == 4945
    loaded = _resolve_source(job, None).payload
    for key in (
        "actor",
        "critic",
        "optimizer",
        "std",
        "training_rng",
        "model_online_updates",
        "warmup_updates_done",
    ):
        equal_tree(source[key], loaded[key])
    assert job.assets["historical_online_updates"] - source["online_updates_including_smoke"] == 55


@pytest.mark.parametrize(
    "change", ["extra_budget", "grant", "running", "refund", "retire", "reward"]
)
def test_remaining_allocation_rejects_refresh_and_unsafe_source(tmp_path, change):
    job, _, _ = continued(tmp_path)
    path = Path(job.assets["branch_initialization"]["path"])
    record = read_json(path)
    if change == "extra_budget":
        job = replace(job, limits=replace(job.limits, formal_updates=4946))
    elif change == "grant":
        record["new_authorization"] = {"path": "invalid", "sha256": "0" * 64}
    elif change == "refund":
        record["authorization_spent_updates"] -= 55
    elif change == "retire":
        record["retired_remaining_updates"] = 4945
    elif change == "reward":
        job = replace(job, reward=replace(job.reward, dt=0.02))
    else:
        p = Path(record["observed_state"]["path"])
        d = read_json(p)
        d["status"] = "RUNNING"
        write_json(p, d)
        record["observed_state"]["sha256"] = file_sha256(p)
    write_json(path, record)
    job.assets["branch_initialization"]["sha256"] = file_sha256(path)
    with pytest.raises(ValueError):
        require_allocated(job)
        _resolve_source(job, None)
