"""Allocated Motion source/worker boundaries; no cloud/GPU/physics execution."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
from test_motion_turn_focus_recipe import prepared_fixture

from native_stand.artifacts.branch_initialization import ALLOCATED_COURSE, require_allocated
from native_stand.artifacts.io import file_sha256, read_json, write_json
from native_stand.simulation.isaac.motion_backend import (
    _check_source_progress,
    _resolve_source,
    validate_job,
)

FROZEN_TEST_SCOPE = "PORTABLE"


def allocated(tmp_path):
    job, source, obs = prepared_fixture(tmp_path)

    def ref(p):
        return dict(path=str(p), sha256=file_sha256(p))

    prep = Path(job.assets["branch_initialization"]["path"])
    record = read_json(prep)
    statepath = Path(record["observed_state"]["path"])
    state = read_json(statepath)
    state["status"] = "FAILED"
    write_json(statepath, state)
    budgetpath = Path(record["observed_budget"]["path"])
    book = read_json(budgetpath)
    book["attempts"] = {"old": {"status": "TERMINATED"}}
    write_json(budgetpath, book)
    record.update(
        version=ALLOCATED_COURSE,
        budget_status="MOTION_TRAINING",
        observed_state=ref(statepath),
        observed_budget=ref(budgetpath),
        retired_remaining_updates=4945,
    )
    settlement = tmp_path / "settlement.json"
    write_json(
        settlement,
        dict(
            state=record["observed_state"],
            budget=record["observed_budget"],
            consumed_total=5152,
            retired_remaining_updates=4945,
        ),
    )
    record["settlement"] = ref(settlement)
    grant = tmp_path / "grant.json"
    write_json(
        grant,
        dict(
            status="APPROVED",
            run_id=job.name,
            branch="motion",
            updates=10000,
            preserved_consumption=5152,
            wall_clock="unlimited",
        ),
    )
    record["new_authorization"] = ref(grant)
    plan = tmp_path / "allocation.json"
    entry = dict(
        run_id=job.name,
        branch="motion",
        smoke_updates=0,
        formal_updates=10000,
        source_sha256=job.initialization.sha256,
        output_contract_sha256=job.assets["output_contract_sha256"],
        execution_sha256=job.assets["branch_course"]["confirmation_execution_sha256"],
    )
    write_json(
        plan,
        dict(
            version="selected_branch_shared_allocations_v1",
            settlement_sha256=record["settlement"]["sha256"],
            already_consumed_updates=5152,
            remaining_before_allocation=10000,
            allocations=[entry],
        ),
    )
    record["allocation"] = ref(plan)
    write_json(prep, record)
    job = replace(
        job,
        limits=replace(job.limits, formal_updates=10000, chain_wall_s=None),
        assets={
            **job.assets,
            "branch_initialization": ref(prep),
            "preparation_only": False,
            "new_independent_authorization": True,
            "original_online_limit": 15152,
        },
    )
    return job, source, obs


def test_new_10000_is_not_added_to_retired_balance_or_refunded_model_gap(tmp_path):
    job, source, _ = allocated(tmp_path)
    validate_job(job)
    entry = require_allocated(job)
    assert entry["formal_updates"] == 10000 and job.limits.chain_wall_s is None
    loaded = _resolve_source(job, None)
    assert _check_source_progress(job, loaded.payload, None) == 0
    assert loaded.payload["model_online_updates"] == 5011
    assert (
        job.assets["historical_online_updates"] - loaded.payload["online_updates_including_smoke"]
        == 55
    )
    assert loaded.payload["warmup_updates_done"] == 50
    from test_motion_spin_branches import equal_tree

    equal_tree(source["optimizer"], loaded.payload["optimizer"])


@pytest.mark.parametrize(
    "change",
    [
        "running_state",
        "running_attempt",
        "refund",
        "retired",
        "grant",
        "allocation",
        "source_progress",
    ],
)
def test_allocation_rejects_changed_settlement_or_grant(tmp_path, change):
    job, _, _ = allocated(tmp_path)
    assets = copy.deepcopy(job.assets)
    prep = Path(assets["branch_initialization"]["path"])
    r = read_json(prep)
    if change == "running_state":
        p = Path(r["observed_state"]["path"])
        d = read_json(p)
        d["status"] = "RUNNING"
        write_json(p, d)
        r["observed_state"]["sha256"] = file_sha256(p)
    elif change == "running_attempt":
        p = Path(r["observed_budget"]["path"])
        d = read_json(p)
        d["attempts"]["old"]["status"] = "RUNNING"
        write_json(p, d)
        r["observed_budget"]["sha256"] = file_sha256(p)
    elif change == "refund":
        r["authorization_spent_updates"] -= 55
    elif change == "retired":
        r["retired_remaining_updates"] = 0
    elif change == "source_progress":
        r["source_model_progress_updates"] += 1
    else:
        key = "new_authorization" if change == "grant" else "allocation"
        p = Path(r[key]["path"])
        d = read_json(p)
        if change == "grant":
            d["preserved_consumption"] -= 55
        else:
            d["allocations"][0]["formal_updates"] += 4945
        write_json(p, d)
        r[key]["sha256"] = file_sha256(p)
    write_json(prep, r)
    assets["branch_initialization"]["sha256"] = file_sha256(prep)
    job = replace(job, assets=assets)
    with pytest.raises(ValueError):
        validate_job(job)
        _resolve_source(job, None)
