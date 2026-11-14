"""Bounded diagnosis reuses the production player; portable tests do not integrate physics."""

import copy

import numpy as np
import pytest

from native_stand.artifacts.io import write_json
from native_stand.simulation import motion_support_probe as probe

FROZEN_TEST_SCOPE = "PORTABLE"


def case(**kw):
    return dict(
        name="flat",
        height_m=0.2,
        vx_mps=0.0,
        yaw_rad_s=0.0,
        leg_scale_rad=0.15,
        duration_s=20.0,
        mapping="rescale_source_raw",
        **kw,
    )


def config():
    return dict(version=probe.VERSION, cases=[case()], wall_s=10, max_output_bytes=32 * 2**20)


@pytest.mark.parametrize(
    "field,value",
    [
        ("height_m", 0.19),
        ("leg_scale_rad", 0.4),
        ("duration_s", float("nan")),
        ("mapping", "implicit"),
    ],
)
def test_reject_ambiguous_or_unbounded_interventions(field, value):
    cfg = config()
    cfg["cases"][0][field] = value
    with pytest.raises(ValueError):
        probe.validate(cfg)


def test_intervention_changes_only_explicit_legs_and_does_not_mutate_teacher():
    c = case()
    c["leg_scale_rad"] = 0.5
    raw = np.array([2.0, -0.5, 1.0, -2.0, 0.25, -1.0])
    target = np.arange(6, dtype=float)
    saved = target.copy()
    result = probe.transform_for(c)(raw, 0.2, target)
    np.testing.assert_allclose(
        result[[0, 1, 3, 4]], saved[[0, 1, 3, 4]] + 0.35 * np.clip(raw[[0, 1, 3, 4]], -1, 1)
    )
    np.testing.assert_array_equal(target, saved)
    np.testing.assert_array_equal(result[[2, 5]], target[[2, 5]])


def rows():
    return [
        dict(
            time=t,
            height=0.2,
            dual_support=True,
            vx=0.0,
            yaw=0.0,
            roll=0.0,
            pitch=0.0,
            dq=np.zeros(6),
            q=np.zeros(6),
            adopted=np.zeros(6),
            raw=np.zeros(6),
            source_requested=np.zeros(6),
            motor=np.zeros(6),
            NET=np.zeros(6),
            position_xy=np.zeros(2),
        )
        for t in np.arange(0.01, 20.001, 0.01)
    ]


def test_only_complete_steady_stationary_loaded_target_can_form_center_candidate(tmp_path):
    sample = rows()
    result = probe.summarize(sample, case(), termination=None)
    write_json(tmp_path / "summary.json", result)
    assert result["center_candidate_eligible"] and not result["capability_accepted"]
    assert not probe.summarize(sample, case(), termination="wall_limit")[
        "center_candidate_eligible"
    ]
    bad = copy.deepcopy(sample)
    for row in bad:
        row["height"] = 0.225
    assert not probe.summarize(bad, case(), termination=None)["center_candidate_eligible"]
    moving = case()
    moving["vx_mps"] = 0.25
    assert not probe.summarize(sample, moving, termination=None)["center_candidate_eligible"]
    assert probe.summarize(sample[:50], case(), termination=None)["status"] == "UNKNOWN"


def test_cli_dry_run_reuses_plan_never_constructs_engine(tmp_path, monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    from native_stand.cli import main
    from native_stand.simulation import playback

    cfg = config()
    cfg.update(
        run="frozen-run",
        checkpoint="model.pt",
        checkpoint_sha256="a" * 64,
        path_mapping="mapping.json",
        solver_profile="closure_contact_tight_v1",
    )
    prepared = SimpleNamespace(
        job=SimpleNamespace(task="motion", action=SimpleNamespace(leg_scale_rad=0.15)),
        record=lambda: {"source": "frozen"},
    )
    monkeypatch.setattr(playback, "plan", lambda *a, **kw: prepared)
    monkeypatch.setattr(probe, "run", lambda *a: pytest.fail("dry-run started integration"))
    path = tmp_path / "config.json"
    write_json(path, cfg)
    assert main(["diagnose", "--config", str(path), "--dry-run", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["starts_engine"] and not result["writes"] and result["updates_charged"] == 0
