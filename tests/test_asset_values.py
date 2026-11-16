"""SHA-bound recipe values through config, relocation and immutable run records."""

import copy
from pathlib import Path

import pytest

from native_stand.artifacts.io import file_sha256, read_json, write_json
from native_stand.artifacts.runtime_paths import RuntimePaths, asset_values, execution_job
from native_stand.experiments.config import job_from_record, load_job_config

FROZEN_TEST_SCOPE = "PORTABLE"
ROOT = Path(__file__).resolve().parents[1]


def test_value_file_is_sha_bound_and_relative_paths_use_the_owning_document(tmp_path, monkeypatch):
    directory = tmp_path / "contracts"
    path = write_json(
        directory / "schema.json",
        dict(width=3, source=dict(path="inner.bin", sha256="a" * 64), prim="/World/robot"),
    )
    assets = dict(
        value_files={"schema": dict(path="contracts/schema.json", sha256=file_sha256(path))},
        model=dict(path="model.pt", sha256="b" * 64),
        ordinary="relative/text",
    )
    original = copy.deepcopy(assets)
    monkeypatch.chdir(tmp_path.parent)
    value = asset_values(assets, tmp_path, RuntimePaths())
    assert assets == original
    assert value["schema"]["source"]["path"] == str(directory / "inner.bin")
    assert value["schema"]["prim"] == "/World/robot" and value["ordinary"] == "relative/text"
    assert value["model"]["path"] == str(tmp_path / "model.pt")
    path.write_text('{"width": 4}')
    with pytest.raises(ValueError, match="SHA differs"):
        asset_values(assets, tmp_path, RuntimePaths())


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        {"schema": "file.json"},
        {"schema": {"path": "a"}},
        {"schema": {"path": "a", "sha256": "wrong"}},
    ],
)
def test_malformed_value_references_are_rejected_at_the_config_boundary(tmp_path, invalid):
    with pytest.raises(ValueError, match="assets.value_files"):
        asset_values(dict(value_files=invalid), tmp_path, RuntimePaths())


def test_bound_value_cannot_overwrite_inline_assets(tmp_path):
    with pytest.raises(ValueError, match="overwrite"):
        asset_values(dict(schema={}, value_files={"schema": {}}), tmp_path, RuntimePaths())


def test_relocation_applies_after_relative_resolution_and_keeps_source_read_only(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    target = write_json(
        new / "contracts/data.json", dict(weight=dict(path=str(old / "model.pt"), sha256="b" * 64))
    )
    paths = RuntimePaths(prefixes=((str(old), str(new)),))
    values = asset_values(
        dict(value_files={"data": dict(path="contracts/data.json", sha256=file_sha256(target))}),
        old,
        paths,
    )
    assert values["value_files"]["data"]["path"] == str(target)
    assert values["data"]["weight"]["path"] == str(new / "model.pt")
    assert read_json(target)["weight"]["path"] == str(old / "model.pt")
    assert not old.exists()


def test_formal_job_expands_once_and_record_restores_without_rereading_source(tmp_path):
    recipe = read_json(ROOT / "configs/jobs/motion_supported015_dqgain.json")
    values = write_json(tmp_path / "goals.json", dict(height=0.34))
    recipe.setdefault("assets", {})["value_files"] = dict(
        example_goals=dict(path="goals.json", sha256=file_sha256(values))
    )
    config = write_json(tmp_path / "recipe.json", recipe)
    job = load_job_config(config)
    assert job.assets["example_goals"] == dict(height=0.34)
    record = job.as_run_record()
    values.unlink()  # Restoring a frozen run consumes its resolved record, not current recipe bytes.
    config.unlink()
    restored = job_from_record(record)
    assert restored.as_run_record() == record
    assert execution_job(restored).as_run_record() == record


def test_config_maps_each_absolute_reference_once_even_with_overlapping_rules(tmp_path):
    recipe = read_json(ROOT / "configs/jobs/motion_supported015_dqgain.json")
    old, new, wrong = (tmp_path / name for name in ("old", "new", "wrong"))
    values = write_json(new / "goals.json", dict(target=str(old / "model.pt")))
    mapping = write_json(
        tmp_path / "paths.json",
        dict(
            version="runtime_path_mapping_v1",
            prefixes={str(old): str(new), str(new): str(wrong)},
        ),
    )
    recipe.setdefault("assets", {}).update(
        path_mapping=dict(path=str(mapping), sha256=file_sha256(mapping)),
        value_files=dict(goals=dict(path=str(old / "goals.json"), sha256=file_sha256(values))),
    )
    job = load_job_config(write_json(tmp_path / "recipe.json", recipe))
    assert job.assets["value_files"]["goals"]["path"] == str(values)
    assert job.assets["goals"]["target"] == str(new / "model.pt")
    assert not wrong.exists()
