"""Portable storage boundary and adoption consumers, without training or engines."""

from pathlib import Path

import pytest

from native_stand.artifacts.io import file_sha256, read_json
from native_stand.artifacts.model_refs import ModelIndex
from native_stand.artifacts.runtime_paths import RuntimePaths, resolve_runs_root
from native_stand.integrations.cloud_bundle import adopt_local_dependencies

FROZEN_TEST_SCOPE = "PORTABLE"


def test_output_resolution_is_explicit_and_side_effect_free(tmp_path, monkeypatch):
    monkeypatch.delenv("WHEEL_LEG_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="WHEEL_LEG_DATA_ROOT"):
        resolve_runs_root()
    root = tmp_path / "data"
    monkeypatch.setenv("WHEEL_LEG_DATA_ROOT", str(root))
    assert resolve_runs_root() == root
    assert resolve_runs_root(tmp_path / "explicit") == tmp_path / "explicit"
    assert not root.exists()
    monkeypatch.setenv("WHEEL_LEG_DATA_ROOT", "relative")
    with pytest.raises(ValueError, match="绝对"):
        resolve_runs_root()


def test_owner_rejects_without_storage_before_preflight(monkeypatch):
    from native_stand.experiments.owner import submit

    monkeypatch.delenv("WHEEL_LEG_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="WHEEL_LEG_DATA_ROOT"):
        submit(None)


def test_adoption_repeat_relocation_and_multiple_consumers(tmp_path):
    source = tmp_path / "old.pt"
    source.write_bytes(b"immutable model")
    root = tmp_path / "data"
    target = root / "history" / "old.pt"
    index = ModelIndex(root / "models.json")
    sha = file_sha256(source)
    rows = [
        dict(
            path=str(source),
            target=str(target),
            sha256=sha,
            immutable=True,
            consumers=[
                dict(consumer="teacher", role="teacher"),
                dict(consumer="resume", role="resume"),
            ],
        )
    ]

    def adopt():
        return adopt_local_dependencies(
            rows,
            root,
            index=index,
            mapping_path=root / "paths.json",
            receipt_path=root / "receipt.json",
        )

    assert adopt()["new_allocated_bytes"] == 0
    assert adopt()["dependencies"][0]["storage"] == "existing_verified"
    assert len(index.rows()) == 1
    index.release(consumer="teacher", role="teacher")
    assert str(target) in index.consumer_paths()
    source.unlink()  # Only disposable fixture; target remains loadable without old path.
    paths = RuntimePaths.from_assets(
        {
            "path_mapping": {
                "path": str(root / "paths.json"),
                "sha256": file_sha256(root / "paths.json"),
            }
        }
    )
    assert Path(paths.resolve(str(source))).read_bytes() == b"immutable model"
    assert read_json(root / "receipt.json")["removed_originals"] == []


@pytest.mark.parametrize("failure", ["source", "target", "escape"])
def test_adoption_rejects_conflicts_without_publishing(tmp_path, failure):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    root = tmp_path / "data"
    root.mkdir()
    target = root / "target"
    sha = file_sha256(source)
    if failure == "source":
        sha = "0" * 64
    if failure == "target":
        target.write_bytes(b"conflict")
    if failure == "escape":
        target = tmp_path / "escape"
    with pytest.raises(ValueError):
        adopt_local_dependencies(
            [dict(path=str(source), target=str(target), sha256=sha, consumers=[])],
            root,
            index=ModelIndex(root / "models.json"),
            mapping_path=root / "paths.json",
            receipt_path=root / "receipt.json",
        )
    assert not (root / "paths.json").exists()


def test_worktrees_share_output_and_index(tmp_path, monkeypatch):
    from native_stand.experiments.config import load_job_config
    from native_stand.experiments.owner import create_run

    project = Path(__file__).resolve().parents[1]
    job = load_job_config(project / "configs/jobs/motion_supported015_dqgain.json")
    root = tmp_path / "data"
    monkeypatch.setenv("WHEEL_LEG_DATA_ROOT", str(root))
    for name in ("worktree-a", "worktree-b"):
        cwd = tmp_path / name
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        run = create_run(job, run_id=name)
        assert run.parent == root
        assert read_json(run / "run.json")["model_index"] == str(root / "models.json")


def test_mutable_dependency_is_copied_and_sha_mismatch_after_restart_rejected(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"version": 1}')
    root = tmp_path / "data"
    row = dict(
        path=str(source),
        target=str(root / "state.json"),
        sha256=file_sha256(source),
        consumers=[dict(consumer="restore", role="resume")],
    )
    kwargs = dict(
        index=ModelIndex(root / "models.json"),
        mapping_path=root / "paths.json",
        receipt_path=root / "receipt.json",
    )
    result = adopt_local_dependencies([row], root, **kwargs)
    assert result["dependencies"][0]["storage"] == "verified_copy"
    assert source.stat().st_ino != (root / "state.json").stat().st_ino
    source.write_text('{"version": 2}')
    with pytest.raises(ValueError, match="Source SHA"):
        adopt_local_dependencies([row], root, **kwargs)
    assert read_json(root / "state.json")["version"] == 1


def test_mapping_publication_failure_keeps_protection_and_replays(tmp_path, monkeypatch):
    from native_stand.artifacts import io

    source = tmp_path / "model.pt"
    source.write_bytes(b"checkpoint")
    root = tmp_path / "data"
    index = ModelIndex(root / "models.json")
    row = dict(
        path=str(source),
        target=str(root / "model.pt"),
        sha256=file_sha256(source),
        consumers=[dict(consumer="resume", role="resume")],
        immutable=True,
    )
    mapping = root / "paths.json"
    write = io.write_json

    def fail_mapping(path, data):
        if Path(path) == mapping:
            raise OSError("publication interrupted")
        return write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(io, "write_json", fail_mapping)
        with pytest.raises(OSError, match="interrupted"):
            adopt_local_dependencies(
                [row], root, index=index, mapping_path=mapping, receipt_path=root / "receipt.json"
            )
    assert not mapping.exists()
    assert str(root / "model.pt") in index.consumer_paths()
    result = adopt_local_dependencies(
        [row], root, index=index, mapping_path=mapping, receipt_path=root / "receipt.json"
    )
    assert result["dependencies"][0]["storage"] == "existing_verified"
    assert len(index.rows()) == 1
