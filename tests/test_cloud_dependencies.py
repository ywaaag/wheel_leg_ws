"""Finite content-identified cloud closure; no engines, model loads or remote I/O."""

import json
from pathlib import Path

import pytest

from native_stand.artifacts.io import file_sha256, write_json
from native_stand.integrations.cloud_bundle import inventory, package

FROZEN_TEST_SCOPE = "PORTABLE"
ROOT = Path(__file__).resolve().parents[1]


def test_existing_bundle_materialization_and_cli_dry_run(tmp_path, capsys):
    from native_stand.artifacts.model_refs import ModelIndex
    from native_stand.cli import main
    from native_stand.integrations.cloud_bundle import materialize_dependencies

    original = tmp_path / "input.pt"
    original.write_bytes(b"immutable fixture")
    alias = tmp_path / "other.pt"
    alias.write_bytes(original.read_bytes())
    root, config = fixture(tmp_path, {"input": ref(original), "other": ref(alias)})
    plan = inventory([config], root)
    archive = tmp_path / "dependencies.tar"
    package(plan, archive)
    manifest = tmp_path / "plan.json"
    write_json(manifest, plan)
    data = tmp_path / "data"
    assert (
        main(
            [
                "cloud-plan",
                "--manifest",
                str(manifest),
                "--restore-dependencies",
                str(data),
                "--dependency-archive",
                str(archive),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    assert not data.exists()
    capsys.readouterr()
    original.unlink()
    alias.unlink()
    result = materialize_dependencies(plan, archive, data, source_root=root)
    mapping = json.loads((data / "runtime_paths.json").read_text())
    one, two = Path(mapping["files"][str(original)]), Path(mapping["files"][str(alias)])
    assert one.stat().st_ino == two.stat().st_ino
    assert file_sha256(one) == ref(one)["sha256"]
    index = ModelIndex(data / "models.json")
    assert str(one.resolve()) in index.consumer_paths()
    assert result["original_inode_hardlinks"] is False
    assert materialize_dependencies(plan, archive, data, source_root=root)["new_content_bytes"] == 0
    one.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing dependency SHA"):
        materialize_dependencies(plan, archive, data, source_root=root)


def test_bundle_materialization_rejects_mismatched_manifest(tmp_path):
    from native_stand.integrations.cloud_bundle import materialize_dependencies

    root, config = fixture(tmp_path, {})
    plan = inventory([config], root)
    archive = tmp_path / "dependencies.tar"
    package(plan, archive)
    with pytest.raises(ValueError, match="manifest identity"):
        materialize_dependencies(
            {**plan, "ready_for_training": True}, archive, tmp_path / "data", source_root=root
        )


def ref(path):
    return dict(path=str(path), sha256=file_sha256(path))


def fixture(tmp_path, assets):
    root = tmp_path / "source"
    for directory in ("src", "configs/jobs", "configs/branches", "tests"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for name in (
        "pyproject.toml",
        "README.md",
        "AGENTS.md",
        "ARCHITECTURE.md",
        "STATUS.md",
        "scripts/check_architecture.py",
        "docs/MIGRATION_20261105.md",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    write_json(root / "configs/model_refs.json", dict(fixed_models=[]))
    raw = json.loads((ROOT / "tests/fixtures/cpu_motion_job.json").read_text())
    raw["assets"] = assets
    config = root / "configs/jobs/test.json"
    write_json(config, raw)
    return root, config


def test_inventory_expands_allocation_prior_proofs_and_nested_center(tmp_path):
    state = tmp_path / "state.json"
    write_json(state, dict(status="FAILED", updates_used=0, latest_model=None))
    budget = tmp_path / "budget.json"
    write_json(budget, dict(used={"smoke": 0}, attempts={}))
    allocation = tmp_path / "allocation.json"
    write_json(
        allocation,
        dict(
            version="selected_branch_shared_allocations_v1",
            retired_unstarted_runs=[dict(state=ref(state), budget=ref(budget))],
        ),
    )
    branch = tmp_path / "branch.json"
    write_json(branch, dict(allocation=ref(allocation)))
    contract = tmp_path / "contract.json"
    write_json(contract, dict(action_version="frozen"))
    source_binding = tmp_path / "source_binding.json"
    write_json(
        source_binding,
        dict(output_contract=str(contract), output_contract_sha256=file_sha256(contract)),
    )
    center = tmp_path / "center.json"
    write_json(center, dict(source_binding=ref(source_binding)))
    # Use neutral asset keys to test inventory, not the unrelated branch validator.
    root, config = fixture(
        tmp_path,
        dict(
            bound_branch={**ref(branch), "attached_contract": ref(contract)},
            support_center_binding=ref(center),
            same_center_again=ref(center),
        ),
    )
    plan = inventory([config], root)
    assert not plan["unresolved"]
    paths = {r["path"]: r for r in plan["artifacts"]}
    for p in (branch, allocation, state, budget, center, source_binding, contract):
        assert str(p) in paths
    assert len(paths[str(center)]["consumers"]) == 2
    assert (
        plan["json_reference_closure"]["visited"] == 7
    )  # seven explicitly identified dependency documents
    archive = tmp_path / "portable.tar"
    package(plan, archive)
    assert archive.is_file()


def test_explicit_preparation_references_preserve_identity_and_consumers(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"complete checkpoint")
    root, config = fixture(tmp_path, {})
    refs = [{**ref(model), "role": role} for role in ("resume", "teacher")]
    plan = inventory([config], root, references=refs)
    assert not plan["unresolved"]
    row = next(row for row in plan["artifacts"] if row["path"] == str(model))
    assert row["consumers"] == ["resume", "teacher"]
    assert not plan["import_authorized"]
    refs[0]["sha256"] = "0" * 64
    bad = inventory([config], root, references=refs)
    assert bad["unresolved"][0]["reason"] == "sha_mismatch"
    with pytest.raises(ValueError, match="依赖闭包"):
        package(bad, tmp_path / "bad.tar")


def test_closure_checks_sha_and_does_not_follow_history_or_unbound_json(tmp_path):
    hidden = tmp_path / "not_for_training.pt"
    hidden.write_bytes(b"not consumed")
    child = tmp_path / "child.json"
    write_json(child, dict(required=ref(hidden)))
    historical = tmp_path / "historical.json"
    write_json(historical, dict(history=[ref(hidden)]))
    unsigned = tmp_path / "unsigned.json"
    write_json(unsigned, dict(required=ref(hidden)))
    root, config = fixture(
        tmp_path,
        dict(
            bad=dict(path=str(child), sha256="0" * 64),
            history_document=ref(historical),
            unsigned_path=str(unsigned),
        ),
    )
    plan = inventory([config], root)
    assert any(r["reason"] == "sha_mismatch" for r in plan["unresolved"])
    assert str(hidden) not in {r["path"] for r in plan["artifacts"]}
    try:
        package(plan, tmp_path / "bad.tar")
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched closure published")


def test_deep_json_closure_is_bounded_and_fail_closed(tmp_path):
    leaf = tmp_path / "leaf.json"
    write_json(leaf, dict(ok=True))
    for i in range(20):
        parent = tmp_path / f"level{i}.json"
        write_json(parent, dict(next=ref(leaf)))
        leaf = parent
    root, config = fixture(tmp_path, dict(chain=ref(leaf)))
    plan = inventory([config], root)
    assert any(r["reason"] == "json_reference_depth_limit" for r in plan["unresolved"])
    assert plan["json_reference_closure"]["visited"] <= 18


def test_nested_closure_relocates_by_manifest_without_old_paths(tmp_path):
    import shutil

    from native_stand.artifacts.runtime_paths import RuntimePaths

    original = tmp_path / "original"
    original.mkdir()
    leaf = original / "evidence.json"
    write_json(leaf, dict(completed_updates=27))
    parent = original / "binding.json"
    write_json(parent, dict(source_binding=ref(leaf)))
    root, config = fixture(tmp_path, dict(binding=ref(parent)))
    plan = inventory([config], root)
    relocated = tmp_path / "persistent"
    files = {}
    for row in plan["artifacts"]:
        target = relocated / row["archive_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
        files[row["path"]] = str(target)
    shutil.rmtree(original)  # /tmp fixture only
    paths = RuntimePaths(files=tuple(files.items()))
    binding = paths.read(str(parent))
    record = binding["source_binding"]
    assert file_sha256(record["path"]) == record["sha256"]
    assert paths.read(record["path"])["completed_updates"] == 27


def test_package_deduplicates_content_and_failure_keeps_previous_archive(tmp_path):
    import tarfile

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same immutable content")
    b.write_bytes(a.read_bytes())
    plan = dict(
        unresolved=[],
        sources=[dict(path=str(p), sha256=file_sha256(p), archive_path=p.name) for p in (a, b)],
        artifacts=[],
    )
    archive = tmp_path / "bundle.tar"
    package(plan, archive)
    previous = file_sha256(archive)
    with tarfile.open(archive) as tar:
        assert tar.getmember("b.txt").islnk()
        assert tar.extractfile("b.txt").read() == a.read_bytes()
        tar.extractall(tmp_path / "moved", filter="data")
    assert (tmp_path / "moved/b.txt").read_text() == a.read_text()
    b.write_text("replaced")
    with pytest.raises(ValueError, match="changed during packaging"):
        package(plan, archive)
    assert file_sha256(archive) == previous and not archive.with_name("bundle.tar.tmp").exists()
