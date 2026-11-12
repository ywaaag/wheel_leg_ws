"""产物层：IO、checkpoint、保存策略、模型引用、评估索引与报告。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from native_stand.artifacts import io
from native_stand.artifacts.checkpoint import CheckpointError, load_full, save_full
from native_stand.artifacts.evaluation import build_result, register, summarize, write_result
from native_stand.artifacts.metrics import MetricsWriter
from native_stand.artifacts.model_refs import ModelIndex, register_model
from native_stand.artifacts.report import build_report
from native_stand.artifacts.retention import SavePolicy, prune

FROZEN_TEST_SCOPE = "PORTABLE"  # 自包含CPU工程契约：tmp fixture + 真实代码，无本机runs/资产依赖

torch = pytest.importorskip("torch")


def _state() -> dict:
    return {
        "actor": {"w": torch.zeros((2, 2))},
        "critic": {"w": torch.ones((2, 1))},
        "optimizer": {"step": torch.zeros((1,))},
        "std": torch.full((6,), 0.2),
        "rng": {"torch": 12345},
        "counters": {"updates": 7},
    }


def test_sha_and_digest_are_distinct_and_stable(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello")
    assert io.file_sha256(path) == io.file_sha256(path)
    assert io.json_digest({"b": 1, "a": 2}) == io.json_digest({"a": 2, "b": 1})
    assert io.json_digest({"a": 1}) != io.json_digest({"a": 2})


def test_json_digest_rejects_nan():
    with pytest.raises(ValueError):
        io.json_digest({"a": float("nan")})


def test_atomic_write_and_read(tmp_path):
    path = tmp_path / "nested" / "value.json"
    io.write_json(path, {"a": 1})
    assert io.read_json(path) == {"a": 1}
    assert io.read_json(tmp_path / "missing.json", {"d": 2}) == {"d": 2}
    with pytest.raises(FileExistsError):
        io.write_json_new(path, {"a": 2})


def test_npz_roundtrip(tmp_path):
    path = tmp_path / "trace.npz"
    io.write_npz(path, {"x": np.arange(4)})
    assert np.array_equal(io.read_npz(path)["x"], np.arange(4))


def test_checkpoint_requires_full_state(tmp_path):
    with pytest.raises(CheckpointError, match="critic"):
        save_full(tmp_path / "x.pt", {"actor": {}})
    path = save_full(tmp_path / "ok.pt", _state())
    payload = load_full(path)
    assert payload["counters"]["updates"] == 7
    assert payload["schema_version"] >= 1


def test_checkpoint_records_source_identity_reference(tmp_path):
    state = _state()
    state["source_identity"] = {"modules_sha256": "abc"}
    path = save_full(tmp_path / "ok.pt", state)
    assert load_full(path)["source_identity"]["modules_sha256"] == "abc"


def _touch(directory: Path, updates: list[int]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for update in updates:
        path = directory / f"model_update_{update}.pt"
        path.write_bytes(b"x")
        paths.append(path)
    return paths


def _register(run, paths):
    from native_stand.artifacts.io import file_sha256
    from native_stand.artifacts.retention import write_inventory

    write_inventory(run, [{"checkpoint": str(path), "update": int(path.stem.split("_")[-1]),
                           "sha256": file_sha256(path), "bytes": path.stat().st_size}
                          for path in paths])


def test_retention_keeps_recent_and_protected(tmp_path):
    """保留集合 = 最近两份 ∪ 里程碑 ∪ protected；较早受保护模型不被删除。"""
    run = tmp_path / "run"
    paths = _touch(run / "checkpoints", [10, 20, 30, 40])
    _register(run, paths)
    policy = SavePolicy(every_updates=10, keep_recent=2, milestones=(10,))
    report = prune(run, policy, protected=[paths[1]])  # 20 在最近窗口之外
    kept = {Path(name).name for name in report["kept"]}
    assert "model_update_40.pt" in kept
    assert "model_update_30.pt" in kept
    assert "model_update_20.pt" in kept, "较早的受保护模型必须保留"
    assert "model_update_10.pt" in kept
    assert paths[1].is_file()
    assert (run / "checkpoints" / "retention.json").is_file()


def test_retention_dry_run_removes_nothing(tmp_path):
    run = tmp_path / "run"
    _register(run, _touch(run / "checkpoints", [10, 20, 30]))
    report = prune(run, SavePolicy(every_updates=10, keep_recent=1), dry_run=True)
    assert len(report["removed"]) == 2
    assert len(list((run / "checkpoints").glob("model_update_*.pt"))) == 3


def test_model_index_registers_and_tracks_ancestors(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    index_path = tmp_path / "model_index.json"
    entry = register_model(index_path, weights, roles=("initialization",), run="r1")
    assert entry["roles"] == ["initialization"]
    index = ModelIndex(index_path)
    assert str(weights.resolve()) in index.referenced_paths()
    with pytest.raises(ValueError, match="角色"):
        register_model(index_path, weights, roles=("unknown_role",))


def test_model_index_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        register_model(tmp_path / "index.json", tmp_path / "nope.pt", roles=("best",))


def test_evaluation_preserves_per_condition_per_env(tmp_path):
    result = build_result(
        model={"sha256": "a" * 64},
        protocol={"version": "p1"},
        conditions=[
            {"condition": "stand", "per_env": [
                {"env": 0, "status": "COMPLETED", "success": True},
                {"env": 1, "status": "FAILED_EARLY"},
            ]},
        ],
        kind="final",
    )
    path = write_result(tmp_path / "final.json", result)
    entry = register(tmp_path / "evaluations", kind="final", path=path, result=result)
    assert entry["conditions"] == 1
    assert result["capability_accepted"] is False
    summary = summarize(result)
    assert summary["envs"] == 2
    assert summary["env_failures"] == 1


def test_report_never_invents_acceptance(tmp_path):
    io.write_json(tmp_path / "run.json", {
        "run_id": "r", "config": {"name": "demo", "task": "motion", "algorithm": "ppo",
                                  "backend": "cpu"},
        "limits": {"smoke_updates": 0, "formal_updates": 10},
    })
    io.write_json(tmp_path / "state.json", {
        "status": "COMPLETED", "stage": "formal", "updates_used": 10, "samples_used": 100,
        "capability_accepted": False,
    })
    text = build_report(tmp_path)
    assert "任务/算法/后端: motion / ppo / cpu" in text
    assert "没有独立评估记录" in text
    assert "能力accepted: False" in text


def test_metrics_writer_is_sparse_and_tagged(tmp_path):
    writer = MetricsWriter(tmp_path / "metrics.jsonl", every_updates=10, tensorboard=False,
                           stage="formal")
    assert writer.write({"update": 5, "reward_mean": 1.0}) is False
    assert writer.write({"update": 10, "reward_mean": 1.0}) is True
    rows = io.read_jsonl(tmp_path / "metrics.jsonl")
    assert rows == [{"update": 10, "reward_mean": 1.0, "stage": "formal"}]
    writer.close()
