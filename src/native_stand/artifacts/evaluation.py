"""完整评估结果的唯一格式与索引。

逐条件逐环境结果保留原始JSON，不压缩成通过率摘要。
运行完成、训练proxy、独立评估与accepted分别记录。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import file_sha256, read_json, write_json

INDEX_NAME = "index.json"


def build_result(
    *,
    model: Mapping[str, Any],
    protocol: Mapping[str, Any],
    conditions: list[Mapping[str, Any]],
    kind: str = "final",
    status: str = "COMPLETED",
    notes: str | None = None,
) -> dict[str, Any]:
    """组装一份完整评估结果；kind区分best、final与diagnostic。"""
    if kind not in ("best", "final", "diagnostic"):
        raise ValueError("evaluation kind必须是best/final/diagnostic")
    missing = [row.get("condition") for row in conditions if "condition" not in row]
    if missing:
        raise ValueError(f"评估条件缺少condition字段: {missing}")
    return {
        "schema_version": 1,
        "kind": kind,
        "status": status,
        "model": dict(model),
        "protocol": dict(protocol),
        "conditions": [dict(row) for row in conditions],
        "notes": notes,
        "capability_accepted": False,
    }


def write_result(path: str | Path, result: Mapping[str, Any]) -> Path:
    """写一份完整评估结果，保留全部条件与环境明细。"""
    return write_json(path, result)


def register(
    evaluations_dir: str | Path,
    *,
    kind: str,
    path: str | Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """在run的评估索引中登记一份结果；best与final分别登记。"""
    directory = Path(evaluations_dir)
    file = Path(path)
    index_path = directory / INDEX_NAME
    index = read_json(index_path, {"evaluations": []})
    entries = index.get("evaluations")
    if isinstance(entries, dict):  # 旧格式：迁移为列表，保留旧明细
        entries = [{**value, "kind": key} for key, value in entries.items()]
    evaluation_id = result.get("evaluation_id") or f"{kind}_{file.stem}"
    entry = {
        "evaluation_id": evaluation_id,
        "kind": kind,
        "path": str(file.resolve()),
        "sha256": file_sha256(file),
        "status": result.get("status"),
        "model_sha256": (result.get("model") or {}).get("sha256"),
        "model_checkpoint": (result.get("model") or {}).get("checkpoint"),
        "protocol_version": (result.get("protocol") or {}).get("version"),
        "conditions": len(result.get("conditions") or []),
        "capability_accepted": result.get("capability_accepted"),
    }
    entries = [row for row in entries if row.get("evaluation_id") != evaluation_id]
    entries.append(entry)
    latest = {row.get("kind"): row.get("evaluation_id") for row in entries}
    write_json(index_path, {"evaluations": entries, "latest": latest})
    return entry


def read_index(evaluations_dir: str | Path) -> dict[str, Any]:
    index = read_json(Path(evaluations_dir) / INDEX_NAME, {"evaluations": []})
    entries = index.get("evaluations")
    if isinstance(entries, dict):
        entries = [{**value, "kind": key} for key, value in entries.items()]
    return {"evaluations": entries, "latest": index.get("latest") or {}}


def summarize(result: Mapping[str, Any]) -> dict[str, Any]:
    """只做计数与状态汇总，不替代逐条件结果。"""
    rows = list(result.get("conditions") or [])
    envs = [env for row in rows for env in (row.get("per_env") or [])]
    failed = [env for env in envs if env.get("status") not in (None, "COMPLETED")]
    return {
        "kind": result.get("kind"),
        "status": result.get("status"),
        "conditions": len(rows),
        "envs": len(envs),
        "env_failures": len(failed),
        "model_sha256": (result.get("model") or {}).get("sha256"),
        "protocol_version": (result.get("protocol") or {}).get("version"),
    }
