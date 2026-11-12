"""新run的默认保存策略与普通轮换。

保留集合 = 最近两份 ∪ 显式里程碑 ∪ best ∪ selected ∪ 恢复来源 ∪ 被引用模型。
只处理inventory登记、且确认属于本run的中间checkpoint；不扫描、不删除旧run或外部路径。
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import file_sha256, read_json, write_json

CHECKPOINT_PATTERN = re.compile(r"^model_update_(\d+)\.pt$")


@dataclass(frozen=True)
class SavePolicy:
    """周期与保留上限；里程碑另行显式配置。"""

    every_updates: int = 100
    keep_recent: int = 2
    milestones: tuple[int, ...] = ()
    save_on_stop: bool = True

    def __post_init__(self) -> None:
        if self.every_updates < 1:
            raise ValueError("checkpoint_every_updates必须至少为1")
        if self.keep_recent < 1:
            raise ValueError("keep_recent_checkpoints必须至少为1")

    def due(self, update: int, *, final: bool = False) -> bool:
        if final:
            return True
        return update % self.every_updates == 0 or update in self.milestones


def _update_of(path: Path) -> int:
    match = CHECKPOINT_PATTERN.match(path.name)
    return int(match.group(1)) if match else -1


def _owned_by_run(path: Path, run_dir: Path) -> bool:
    """文件必须真实位于本run的checkpoints目录（不解析软链接后跑到外部）。"""
    if path.is_symlink():
        return False
    try:
        return path.resolve().parent == (run_dir / "checkpoints").resolve()
    except OSError:
        return False


def inventory_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoints" / "inventory.json"


def read_inventory(run_dir: str | Path) -> list[dict[str, Any]]:
    record = read_json(inventory_path(run_dir), {}) or {}
    return list(record.get("checkpoints") or [])


def write_inventory(run_dir: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    rows = [dict(row) for row in rows]
    return write_json(inventory_path(run_dir), {"checkpoints": rows})


def _owned_candidates(directory: Path, checkpoint_dir: Path, referenced: set[str]) -> list[Path]:
    """本run真正登记过的checkpoint；未登记文件、外部路径与软链接一律跳过。"""
    rows = read_inventory(directory)
    known: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get("checkpoint")
        if not value:
            continue
        path = Path(value)
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if not _inside(path, checkpoint_dir):
            continue
        expected = row.get("sha256")
        # 身份必须可核验：缺失或非法SHA一律不作候选，不能绕过检查。
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            continue
        if expected != file_sha256(path):
            continue  # 同名文件被替换：跳过并保留原记录，不删新内容
        seen.add(resolved)
        known.append(path)
    # 状态引用（protected/registered）也必须有inventory记录才纳入；这里只返回登记项。
    return known


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _skipped_records(
    directory: Path, checkpoint_dir: Path, candidates: list[Path]
) -> list[dict[str, str]]:
    """登记但本次不能作为候选的记录及原因；这些记录必须留在inventory里。"""
    considered = {str(path.resolve()) for path in candidates}
    rows: list[dict[str, str]] = []
    for row in read_inventory(directory):
        value = row.get("checkpoint")
        if not value:
            continue
        path = Path(value)
        resolved = str(path.resolve())
        if resolved in considered:
            continue
        if path.is_symlink():
            reason = "symlink"
        elif not path.is_file():
            reason = "missing"
        elif not _inside(path, checkpoint_dir):
            reason = "outside_run"
        else:
            reason = "identity_changed"
        rows.append({"checkpoint": resolved, "reason": reason})
    return rows


def plan_prune(
    run_dir: str | Path,
    policy: SavePolicy,
    *,
    protected: Iterable[str | Path] = (),
    registered: Iterable[str | Path] = (),
    keep_latest: int | None = None,
) -> dict[str, Any]:
    """计算删除计划，不修改磁盘。

    protected：best/selected/恢复来源/显式固定模型。
    registered：模型引用索引中仍有真实消费者的路径。
    keep_latest：最近窗口大小，默认取policy.keep_recent。
    """
    directory = Path(run_dir)
    checkpoint_dir = directory / "checkpoints"
    protected_set = {str(Path(item).resolve()) for item in protected if item}
    registered_set = {str(Path(item).resolve()) for item in registered if item}
    window = policy.keep_recent if keep_latest is None else max(1, int(keep_latest))

    # 候选集合只来自本run的inventory与状态引用，不按文件名猜测。
    candidates = sorted(
        (path for path in _owned_candidates(directory, checkpoint_dir, protected_set)),
        key=_update_of,
        reverse=True,
    )
    keep: set[Path] = set()
    reasons: dict[str, list[str]] = {}

    def add(path: Path, reason: str) -> None:
        keep.add(path)
        reasons.setdefault(str(path.resolve()), []).append(reason)

    newest = candidates[:window]
    for path in newest:
        add(path, "recent")
    for path in candidates:
        resolved = str(path.resolve())
        if _update_of(path) in policy.milestones:
            add(path, "milestone")
        if resolved in protected_set:
            add(path, "protected")
        if resolved in registered_set:
            add(path, "referenced")

    kept = sorted(keep, key=_update_of)
    removed = [path for path in candidates if path not in keep]
    skipped = _skipped_records(directory, checkpoint_dir, candidates)
    return {
        "policy": {
            "every_updates": policy.every_updates,
            "keep_recent": policy.keep_recent,
            "window_used": window,
            "milestones": list(policy.milestones),
        },
        "kept": [str(path.resolve()) for path in kept],
        "removed": [str(path.resolve()) for path in removed],
        "keep_reasons": {key: sorted(set(value)) for key, value in sorted(reasons.items())},
        "protected": sorted(protected_set),
        "registered": sorted(registered_set),
        "candidates": len(candidates),
        "skipped": skipped,
        "dry_run": True,
    }


def prune(
    run_dir: str | Path,
    policy: SavePolicy,
    *,
    protected: Iterable[str | Path] = (),
    registered: Iterable[str | Path] = (),
    dry_run: bool = False,
    index=None,
) -> dict[str, Any]:
    """执行删除计划，并把真实保留集合写回inventory与retention.json。"""
    if index is None:
        from .model_refs import ModelIndex

        record = read_json(Path(run_dir) / "run.json", {}) or {}
        index = ModelIndex(record.get("model_index", Path(run_dir).parent / "models.json"))
    registered = set(registered) | index.consumer_paths()
    plan = plan_prune(run_dir, policy, protected=protected, registered=registered)
    directory = Path(run_dir)
    previous = read_inventory(directory)
    report = {**plan, "dry_run": dry_run}
    if not dry_run:
        removed: list[str] = []
        failures: list[dict[str, str]] = []
        logical_bytes = released_bytes = 0
        planned_sha = {
            str(Path(row["checkpoint"]).resolve()): row.get("sha256")
            for row in previous
            if row.get("checkpoint")
        }
        for item in plan["removed"]:
            path = Path(item)
            try:
                if not path.exists():
                    raise FileNotFoundError(item)
                if path.is_symlink() or not _inside(path, directory / "checkpoints"):
                    failures.append({"path": item, "error": "not_owned"})
                    continue
                expected = planned_sha.get(str(path.resolve()))
                if not expected or expected != file_sha256(path):
                    failures.append({"path": item, "error": "identity_changed"})
                    continue
                result = index.delete_owned(path, expected, run_dir=directory)
                if not result["deleted"]:
                    failures.append({"path": item, "error": result["reason"]})
                    continue
                logical_bytes += result["logical_bytes"]
                released_bytes += result["released_bytes"]
                removed.append(item)
            except FileNotFoundError:
                # 文件已不在：不算删除成功，保留其记录，如实记一条跳过原因。
                failures.append({"path": item, "error": "missing"})
            except OSError as exc:
                failures.append({"path": item, "error": repr(exc)})
        removed_set = {str(Path(item).resolve()) for item in removed}
        survivors = [
            row for row in previous if str(Path(row["checkpoint"]).resolve()) not in removed_set
        ]
        report["removed"] = removed
        report["logical_bytes_removed"] = logical_bytes
        report["released_bytes"] = released_bytes
        if failures:
            report["failures"] = failures
        write_inventory(directory, survivors)
        write_json(directory / "checkpoints" / "retention.json", report)
    return report


def latest_kept(
    run_dir: str | Path, *, registered: Iterable[str | Path] = ()
) -> dict[str, Any] | None:
    """本run当前真实存在的最新完整checkpoint（用于默认恢复点）。"""
    directory = Path(run_dir)
    rows = [row for row in read_inventory(directory) if Path(row["checkpoint"]).is_file()]
    if not rows:
        return None
    return max(rows, key=lambda row: int(row.get("update") or -1))


def disk_usage(checkpoint_dir: str | Path) -> int:
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        return 0
    total = 0
    for entry in os.scandir(directory):
        if entry.is_file() and not entry.is_symlink():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total
