"""单一预算账本：按attempt ID记账，保留全部历史消耗。

新run的预算文件只保存limits、used与attempt状态，不嵌入配置副本。
"""

from __future__ import annotations

import fcntl
import time
from pathlib import Path
from typing import Any

from native_stand.artifacts.io import read_json, write_json

SCHEMA_VERSION = 1
TERMINAL_STATUSES = ("COMPLETED", "FAILED", "STOPPED_BOUNDED", "TERMINATED")


class BudgetError(ValueError):
    """预算或attempt身份不合法。"""


class Budget:
    """单文件预算账本，所有变更在独占锁内完成。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _lock_path(self) -> Path:
        return self.path.with_suffix(".lock")

    def _change(self, operation):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path().open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            book = read_json(self.path)
            if book is None:
                raise BudgetError(f"预算文件不存在: {self.path}")
            result = operation(book)
            write_json(self.path, book)
            return result

    @staticmethod
    def create(
        path: str | Path,
        *,
        smoke_updates: int,
        formal_updates: int,
        sample_limit: int | None = None,
        evaluation_wall_s: float | None = None,
        distill_rounds: int | None = None,
    ) -> Budget:
        """创建新预算文件；已存在则失败，避免重复使用旧额度。"""
        target = Path(path)
        if target.exists():
            raise BudgetError(f"预算文件已存在: {target}")
        write_json(
            target,
            {
                "schema_version": SCHEMA_VERSION,
                "authorized": True,
                "limits": {
                    "smoke": int(smoke_updates),
                    "formal": int(formal_updates),
                    **({"distill": int(distill_rounds)} if distill_rounds is not None else {}),
                },
                "used": {
                    "smoke": 0,
                    "formal": 0,
                    **({"distill": 0} if distill_rounds is not None else {}),
                },
                "sample_limit": sample_limit,
                "evaluation_wall_s": evaluation_wall_s,
                "attempts": {},
                "created_unix_s": time.time(),
            },
        )
        return Budget(target)

    def reserve(
        self, attempt_id: str, stage: str, output_dir: str | Path, *, finalize_only: bool = False
    ) -> dict[str, Any]:
        """预留一次attempt；同一时刻只允许一个RUNNING。"""
        output = str(Path(output_dir).resolve())

        def change(book):
            if not book.get("authorized"):
                raise BudgetError("预算未授权")
            if stage not in book["limits"]:
                raise BudgetError(f"未知阶段: {stage}")
            if attempt_id in book["attempts"]:
                raise BudgetError("重复attempt ID")
            if any(row["output_dir"] == output for row in book["attempts"].values()):
                raise BudgetError("输出目录已被预留")
            if any(row["status"] == "RUNNING" for row in book["attempts"].values()):
                raise BudgetError("已有RUNNING attempt")
            if finalize_only and stage != "distill":
                raise BudgetError("Only supervision can finalize already charged phases")
            if (
                stage != "evaluate"
                and not finalize_only
                and book["used"][stage] >= book["limits"][stage]
            ):
                raise BudgetError("阶段预算已耗尽")
            row = {
                "stage": stage,
                "output_dir": output,
                "status": "RUNNING",
                "updates": 0,
                "started_unix_s": time.time(),
            }
            book["attempts"][attempt_id] = row
            return row

        return self._change(change)

    def charge_rounds(self, attempt_id: str, output_dir: str | Path, *, rounds: int) -> int:
        """BC轮次按增量计费：本轮attempt只记它自己完成的部分。"""
        output = str(Path(output_dir).resolve())
        if rounds < 1:
            raise BudgetError("charge_rounds rounds必须至少为1")

        def change(book):
            row = book["attempts"].get(attempt_id)
            if row is None or row["output_dir"] != output or row["status"] != "RUNNING":
                raise BudgetError("attempt身份或状态不符")
            if row["stage"] != "distill":
                raise BudgetError("只有distill阶段使用轮次计费")
            if book["used"]["distill"] + rounds > book["limits"]["distill"]:
                raise BudgetError("蒸馏轮次预算已耗尽")
            row["rounds"] = row.get("rounds", 0) + rounds
            book["used"]["distill"] += rounds
            return row["rounds"]

        return self._change(change)

    def charge(self, attempt_id: str, output_dir: str | Path, *, updates: int = 1) -> int:
        """按实际完成的update计费；不因曲线稀疏而少记。"""
        output = str(Path(output_dir).resolve())
        if updates < 1:
            raise BudgetError("charge updates必须至少为1")

        def change(book):
            row = book["attempts"].get(attempt_id)
            if row is None or row["output_dir"] != output or row["status"] != "RUNNING":
                raise BudgetError("attempt身份或状态不符")
            stage = row["stage"]
            if stage not in ("smoke", "formal"):
                raise BudgetError(f"{stage}阶段没有PPO update（BC轮次请使用charge_rounds）")
            if book["used"][stage] + updates > book["limits"][stage]:
                raise BudgetError("阶段预算已耗尽")
            row["updates"] += updates
            book["used"][stage] += updates
            return row["updates"]

        return self._change(change)

    def charge_resources(self, attempt_id, output_dir, **amounts):
        """Independent actual counters, inside the existing budget lock."""
        if set(amounts) - {"supervised_updates", "physical_steps"} or any(
            type(v) is not int or v < 0 for v in amounts.values()
        ):
            raise BudgetError("Invalid resource consumption")

        def change(book):
            row = book["attempts"].get(attempt_id)
            if (
                row is None
                or row["status"] != "RUNNING"
                or row["output_dir"] != str(Path(output_dir).resolve())
            ):
                raise BudgetError("attempt identity differs")
            for key, value in amounts.items():
                row[key] = row.get(key, 0) + value
                book.setdefault("resources_used", {})[key] = (
                    book.get("resources_used", {}).get(key, 0) + value
                )

        return self._change(change)

    def finalize(
        self,
        attempt_id: str,
        output_dir: str | Path,
        status: str,
        *,
        exit_code: int | None = None,
        reason: str | None = None,
        wall_s: float | None = None,
    ) -> dict[str, Any]:
        """结算attempt；不得改写已结算记录。"""
        if status not in TERMINAL_STATUSES:
            raise BudgetError(f"非法终态: {status}")
        output = str(Path(output_dir).resolve())

        def change(book):
            row = book["attempts"].get(attempt_id)
            if row is None or row["output_dir"] != output:
                raise BudgetError("attempt身份不符")
            if row["status"] != "RUNNING":
                raise BudgetError("attempt已结算")
            row.update(
                status=status,
                exit_code=exit_code,
                reason=reason,
                wall_s=wall_s,
                finished_unix_s=time.time(),
            )
            return row

        return self._change(change)

    def used(self) -> dict[str, int]:
        return dict(read_json(self.path)["used"])

    def limits(self) -> dict[str, int]:
        return dict(read_json(self.path)["limits"])

    def rows(self) -> dict[str, dict[str, Any]]:
        return dict(read_json(self.path)["attempts"])
