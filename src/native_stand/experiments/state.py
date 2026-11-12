"""权威运行状态：state.json 的唯一写入协议。

写入方约定：
- launcher（`wheel-leg` 进程）只写 SUBMITTED，然后交棒；
- worker 用 `claim()` 接管owner身份，此后只有该 PID 能写运行状态与终态；
- 终态只写一次：重复提交相同内容是无操作，内容冲突则拒绝。

预算计数从这里读取，不从稀疏曲线反推。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from native_stand.artifacts.io import read_json, write_json

SCHEMA_VERSION = 2
TERMINAL = ("COMPLETED", "FAILED", "STOPPED_BOUNDED")

STATUS_SUBMITTED = "SUBMITTED"
STATUS_RUNNING = "RUNNING"
STATUS_EVALUATING = "EVALUATING"

# 终态一旦写入，这些字段不可再被改写。
FROZEN_ON_TERMINAL = ("status", "failure", "updates_used", "samples_used", "capability_accepted")


class StateError(ValueError):
    """状态迁移不合法。"""


class RunStateStore:
    """state.json的读写入口：单写入者、终态幂等。"""

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.path = self.dir / "state.json"

    def read(self) -> dict[str, Any]:
        return read_json(self.path, {})

    # ---- 生命周期 -------------------------------------------------
    def create(
        self,
        *,
        run_id: str,
        owner: dict[str, Any],
        stage: str,
        updates_used: int = 0,
        samples_used: int = 0,
    ) -> dict[str, Any]:
        if self.path.exists():
            raise StateError(f"state.json已存在: {self.path}")
        state = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": STATUS_SUBMITTED,
            "stage": stage,
            "updates_used": int(updates_used),
            "samples_used": int(samples_used),
            "owner": dict(owner),
            "latest_model": None,
            "best_model": None,
            "selected_model": None,
            "continuation_model": None,
            "capability_accepted": False,
            "failure": None,
            "revision": 0,
            "created_unix_s": time.time(),
            "updated_unix_s": time.time(),
        }
        write_json(self.path, state)
        return state

    def claim(
        self, *, pid: int | None = None, attempt_id: str | None = None, stage: str | None = None
    ) -> dict[str, Any]:
        """worker接管写入权；只在非终态时允许。"""
        state = self._require()
        if state.get("status") in TERMINAL:
            raise StateError(f"终态{state['status']}不能被worker接管")
        fields: dict[str, Any] = {
            "status": STATUS_RUNNING,
            "owner": {"pid": int(pid or os.getpid()), "role": "worker"},
            "worker_pid": int(pid or os.getpid()),
        }
        if attempt_id:
            fields["attempt_id"] = attempt_id
        if stage:
            fields["stage"] = stage
        # 交棒只允许从SUBMITTED发生：launcher写完初始状态后不再写运行状态。
        takeover = state.get("status") == STATUS_SUBMITTED
        return self._write(
            state, fields, writer_pid=int(pid or os.getpid()), allow_foreign=takeover
        )

    # ---- 写入 -----------------------------------------------------
    def update(
        self, writer_pid: int | None = None, allow_foreign: bool = False, **fields: Any
    ) -> dict[str, Any]:
        """按owner身份合并字段；非owner写入被拒绝。"""
        state = self._require()
        return self._write(state, fields, writer_pid=writer_pid, allow_foreign=allow_foreign)

    def reopen_for_resume(
        self, *, pid: int, attempt_id: str, stage: str, allow_failed: bool = False
    ) -> dict[str, Any]:
        """有界续训迁移：仅限COMPLETED/STOPPED_BOUNDED，且保留累计消耗。

        FAILED不允许直接续训；updates_used等冻结字段保持原值不变。
        """
        state = self._require()
        allowed = (
            ("COMPLETED", "STOPPED_BOUNDED", "FAILED")
            if allow_failed
            else ("COMPLETED", "STOPPED_BOUNDED")
        )
        if state.get("status") not in allowed:
            raise StateError(f"状态{state.get('status')}不能续训")
        updated = {
            **state,
            "status": "RUNNING",
            "attempt_id": attempt_id,
            "stage": stage,
            "owner": {"pid": pid, "module": Path(__file__).name},
            "worker_pid": pid,
            "failure": None,
            "previous_failures": state.get("previous_failures", [])
            + (
                [dict(attempt_id=state.get("attempt_id"), failure=state.get("failure"))]
                if state.get("failure")
                else []
            ),
        }
        self._write(updated, {}, writer_pid=os.getpid(), allow_foreign=True)
        return updated

    def terminal(
        self,
        status: str,
        *,
        failure: str | None = None,
        writer_pid: int | None = None,
        allow_foreign: bool = False,
        **fields: Any,
    ) -> dict[str, Any]:
        """写终态；相同内容重复提交无操作，冲突内容拒绝。"""
        if status not in TERMINAL:
            raise StateError(f"非法终态: {status}")
        state = self._require()
        incoming = {"status": status, "failure": failure, **fields}
        if state.get("status") in TERMINAL:
            conflicts = {
                key: (state.get(key), value)
                for key, value in incoming.items()
                if key in FROZEN_ON_TERMINAL and state.get(key) != value
            }
            if conflicts:
                raise StateError(f"终态{state['status']}已冻结，拒绝改写: {sorted(conflicts)}")
            return state
        return self._write(state, incoming, writer_pid=writer_pid, allow_foreign=allow_foreign)

    # ---- 内部 -----------------------------------------------------
    def _require(self) -> dict[str, Any]:
        state = self.read()
        if not state:
            raise StateError(f"state.json不存在: {self.path}")
        return state

    def _check_writer(self, state: dict[str, Any], writer: int, *, allow_foreign: bool) -> None:
        owner_pid = (state.get("owner") or {}).get("pid")
        # 没有owner（或仍是占位的0）表示尚无worker接管，谁写都只有一份记录。
        if allow_foreign or not owner_pid:
            return
        if int(owner_pid) != writer:
            raise StateError(f"state.json只允许owner写入: owner={owner_pid} writer={writer}")

    def _write(
        self,
        state: dict[str, Any],
        fields: dict[str, Any],
        *,
        writer_pid: int | None,
        allow_foreign: bool = False,
    ) -> dict[str, Any]:
        self._check_writer(state, int(writer_pid or os.getpid()), allow_foreign=allow_foreign)
        current = state.get("status")
        incoming = fields.get("status")
        if current in TERMINAL and incoming is not None and incoming != current:
            raise StateError(f"终态{current}不能被{incoming}覆盖")
        state.update({key: value for key, value in fields.items() if not key.startswith("_")})
        state["revision"] = int(state.get("revision", 0)) + 1
        state["updated_unix_s"] = time.time()
        write_json(self.path, state)
        return state


def write_stop_request(
    run_dir: str | Path,
    *,
    reason: str,
    requester: str | None = None,
    after_round: int | None = None,
) -> Path:
    """记录停止请求；请求本身不修改训练终态。

    after_round：可选，累计轮次到达后的轮边界停止（BC使用）。
    """
    payload = {
        "requested_unix_s": time.time(),
        "reason": reason,
        "requester": requester,
        "applies_to_run": str(Path(run_dir).resolve()),
    }
    if after_round is not None:
        payload["after_round"] = int(after_round)
    return write_json(Path(run_dir) / "stop_request.json", payload)
