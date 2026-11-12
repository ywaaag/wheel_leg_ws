"""HAPI是可选集成：只消费标准运行状态，不参与训练判定。

没有HAPI_SESSION_ID时核心功能全部可用；通知失败不改变训练与能力结果。
本次重构不发送任何消息。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from native_stand.artifacts.io import read_json


class NotificationError(RuntimeError):
    """通知投递失败；调用方只记录，不改写训练终态。"""


def session_id() -> str | None:
    """当前会话标识；不存在时返回None，不视为错误。"""
    value = os.environ.get("HAPI_SESSION_ID")
    return value.strip() or None if value else None


def available() -> bool:
    return session_id() is not None


def notification_payload(run_dir: str | Path) -> dict[str, Any]:
    """从标准运行状态构造通知内容，不重新推导训练阶段。"""
    directory = Path(run_dir)
    state = read_json(directory / "state.json", {})
    return {
        "run": str(directory.resolve()),
        "run_id": state.get("run_id"),
        "status": state.get("status"),
        "stage": state.get("stage"),
        "updates_used": state.get("updates_used"),
        "capability_accepted": bool(state.get("capability_accepted")),
        "latest_model": state.get("latest_model"),
        "failure": state.get("failure"),
    }


def notify(
    run_dir: str | Path,
    *,
    enabled: bool,
    target: str | None,
    sender: Callable[[dict[str, Any], str], None] | None = None,
) -> dict[str, Any]:
    """按配置发送一次通知。

    未启用或没有目标时直接返回disabled状态，不创建回执文件。
    发送异常被收敛为错误记录，绝不向上抛出改变训练结果。
    """
    if not enabled:
        return {"status": "disabled", "sent": False}
    if not target:
        return {"status": "disabled", "sent": False, "reason": "没有通知目标"}
    payload = notification_payload(run_dir)
    if sender is None:
        return {"status": "skipped", "sent": False, "reason": "未配置通知传输实现；不发送消息"}
    try:
        sender(payload, target)
    except Exception as exc:  # 通知失败不改变训练结果
        print(f"通知发送失败，不影响训练终态: {exc!r}", file=sys.stderr)
        return {"status": "failed", "sent": False, "error": repr(exc)}
    return {"status": "sent", "sent": True, "target": target}
