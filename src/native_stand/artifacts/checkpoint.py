"""完整恢复checkpoint的读写、字段校验与Actor-only导出。

torch在函数内延迟导入，模块顶层只做纯数据处理。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import file_sha256

SCHEMA_VERSION = 2
FULL_STATE_KEYS = ("actor", "critic", "optimizer", "std", "rng", "counters")


class CheckpointError(ValueError):
    """checkpoint缺少完整恢复状态或字段语义不符。"""


def save_full(path: str | Path, state: Mapping[str, Any], *, exclusive: bool = False) -> Path:
    """写入完整恢复状态；先校验字段，再原子替换。"""
    missing = [key for key in FULL_STATE_KEYS if key not in state]
    if missing:
        raise CheckpointError(f"完整checkpoint缺少字段: {missing}")
    payload = {"schema_version": SCHEMA_VERSION, **dict(state)}
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    import torch

    try:
        torch.save(payload, temporary)
        if exclusive:
            # Publish once without replacing a file already consumed by a
            # candidate/resume reference. Same-update retries cannot mutate it.
            os.link(temporary, file)
        else:
            os.replace(temporary, file)
    finally:
        temporary.unlink(missing_ok=True)
    return file


class LoadedSource:
    """一次加载的完整恢复来源：payload与其真实文件身份一起传递。"""

    def __init__(self, payload: dict[str, Any], path: Path, sha256: str):
        self.payload = payload
        self.path = path
        self.sha256 = sha256

    def __repr__(self) -> str:
        return f"LoadedSource(path={self.path}, sha256={self.sha256[:8]}...)"


def _load_identified(path, *, map_location="cpu"):
    """Deserialize and hash the same opened inode; reject in-place mutation."""
    import hashlib

    import torch

    file = Path(path).resolve()
    with file.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        payload = torch.load(stream, map_location=map_location, weights_only=False)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise CheckpointError("来源文件在加载期间发生修改")
    return LoadedSource(payload, file, digest)


def load_complete_state(path: str | Path) -> LoadedSource:
    """生产V4完整格式的读取边界：不要求本项目的schema_version。

    完整性的判定是结构性的：必须同时含Actor与Critic，且是torch可读的dict。
    不补字段、不重写文件、不接受Actor-only。
    """
    file = Path(path).resolve()
    if not file.is_file():
        raise FileNotFoundError(f"恢复来源不存在: {file}")
    loaded = _load_identified(file)
    payload = loaded.payload
    if not isinstance(payload, dict):
        raise CheckpointError(f"不是完整恢复状态: {file}")
    schema = payload.get("schema_version")
    if schema == SCHEMA_VERSION:
        required = ("actor", "critic", "optimizer", "std", "rng", "counters")
    else:
        # 已声明支持的V4完整格式：优化器、训练RNG与计数缺一不可。
        required = (
            "actor",
            "critic",
            "optimizer",
            "training_rng",
            "online_updates_including_smoke",
            "network_architecture",
        )
    missing = [name for name in required if name not in payload]
    if missing:
        raise CheckpointError(f"不是可完整恢复的状态（缺{missing}）: {file}")
    return loaded


def load_source(path: str | Path) -> LoadedSource:
    """在加载边界构造来源对象；身份来自实际文件，不靠字段猜测。"""
    file = Path(path).resolve()
    if not file.is_file():
        raise FileNotFoundError(f"恢复来源不存在: {file}")
    loaded = _load_identified(file)
    _validate_full(loaded.payload)
    return loaded


def load_actor_source(path: str | Path, *, map_location: str = "cpu") -> LoadedSource:
    """Actor-only来源：只需Actor权重（可带output_contract标识Recovery格式）。

    这不是完整恢复状态；不能拿去走restore_full。
    """
    file = Path(path).resolve()
    if not file.is_file():
        raise FileNotFoundError(f"Actor来源不存在: {file}")
    loaded = _load_identified(file, map_location=map_location)
    payload = loaded.payload
    if not isinstance(payload, dict) or "actor" not in payload:
        raise CheckpointError(f"不是Actor来源（缺actor权重）: {file}")
    return loaded


def load_full(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """读取完整恢复状态并校验schema。"""
    import torch

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    _validate_full(payload)
    return payload


def _validate_full(payload):
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError("不支持的checkpoint schema")
    missing = [key for key in FULL_STATE_KEYS if key not in payload]
    if missing:
        raise CheckpointError(f"完整checkpoint缺少字段: {missing}")


def export_actor(path: str | Path, checkpoint: str | Path, *, metadata: Mapping[str, Any]) -> Path:
    """导出Actor-only交付件；附带来源身份，不能替代完整恢复状态。"""
    import torch

    source = _load_identified(checkpoint)
    payload = source.payload
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": payload["actor"],
            "interface_version": metadata.get("interface_version"),
            "interface_sha256": metadata.get("interface_sha256"),
            "source_checkpoint_sha256": source.sha256,
            "actor_only": True,
            "exports_full_recovery_state": False,
        },
        file,
    )
    return file


def reference(path: str | Path, *, update: int | None = None, stage: str | None = None) -> dict:
    """生成模型引用记录：路径、SHA与计数，不含权重。"""
    file = Path(path)
    return {
        "checkpoint": str(file.resolve()),
        "sha256": file_sha256(file),
        "bytes": file.stat().st_size,
        "update": update,
        "stage": stage,
    }
