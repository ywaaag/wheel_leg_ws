"""源码身份：每次run只计算一次，checkpoint只引用该记录。

位置无关摘要按“相对路径 + 内容”计算，因此运行目录或软链接视图不影响结果。
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

from .io import read_json, write_json

MODULE_SUFFIXES = (".py",)


def module_digests(root: str | Path) -> dict[str, str]:
    """按相对路径排序计算每个模块的SHA。"""
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"源码根不存在: {base}")
    digests: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix not in MODULE_SUFFIXES:
            continue
        digests[path.relative_to(base).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def modules_digest(digests: dict[str, str]) -> str:
    """位置无关整体摘要：路径与内容成对参与。"""
    joined = "\n".join(f"{name}:{sha}" for name, sha in sorted(digests.items()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def record(root: str | Path, *, package: str = "native_stand") -> dict[str, Any]:
    """记录一次源码身份；只保存摘要与版本，不复制源码树。"""
    base = Path(root)
    digests = module_digests(base)
    return {
        "root": str(base.resolve()),
        "package": package,
        "module_count": len(digests),
        "modules_sha256": modules_digest(digests),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def write(run_dir: str | Path, root: str | Path) -> dict[str, Any]:
    """写入run的source/identity.json，返回记录。"""
    payload = record(root)
    write_json(Path(run_dir) / "source" / "identity.json", payload)
    return payload


def read_reference(run_dir: str | Path) -> dict[str, Any]:
    """读取已记录的源码身份；用于checkpoint引用与复盘。"""
    return read_json(Path(run_dir) / "source" / "identity.json", {})
