"""唯一的基础文件IO实现：SHA、规范化JSON摘要、原子JSON写入与路径解析。

本模块不导入torch，也不读取任何任务配置。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

JSON_SEPARATORS = (",", ":")


def file_sha256(path: str | Path) -> str:
    """普通文件SHA256；不存在时抛出FileNotFoundError。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    """规范化JSON内容摘要：键排序、无空格、拒绝NaN。"""
    encoded = json.dumps(value, sort_keys=True, separators=JSON_SEPARATORS, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: str | Path, default: Any = None) -> Any:
    """读取JSON；文件不存在时返回default。"""
    file = Path(path)
    if not file.is_file():
        return default
    return json.loads(file.read_text())


def write_json(path: str | Path, value: Any) -> Path:
    """原子写入JSON：同目录临时文件、fsync、replace。"""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, file)
    return file


def write_json_new(path: str | Path, value: Any) -> Path:
    """原子地创建新JSON；已存在则失败，用于一次性回执。"""
    file = Path(path)
    if file.exists():
        raise FileExistsError(str(file))
    return write_json(file, value)


def append_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    """追加紧凑JSONL行；每行一个记录，不做重排。"""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    return file


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取JSONL；跳过空行，损坏行直接报错。"""
    file = Path(path)
    if not file.is_file():
        return []
    return [json.loads(line) for line in file.read_text().splitlines() if line.strip()]


def write_npz(path: str | Path, arrays: Mapping[str, Any]) -> Path:
    """原子写入NPZ。"""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, file)
    return file


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    """读取NPZ为普通字典。"""
    with np.load(Path(path)) as data:
        return {key: data[key] for key in data.files}


def new_dir(path: str | Path) -> Path:
    """创建全新目录；已存在则失败，避免追加到旧运行。"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def tree_bytes(path: str | Path) -> int:
    """目录内普通文件总字节数；软链接不重复计入。"""
    root = Path(path)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def resolve_path(value: str | Path, base: str | Path) -> Path:
    """按显式基准解析路径，不依赖调用者的当前目录。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(base) / path).resolve()
