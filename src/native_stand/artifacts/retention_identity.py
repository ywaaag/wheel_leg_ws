"""固定来源损失的内容身份；存储位置只在 RuntimePaths 边界解析。"""

from __future__ import annotations

import copy

from .io import file_sha256, json_digest
from .runtime_paths import RuntimePaths


def semantic_identity(configuration):
    """只剥离已声明、带内容 SHA 的产物位置，不剥离任意语义字段。"""
    value = copy.deepcopy(configuration)
    if not isinstance(value, dict):
        raise ValueError("缺少固定来源损失声明")
    for key in ("bank", "bank_metadata"):
        if key not in value:
            continue
        reference = value[key]
        if set(reference) != {"path", "sha256"} or len(reference["sha256"]) != 64:
            raise ValueError("固定来源产物必须有明确位置和内容SHA: " + key)
        value[key] = {"sha256": reference["sha256"]}
    return json_digest(value)


def resolve_retention(configuration, *, assets):
    """加载边界返回映射后的声明，同时核对实际消费的 bank/metadata。"""
    value = RuntimePaths.from_assets(assets).view(configuration)
    semantic_identity(value)
    for key in ("bank", "bank_metadata"):
        if key in value and file_sha256(value[key]["path"]) != value[key]["sha256"]:
            raise ValueError("固定来源产物身份不同: " + key)
    return value


def require_same_retention(source, current):
    if semantic_identity(source) != semantic_identity(current):
        raise ValueError("固定来源Teacher/损失语义声明不一致")
