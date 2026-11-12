"""按完整episode划分训练、验证与留出集。

同一episode的任何状态不得同时出现在两个集合；验证与留出不回用于补教。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


class SplitError(ValueError):
    """划分不合法或存在泄漏。"""


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    holdout: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = {
            "train": set(self.train),
            "validation": set(self.validation),
            "holdout": set(self.holdout),
        }
        if not groups["train"]:
            raise SplitError("训练集不能为空")
        overlap = (groups["train"] & groups["validation"]) | (groups["train"] & groups["holdout"])
        overlap |= groups["validation"] & groups["holdout"]
        if overlap:
            raise SplitError(f"episode跨集合泄漏: {sorted(overlap)}")
        total = sum(len(values) for values in groups.values())
        if total != len(set().union(*groups.values())):
            raise SplitError("episode重复出现")

    def as_record(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "holdout": list(self.holdout),
        }


def split_by_episode(
    episode_ids: Iterable[int],
    *,
    validation_fraction: float = 0.15,
    holdout_fraction: float = 0.15,
    seed: int = 0,
) -> EpisodeSplit:
    """按episode标识确定性划分，不做状态级随机切分。"""
    if not 0 <= validation_fraction < 1 or not 0 <= holdout_fraction < 1:
        raise SplitError("划分比例必须在[0,1)")
    if validation_fraction + holdout_fraction >= 1:
        raise SplitError("验证与留出比例之和必须小于1")
    unique = sorted(set(int(value) for value in episode_ids))
    if not unique:
        raise SplitError("没有可用的episode")
    import random

    order = list(unique)
    random.Random(seed).shuffle(order)
    total = len(order)
    holdout_count = int(total * holdout_fraction)
    validation_count = int(total * validation_fraction)
    if total >= 3 and holdout_fraction > 0:
        holdout_count = max(1, holdout_count)
    if total >= 3 and validation_fraction > 0:
        validation_count = max(1, validation_count)
    train_count = total - holdout_count - validation_count
    if train_count < 1:
        raise SplitError(f"episode数{total}不足以形成训练集")
    return EpisodeSplit(
        train=tuple(sorted(order[:train_count])),
        validation=tuple(sorted(order[train_count : train_count + validation_count])),
        holdout=tuple(sorted(order[train_count + validation_count :])),
    )


def check_no_leakage(rows: Iterable[Mapping[str, Any]], split: EpisodeSplit) -> None:
    """核对样本行的episode_id确实落在其声明集合内。"""
    mapping = {}
    for name, values in (
        ("train", split.train),
        ("validation", split.validation),
        ("holdout", split.holdout),
    ):
        for value in values:
            mapping[value] = name
    for row in rows:
        episode = int(row["episode_id"])
        declared = row.get("split")
        if episode not in mapping:
            raise SplitError(f"episode {episode} 不在任何集合")
        if declared is not None and declared != mapping[episode]:
            raise SplitError(f"episode {episode} 声明为{declared}，实际属于{mapping[episode]}")
