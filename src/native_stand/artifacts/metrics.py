"""精简标量曲线：metrics.jsonl与TensorBoard使用同一批记录。

预算计数以运行状态为准，这里只写展示用的稀疏标量。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import append_jsonl

SCALAR_KEYS = (
    "update",
    "actual_updates",
    "samples",
    "reward_mean",
    "episode_completed",
    "nonzero_rewards",
    "value_loss",
    "surrogate",
    "entropy",
    "kl",
    "clip_fraction",
    "actor_gradient_norm",
    "critic_gradient_norm",
    "critic_warmup",
    "actor_optimizer_steps",
    "critic_optimizer_steps",
    "accepted_KL_peak",
    "actor_backtracks",
    "seconds_per_update",
    "rollout_seconds",
    "returns_seconds",
    "diagnostics_seconds",
    "learning_seconds",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
    "actor_lr",
    "critic_lr",
)


# Small, opt-in reporting summaries; TensorBoard still consumes scalars only.
PERFORMANCE_KEYS = (
    "performance_seconds",
    "performance_counters",
    "performance_parents",
    "performance_clocks",
    "initialization_measurement",
    "saved_this_update",
    "whole_update_wall_seconds_before_metrics",
    "previous_metrics_write_seconds",
    "stage_mean_wall_seconds",
)


class MetricsWriter:
    """按配置间隔写入一条精简记录，并复用同一行驱动TensorBoard。"""

    def __init__(
        self,
        path: str | Path,
        *,
        every_updates: int = 10,
        tensorboard: bool = True,
        stage: str | None = None,
    ):
        if every_updates < 1:
            raise ValueError("metrics_every_updates必须至少为1")
        self.path = Path(path)
        self.stage = stage
        self.every_updates = every_updates
        self.rows_written = 0
        self._last_update = 0
        self._board = None
        if tensorboard:
            logdir = self.path.parent / "tensorboard"
            logdir.mkdir(parents=True, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter

            self._board = SummaryWriter(str(logdir), flush_secs=30)

    def due(self, update: int) -> bool:
        return update % self.every_updates == 0

    def write(self, row: Mapping[str, Any], *, force: bool = False) -> bool:
        """写一条记录；未到期且非force时返回False。"""
        update = int(row.get("update", 0))
        if not force and not self.due(update):
            return False
        record = {key: row[key] for key in (*SCALAR_KEYS, *PERFORMANCE_KEYS) if key in row}
        record["update"] = update
        if self.stage is not None:
            record["stage"] = self.stage
        append_jsonl(self.path, [record])
        self.rows_written += 1
        self._last_update = update
        if self._board is not None:
            for key, value in record.items():
                if key != "update" and isinstance(value, int | float):
                    self._board.add_scalar(key, float(value), update)
        return True

    def close(self) -> None:
        if self._board is not None:
            self._board.flush()
            self._board.close()
            self._board = None
