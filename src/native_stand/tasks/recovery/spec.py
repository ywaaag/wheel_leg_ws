"""Recovery任务声明：合法Swing进入、自主起身、连续保持与接触质量门槛。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from native_stand.experiments.config import ConfigError, JobConfig

DEFAULT_HOLD_SECONDS = 10.0


@dataclass(frozen=True)
class RecoveryGate:
    """起身验收门槛；缺字段不能授予通过。"""

    hold_seconds: float
    standing_height_m: tuple[float, float]
    dual_contact_min: float
    peak_tilt_max_rad: float
    closure_max_m: float
    saturation_fraction_max: float

    def __post_init__(self) -> None:
        if self.hold_seconds <= 0:
            raise ConfigError("recovery.hold_seconds 必须为正")
        low, high = self.standing_height_m
        if not low < high:
            raise ConfigError("recovery站立高度范围不合法")
        if not 0 < self.dual_contact_min <= 1:
            raise ConfigError("recovery dual_contact_min 必须在(0,1]")

    def as_record(self) -> Mapping[str, Any]:
        return {
            "hold_seconds": self.hold_seconds,
            "standing_height_m": list(self.standing_height_m),
            "dual_contact_min": self.dual_contact_min,
            "peak_tilt_max_rad": self.peak_tilt_max_rad,
            "closure_max_m": self.closure_max_m,
            "saturation_fraction_max": self.saturation_fraction_max,
        }


@dataclass(frozen=True)
class RecoveryTask:
    job: JobConfig
    gate: RecoveryGate
    entry_family: str
    stand_checkpoint: str | None
    stand_sha256: str | None

    def evaluation_protocol(self) -> dict[str, Any]:
        evaluation = self.job.evaluation
        return {
            "version": evaluation.protocol_version,
            "envs": evaluation.envs,
            "scored_envs": evaluation.scored_envs,
            "seconds": evaluation.seconds,
            "seed": evaluation.seed,
            "gate": dict(self.gate.as_record()),
            "conditions": [dict(row) for row in evaluation.conditions],
        }


def build_task(job: JobConfig) -> RecoveryTask:
    """校验并构造Recovery任务视图。"""
    if job.task != "recovery":
        raise ConfigError(f"该配置的 task={job.task}，不能作为Recovery任务运行")
    recovery = dict(job.assets.get("recovery") or {})
    if job.action.version.startswith("motion31"):
        raise ConfigError("Recovery不能复用Motion动作版本")
    gate_raw = dict(recovery.get("gate") or {})
    evaluation = job.evaluation
    gate = RecoveryGate(
        hold_seconds=float(gate_raw.get("hold_seconds", DEFAULT_HOLD_SECONDS)),
        standing_height_m=tuple(
            float(v) for v in (gate_raw.get("standing_height_m") or (0.34, 0.38))
        ),
        dual_contact_min=float(gate_raw.get("dual_contact_min", 0.95)),
        peak_tilt_max_rad=float(gate_raw.get("peak_tilt_max_rad", 0.18)),
        closure_max_m=float(gate_raw.get("closure_max_m", 0.005)),
        saturation_fraction_max=float(gate_raw.get("saturation_fraction_max", 0.05)),
    )
    if evaluation.seconds < gate.hold_seconds:
        raise ConfigError(
            f"evaluation.seconds={evaluation.seconds} 短于保持要求{gate.hold_seconds}s"
        )
    entry = str(recovery.get("entry_family") or "swing")
    if entry != "swing":
        raise ConfigError(f"Recovery只从合法Swing进入；收到 {entry}")
    stand = recovery.get("stand_checkpoint")
    stand_sha = recovery.get("stand_sha256")
    if stand and not stand_sha:
        raise ConfigError("assets.recovery.stand_checkpoint 需要同时给出 stand_sha256")
    return RecoveryTask(
        job=job,
        gate=gate,
        entry_family=entry,
        stand_checkpoint=str(stand) if stand else None,
        stand_sha256=str(stand_sha) if stand_sha else None,
    )
