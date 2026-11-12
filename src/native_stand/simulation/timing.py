"""共享时相换算：物理、PD、策略三个周期的纯函数。

不依赖任何引擎；训练、评估、播放用同一份换算。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Timing:
    physics_dt_s: float
    pd_dt_s: float
    policy_dt_s: float

    def __post_init__(self) -> None:
        if not self.physics_dt_s < self.pd_dt_s <= self.policy_dt_s:
            raise ValueError("时相必须满足 physics < pd <= policy")
        # 构造时即校验整数分频，避免运行中才发现时相无法对齐。
        _ratio(self.policy_dt_s, self.physics_dt_s, "physics")
        _ratio(self.policy_dt_s, self.pd_dt_s, "pd")
        _ratio(self.pd_dt_s, self.physics_dt_s, "pd/physics")

    @staticmethod
    def from_config(control) -> Timing:
        return Timing(
            physics_dt_s=float(control.physics_dt_s),
            pd_dt_s=float(control.pd_dt_s),
            policy_dt_s=float(control.policy_dt_s),
        )

    @property
    def physics_steps_per_policy(self) -> int:
        return _ratio(self.policy_dt_s, self.physics_dt_s, "physics")

    @property
    def pd_steps_per_policy(self) -> int:
        return _ratio(self.policy_dt_s, self.pd_dt_s, "pd")

    @property
    def physics_steps_per_pd(self) -> int:
        return _ratio(self.pd_dt_s, self.physics_dt_s, "pd/physics")

    def seconds_for_updates(self, updates: int, steps_per_env: int) -> float:
        """训练消耗的策略时间；不是墙钟时间。"""
        if updates < 0 or steps_per_env < 1:
            raise ValueError("update与step必须为非负/正")
        return updates * steps_per_env * self.policy_dt_s


def _ratio(numerator: float, denominator: float, name: str) -> int:
    if denominator <= 0:
        raise ValueError(f"{name}周期必须为正")
    value = numerator / denominator
    rounded = round(value)
    if abs(value - rounded) > 1e-9 or rounded < 1:
        raise ValueError(f"{name}周期必须是策略周期的整数分频，实际比值 {value}")
    return int(rounded)
