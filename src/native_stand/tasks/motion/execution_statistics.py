"""Student执行诊断；按cohort累计策略步末快照，不参与评分或优化。"""

from __future__ import annotations

import torch

CHANNEL_FIELDS = (
    "raw_action_mean",
    "raw_clip_fraction",
    "requested_target_mean",
    "adopted_target_mean",
    "target_gap_abs_mean",
    "request_over_cap_fraction",
    "driver_at_cap_fraction",
    "motor_at_cap_fraction",
    "motor_capacity_fraction_mean",
)


class ExecutionStatistics:
    def __init__(self, commands, device):
        self.commands = commands
        self.count = torch.zeros(len(commands), device=device, dtype=torch.long)
        self.sums = torch.zeros(len(commands), len(CHANNEL_FIELDS), 6, device=device)
        self.steps = 0

    def observe(self, index, valid, raw, requested, adopted, saturation):
        # 调用方提供reset前的post快照及本步原始采样动作，不读取可变执行缓存。
        values = torch.stack(
            (
                raw,
                (raw.abs() > 1).float(),
                requested,
                adopted,
                (requested - adopted).abs(),
                saturation["request_over_cap"].float(),
                saturation["driver_at_cap"].float(),
                saturation["motor_at_cap"].float(),
                saturation["motor_capacity_fraction"],
            ),
            1,
        )
        self.count.index_add_(0, index, valid.long())
        self.sums.index_add_(0, index, torch.where(valid[:, None, None], values, 0.0))
        self.steps += 1

    def export(self):
        counts = self.count.cpu().tolist()
        means = (self.sums / self.count.clamp_min(1)[:, None, None]).cpu().tolist()
        return {
            "version": "motion_cohort_execution_statistics_v1",
            "scope": "report update rollout only; Student-valid policy-step-end pre-reset snapshots",
            "policy_steps": self.steps,
            "not_physics_substep_integration": True,
            "not_review_window": True,
            "channel_order": "model left hip/active/wheel, right hip/active/wheel",
            "target_units": ["rad", "rad", "rad/s", "rad", "rad", "rad/s"],
            "motor_statistics_exclude": ["NET", "passive", "external"],
            "rows": [
                {
                    "cohort_index": i,
                    "command": command,
                    "samples": counts[i],
                    "mean": dict(zip(CHANNEL_FIELDS, means[i], strict=True)) if counts[i] else None,
                }
                for i, command in enumerate(self.commands)
            ],
        }
