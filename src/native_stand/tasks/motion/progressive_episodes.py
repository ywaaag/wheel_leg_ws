"""为冻结物理统计增加真实完整回合计数和课程身份；不额外推进仿真。"""

from __future__ import annotations


def episode_class(base_class):
    class ProgressiveEpisodes(base_class):
        def __init__(self, env):
            super().__init__(env)
            self.complete_count = self.count.clone()
            self.student_ticks = self.count.clone()
            self.teacher_ticks = self.count.clone()

        def completed_state(self):
            names = (
                "count",
                "passed",
                "quality",
                "aggregate",
                "failures",
                "complete_count",
                "student_ticks",
                "teacher_ticks",
                "movement_components_failed",
                "movement_components_applicable",
                "movement_union_failed",
                "rolling_aggregate",
                "rolling_mean_equality_passed",
                "progression_passed",
            )
            return {
                "keys": self.env.stage_spec["cohort_keys"],
                "tensors": {
                    name: getattr(self, name).detach().cpu().clone()
                    for name in names
                    if hasattr(self, name)
                },
            }

        def restore_completed(self, record):
            import torch

            if record["keys"] != self.env.stage_spec["cohort_keys"]:
                raise ValueError("已结束回合统计布局不一致")
            expected = set(self.completed_state()["tensors"])
            if set(record["tensors"]) != expected:
                raise ValueError("已结束回合统计字段缺失")
            for name, value in record["tensors"].items():
                target = getattr(self, name)
                if (
                    value.shape != target.shape
                    or not torch.isfinite(value).all()
                    or (value < 0).any()
                    and name not in ("aggregate", "rolling_aggregate")
                ):
                    raise ValueError("已结束回合统计损坏")
                target.copy_(value.to(target.device))
            # No unfinished per-environment episode is restored into a new world.

        def observe(self, result):
            e = self.env
            valid = e.student_actor_valid
            self.student_ticks.scatter_add_(0, e.command_index, valid.float())
            self.teacher_ticks.scatter_add_(0, e.command_index, (~valid).float())
            age = e.rise_hold.episode.steps.float() * 0.01
            n = self.n + valid.float()
            hold = self.hold + (valid & (age >= 2.0) & (age < 27.0)).float()
            ended = result[0] | result[1]
            complete = ended & (n >= 2800) & (hold >= 1000) & ~result[0]
            self.complete_count.scatter_add_(0, e.command_index, complete.float())
            super().observe(result)

        def export(self, update):
            report = super().export(update)
            complete = self.complete_count.cpu().tolist()
            stage = self.env.stage_spec
            student_seconds = (self.student_ticks * 0.01).cpu().tolist()
            teacher_seconds = (self.teacher_ticks * 0.01).cpu().tolist()
            self.student_ticks.zero_()
            self.teacher_ticks.zero_()
            for i, row in enumerate(report["rows"]):
                row["ended_episodes"] = row.pop("completed_episodes")
                row["completed_episodes"] = int(complete[i])
                row["student_control_seconds"] = student_seconds[i]
                row["teacher_control_seconds"] = teacher_seconds[i]
                row["axis"] = stage["motion_axes"][i]
                row["endpoint"] = stage["motion_endpoints"][i]
                row["mean_aligned_response"] = row.pop("mean_signed_response")
                target = row["command"][1] or row["command"][0]
                row["mean_physical_response"] = (
                    row["mean_aligned_response"] * (1 if target > 0 else -1)
                    if target and row["mean_aligned_response"] is not None
                    else None
                )
            self.complete_count.zero_()
            report.update(
                version="motion_progressive_complete_episode_v1",
                stage_revision=stage["id"],
                rate_denominator="all ended episodes, including early failures; completed_episodes means full-duration surviving episodes",
                response_sign="aligned=response*command_sign; physical_response is the native signed physical velocity",
            )
            return report

    return ProgressiveEpisodes
