"""完整训练回合的movement分项读回；不改变原判定、控制或采样。"""

from __future__ import annotations

import torch

VERSION = "motion_movement_components_v1"
CHECKS = ("response", "direction", "sustained", "translation", "rolling_common", "rolling_equality")
MEANS = (
    "aligned_response",
    "direction_fraction",
    "sustained_fraction",
    "horizontal_speed_mps",
    "dual_support_fraction",
    "rolling_common_abs_mps",
    "instantaneous_relative_imbalance",
    "physical_HF_tail_rad_s",
)
ROLLING_MEANS = (
    "pair0_mean_mps",
    "pair1_mean_mps",
    "relative_imbalance_of_mean",
    "opposite_fraction",
    "common_abs_of_mean_mps",
)
LIMITS = dict(
    response_fraction=0.8,
    endpoint4_response_fraction=0.95,
    endpoint4_threshold=3.9,
    direction_fraction=0.9,
    sustained_fraction=0.7,
    horizontal_speed_mps=0.30,
    rolling_common_abs_mps=0.12,
    instantaneous_relative_imbalance=0.20,
    imbalance_denominator_floor_mps=0.02,
)


def movement_checks(mean, command):
    """逐列复算冻结统计器原有六项；返回判定及其适用范围。"""
    mag = command.abs().sum(-1)
    yaw = command[:, 1] != 0
    moving = (command != 0).any(-1)
    minimum = torch.where(mag >= 3.9, mag * 0.95, mag * 0.8)
    checks = torch.stack(
        (
            mean[:, 0] >= minimum,
            mean[:, 1] >= 0.9,
            mean[:, 2] >= 0.7,
            mean[:, 3] <= 0.30,
            mean[:, 5] <= 0.12,
            mean[:, 6] <= 0.20,
        ),
        -1,
    )
    applicable = torch.stack((moving, moving, moving, yaw, yaw, yaw), -1)
    return checks | ~applicable, applicable


def episode_class(base_class, *, progression_gate=None):
    """包装已有完整回合统计器。必须由新声明的进程显式启用。"""

    class MovementEpisodes(base_class):
        movement_statistics_version = VERSION

        def __init__(self, env):
            super().__init__(env)
            self.movement_components_failed = torch.zeros(self.c, 6, device=env.device)
            self.movement_components_applicable = self.movement_components_failed.clone()
            self.movement_union_failed = self.count.clone()
            self.rolling_hold_sum = torch.zeros(env.num_envs, 2, device=env.device)
            self.opposite_hold_sum = self.n.clone()
            self.rolling_aggregate = torch.zeros(self.c, 5, device=env.device)
            self.rolling_mean_equality_passed = self.count.clone()
            self.observed_policy_steps = 0
            if progression_gate:
                self.progression_passed = self.count.clone()

        def reset(self, ids):
            super().reset(ids)
            self.rolling_hold_sum[ids] = 0
            self.opposite_hold_sum[ids] = 0

        def observe(self, result):
            # The base clears finished rows during observe. Reconstruct the same
            # final hold means first, without editing its buffers or the env.
            e = self.env
            self.observed_policy_steps += 1
            valid = e.student_actor_valid
            age = e.rise_hold.episode.steps.float() * 0.01
            hold = valid & (age >= 2.0) & (age < 27.0)
            command = e.target_xy
            yaw = command[:, 1] != 0
            magnitude = command.abs().sum(-1)
            response = torch.where(yaw, e.recovery_snapshot[:, 9], e.recovery_snapshot[:, 7])
            sign = torch.where(yaw, command[:, 1], command[:, 0]).sign()
            signed = response * sign
            minimum = torch.where(magnitude >= 3.9, magnitude * 0.95, magnitude * 0.8)
            sf = e.student_reward_result["symmetry_diagnostics"]
            diff = e.student_reward_result["differential_diagnostics"]
            values = torch.stack(
                (
                    signed,
                    (signed > 0).float(),
                    (signed >= minimum).float(),
                    e.recovery_snapshot[:, 7:9].norm(dim=-1),
                    (e.recovery_snapshot[:, 10:12] > 2).all(-1).float(),
                    sf["rolling_common_differential_mps"][:, 0].abs(),
                    diff["imbalance"].reshape(-1),
                ),
                -1,
            )
            hold_count = self.hold + hold.float()
            mean = (self.sums + values * hold[:, None]) / hold_count.clamp_min(1)[:, None]
            ended = result[0] | result[1]
            finished = ended & (self.n + valid.float() > 0)
            checks, applicable = movement_checks(mean, command)
            index = e.command_index
            self.movement_components_failed.scatter_add_(
                0, index[:, None].expand(-1, 6), ((~checks) & finished[:, None]).float()
            )
            self.movement_components_applicable.scatter_add_(
                0, index[:, None].expand(-1, 6), (applicable & finished[:, None]).float()
            )
            self.movement_union_failed.scatter_add_(0, index, (finished & ~checks.all(-1)).float())

            rolling = diff["rolling"]
            if rolling.shape != self.rolling_hold_sum.shape:
                raise ValueError("movement rolling必须为每环境两轮，保持原测量顺序")
            self.rolling_hold_sum += rolling * hold[:, None]
            self.opposite_hold_sum += ((rolling[:, 0] * rolling[:, 1] < 0) & hold).float()
            average = self.rolling_hold_sum / hold_count.clamp_min(1)[:, None]
            ratio = average.sum(-1).abs() / average.abs().sum(-1).clamp_min(0.02)
            extra = torch.cat(
                (
                    average,
                    ratio[:, None],
                    (self.opposite_hold_sum / hold_count.clamp_min(1))[:, None],
                    average.mean(-1).abs()[:, None],
                ),
                -1,
            )
            self.rolling_aggregate.scatter_add_(
                0, index[:, None].expand(-1, 5), extra * finished[:, None]
            )
            self.rolling_mean_equality_passed.scatter_add_(
                0, index, (finished & yaw & (ratio <= 0.20)).float()
            )
            if progression_gate:
                from .highspeed_course import PROGRESSION

                if progression_gate != PROGRESSION:
                    raise ValueError("未知训练晋级门")
                # 复算同一完整回合的物理质量；原passed和各失败分项保持不变。
                from native_stand.recovery.standing import longitudinal_geometry

                p = e.standing_snapshot
                g = e.wheel_geometry_snapshot
                s = e.recovery_snapshot
                offset = longitudinal_geometry(g[:, :6].reshape(-1, 2, 3), g[:, 6:9]).abs()
                peaks = torch.stack(
                    (
                        p[:, 0].abs(),
                        p[:, 1].abs(),
                        sf["leg_swing_signed_difference_rad"].reshape(-1).abs(),
                        (sf["leg_lengths_m"][:, 0] - sf["leg_lengths_m"][:, 1]).abs(),
                        offset,
                        s[:, 12],
                        (s[:, 6] - 0.34).abs(),
                    ),
                    -1,
                )
                peak = torch.maximum(self.peak, torch.where(hold[:, None], peaks, 0.0))
                loss_run = torch.where(hold & ~(s[:, 10:12] > 2).all(-1), self.loss_run + 1, 0.0)
                energy = e.antiosc_physical_stats_post[:, 4:8].square().mean(-1) * valid
                hf = (
                    (self.hf_sum + energy - self.hf[self.tick % 200]).clamp_min(0)
                    / (self.n + valid.float()).clamp(1, 200)
                ).sqrt()
                complete = (self.n + valid.float() >= 2800) & (hold_count >= 1000) & ~result[0]
                quality = (
                    complete
                    & (peak <= peaks.new_tensor([0.10, 0.05, 0.05, 0.012, 0.02, 0.005, 0.03])).all(
                        -1
                    )
                    & (mean[:, 4] >= 0.95)
                    & (torch.maximum(self.loss_max, loss_run) <= 10)
                    & (hf <= 0.25)
                )
                differential = (average[:, 0] - average[:, 1]).abs() * 0.5
                common = average.mean(-1).abs()
                equality = common <= torch.maximum(
                    common.new_full(common.shape, 0.03), 0.20 * differential
                )
                wheel_ok = (average[:, 0] * average[:, 1] < 0) & equality
                moving = command.ne(0).any(-1)
                motion = checks[:, :3].all(-1) & (
                    ~yaw | ((mean[:, 3] <= 0.30) & (mean[:, 5] <= 0.12) & wheel_ok)
                )
                accepted = quality & torch.where(moving, motion, mean[:, 3] <= 0.30)
                self.progression_passed.scatter_add_(0, index, (finished & accepted).float())
            super().observe(result)
            keep = (~ended).float()
            self.rolling_hold_sum *= keep[:, None]
            self.opposite_hold_sum *= keep

        def export(self, update):
            # Preserve the original complete-episode means before base.export
            # clears aggregates. In-progress episodes continue across exports.
            means = (self.aggregate / self.count.clamp_min(1)[:, None]).cpu().tolist()
            rolling = (self.rolling_aggregate / self.count.clamp_min(1)[:, None]).cpu().tolist()
            failures = self.movement_components_failed.cpu().tolist()
            applicable = self.movement_components_applicable.cpu().tolist()
            union = self.movement_union_failed.cpu().tolist()
            equality = self.rolling_mean_equality_passed.cpu().tolist()
            progression = self.progression_passed.cpu().tolist() if progression_gate else None
            report = super().export(update)
            for i, row in enumerate(report["rows"]):
                count = row.get("ended_episodes", row.get("completed_episodes", 0))
                if int(union[i]) != row["failed_check_counts"]["movement"]:
                    raise ValueError("movement分项复算与冻结联合判定不一致")
                row["movement_components"] = dict(
                    failed_episode_counts=dict(zip(CHECKS, map(int, failures[i]), strict=False)),
                    applicable_episode_counts=dict(
                        zip(CHECKS, map(int, applicable[i]), strict=False)
                    ),
                    union_failed_episodes=int(union[i]),
                    episode_means=dict(zip(MEANS, means[i], strict=False)) if count else None,
                    rolling_episode_means=dict(zip(ROLLING_MEANS, rolling[i], strict=False))
                    if count
                    else None,
                    mean_rolling_equality_pass_fraction=(
                        equality[i] / count if count and row["command"][1] else None
                    ),
                )
                if progression is not None:
                    row["progression_passed_episodes"] = int(progression[i])
                    row["progression_pass_fraction"] = progression[i] / count if count else None
            report["movement_statistics"] = dict(
                version=VERSION,
                thresholds=dict(LIMITS),
                gate_changed=False,
                hold_window_s=[2.0, 27.0],
                observation_sample_rate_Hz=100,
                denominator="ended episodes, same as original pass_fraction; episode means have equal weight",
                rolling_pair_order="same as differential_diagnostics.rolling, no channel remapping",
                instantaneous_equality_is_current_gate=True,
                equality_of_mean_is_diagnostic_only=True,
                subcheck_failures_can_overlap=True,
                extra_physics_steps=0,
                observed_policy_steps=self.observed_policy_steps,
            )
            self.movement_components_failed.zero_()
            self.movement_components_applicable.zero_()
            self.movement_union_failed.zero_()
            self.rolling_aggregate.zero_()
            self.rolling_mean_equality_passed.zero_()
            if progression_gate:
                report["progression_gate"] = dict(
                    version=progression_gate,
                    physics_quality_unchanged=True,
                    historical_checks_unchanged=True,
                    rolling="opposite hold means; abs common of means <= max(.03mps,.20*abs differential of means)",
                    scope="training curriculum only, not independent acceptance",
                )
                self.progression_passed.zero_()
            return report

    return MovementEpisodes
