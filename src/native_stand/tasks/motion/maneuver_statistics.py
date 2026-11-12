"""Per-episode actual support, posture, response and HF; no extra simulation."""

from __future__ import annotations

import torch

from .maneuver_targets import MINIMUM_SIGNED_YAW_GAIN, mode_masks

VERSION = "motion48_complete_episode_physical_statistics_v1"
RESPONSE_VERSION = "motion48_response_checked_physical_statistics_v2"


def episode_class(base_class, *, version=VERSION):
    if version not in (VERSION, RESPONSE_VERSION):
        raise ValueError("Explicit maneuver statistics version required")
    response_checked = version == RESPONSE_VERSION

    class Episodes(base_class):
        def __init__(self, env):
            self.env = env
            self.movement_statistics_version = version
            self.observed_policy_steps = 0
            n = env.num_envs
            k = len(env.stage_spec["commands"])
            device = env.device
            self.n = torch.zeros(n, device=device)
            self.hold = self.n.clone()
            self.sums = torch.zeros(n, 13, device=device)
            self.bad = torch.zeros(n, 6, device=device)
            self.longest_loss = self.n.clone()
            self.loss_run = self.n.clone()
            self.count = torch.zeros(k, device=device)
            self.complete = self.count.clone()
            self.passed = self.count.clone()
            self.quality = self.count.clone()
            self.group_sums = torch.zeros(k, 15, device=device)
            self.pending = None
            self.valid_total = torch.zeros((), device=device, dtype=torch.long)
            self.samples_by_cohort = torch.zeros(k, device=device, dtype=torch.long)
            if response_checked:
                self.response_sums = torch.zeros(n, 4, device=device)
                self.response_groups = torch.zeros(k, 6, device=device)

        def completed_state(self):
            names = [
                "count",
                "complete",
                "passed",
                "quality",
                "group_sums",
                "valid_total",
                "samples_by_cohort",
            ]
            if response_checked:
                names.append("response_groups")
            return dict(
                keys=self.env.stage_spec["cohort_keys"],
                version=version,
                tensors={name: getattr(self, name).detach().cpu().clone() for name in names},
            )

        def restore_completed(self, record):
            current = self.completed_state()
            if (
                record.get("version") != version
                or record["keys"] != current["keys"]
                or record["tensors"].keys() != current["tensors"].keys()
            ):
                raise ValueError("Maneuver completed statistics contract differs")
            for name, value in record["tensors"].items():
                target = getattr(self, name)
                if value.shape != target.shape or not torch.isfinite(value).all():
                    raise ValueError("Maneuver completed statistics damaged")
                target.copy_(value.to(target.device))

        def reset(self, ids):
            self.n[ids] = 0
            self.hold[ids] = 0
            self.sums[ids] = 0
            self.bad[ids] = 0
            self.longest_loss[ids] = 0
            self.loss_run[ids] = 0
            if response_checked:
                self.response_sums[ids] = 0

        def observe(self, result):
            # Called by the frozen layer before the final mode-aware reward.
            self.pending = result

        def observe_final(self, result):
            e = self.env
            self.observed_policy_steps += 1
            valid = e.student_actor_valid
            age = e.rise_hold.episode.steps.float() * 0.01
            c = e.motion_action_command
            s = e.recovery_snapshot
            p = e.standing_snapshot
            mode = mode_masks(c, is_flat=getattr(e, "motion_flat_terrain", None))
            changing = (e.maneuver_goal_height - 0.34).abs() > 1e-6
            start = torch.where(
                changing,
                torch.where(mode["travel_turn"], 13.0, 10.0),
                torch.where(mode["travel_turn"], 7.0, 5.0),
            )
            if (
                e.stage_spec.get("command_program_version")
                == "motion48_height020_rate_limited_schedule_v3"
            ):
                from .maneuver_program import extended_scored_start

                start = extended_scored_start(e.target_xy, e.maneuver_goal_height)
            window = valid & (age >= start) & (age <= 31.0)
            self.n.add_(valid.float())
            self.hold.add_(window.float())
            self.valid_total.add_(valid.sum())
            self.samples_by_cohort.index_add_(0, e.command_index, valid.long())
            support = (s[:, 10:12] > 2).all(-1)
            self.loss_run.copy_(torch.where(window & ~support, self.loss_run + 1, 0))
            self.longest_loss.copy_(torch.maximum(self.longest_loss, self.loss_run))
            hf = e.antiosc_physical_stats_post[:, 4:8].square().mean(-1)
            sym = mode["symmetry_required"]
            roll_limit = torch.where(mode["travel_turn"], s.new_tensor(0.35), s.new_tensor(0.05))
            rolling = e.motion_spin_measurement_post["rolling_speed_forward_mps"]
            values = torch.stack(
                (
                    support.float(),
                    s[:, 7],
                    s[:, 9],
                    (s[:, 6] - c[:, 2]).abs(),
                    (s[:, 7] - c[:, 0]).abs(),
                    (s[:, 9] - c[:, 1]).abs(),
                    p[:, 0].abs(),
                    p[:, 1].abs(),
                    hf,
                    s[:, 8].abs(),
                    rolling[:, 0],
                    rolling[:, 1],
                    (rolling[:, 0] * rolling[:, 1] < 0).float(),
                ),
                -1,
            )
            self.sums.add_(values * window[:, None])
            if response_checked:
                from .maneuver_targets import roll_target

                command_bank = roll_target(c)
                # Signed physical roll and measured-motion reference are diagnostics,
                # not a new reward or a claim of centripetal-force identification.
                actual_bank = torch.where(
                    mode["travel_turn"],
                    -torch.atan(s[:, 7] * s[:, 9] / 9.81).clamp(-0.30, 0.30),
                    torch.zeros_like(s[:, 7]),
                )
                diagnostic = torch.stack(
                    (p[:, 1], command_bank, actual_bank, (p[:, 1] - actual_bank).abs()), -1
                )
                self.response_sums.add_(diagnostic * window[:, None])
            violations = torch.stack(
                (
                    p[:, 0].abs() > 0.1,
                    p[:, 1].abs() > roll_limit,
                    ((p[:, 2] - p[:, 3]).abs() > 0.02) & sym,
                    (p[:, 4:6].abs().amax(-1) > 0.10) & sym,
                    (e.rise_hold.offset.abs() > 0.04) & sym,
                    (e.maneuver_swing_difference.abs() > 0.05) & sym,
                ),
                -1,
            )
            self.bad.add_(violations * window[:, None])
            ended = (result[0] | result[1]) & valid
            ids = ended.nonzero(as_tuple=True)[0]
            if not ids.numel():
                return
            index = e.command_index[ids]
            count = self.hold[ids]
            mean = self.sums[ids] / count.clamp_min(1)[:, None]
            full = (self.n[ids] >= 3000) & (count >= 1500) & ~result[0][ids]
            pose = (self.bad[ids] / count.clamp_min(1)[:, None] <= 0.05).all(-1)
            support_ok = (mean[:, 0] >= 0.95) & (self.longest_loss[ids] <= 20)
            # No precision parking requirement; coarse standing drift is independent.
            vx_tol = torch.where(
                mode["standing"][ids],
                s.new_tensor(0.4),
                torch.maximum(c[ids, 0].abs() * 0.20, s.new_tensor(0.30)),
            )
            yaw_tol = torch.maximum(c[ids, 1].abs() * 0.20, s.new_tensor(0.30))
            quality = full & pose & support_ok & (mean[:, 3] <= 0.02) & (mean[:, 8].sqrt() <= 0.30)
            common = (mean[:, 10] + mean[:, 11]).abs() * 0.5
            differential = (mean[:, 10] - mean[:, 11]).abs() * 0.5
            spin_ok = ~mode["spin"][ids] | (
                (common <= torch.maximum(common.new_tensor(0.03), differential * 0.20))
                & (mean[:, 12] >= 0.95)
            )
            passed = (
                quality
                & (mean[:, 4] <= vx_tol)
                & (mean[:, 5] <= yaw_tol)
                & (mean[:, 9] <= 0.30)
                & spin_ok
            )
            if response_checked:
                required = c[ids, 1].abs() > 0.05
                gain = mean[:, 2] * c[ids, 1].sign() / c[ids, 1].abs().clamp_min(0.05)
                response_ok = ~required | (gain >= MINIMUM_SIGNED_YAW_GAIN)
                passed &= response_ok
                extra = torch.cat(
                    (
                        self.response_sums[ids] / count.clamp_min(1)[:, None],
                        torch.where(required, gain, torch.zeros_like(gain))[:, None],
                        response_ok.float()[:, None],
                    ),
                    -1,
                )
                self.response_groups.index_add_(0, index, extra)
            self.count.index_add_(0, index, torch.ones_like(count))
            self.complete.index_add_(0, index, full.float())
            self.quality.index_add_(0, index, quality.float())
            self.passed.index_add_(0, index, passed.float())
            metrics = torch.cat(
                (mean, self.longest_loss[ids, None] * 0.01, full.float()[:, None]), -1
            )
            self.group_sums.index_add_(0, index, metrics)

        def export(self, update):
            stage = self.env.stage_spec
            all_values = (
                torch.cat(
                    (
                        self.count[:, None],
                        self.complete[:, None],
                        self.passed[:, None],
                        self.quality[:, None],
                        self.group_sums,
                    ),
                    -1,
                )
                .cpu()
                .tolist()
            )
            samples = self.samples_by_cohort.cpu().tolist()
            rows = []
            responses = self.response_groups.cpu().tolist() if response_checked else None
            for i, a in enumerate(all_values):
                n = a[0]
                v = [x / n if n else None for x in a[4:]]
                rows.append(
                    dict(
                        key=stage["cohort_keys"][i],
                        command=stage["commands"][i],
                        height=stage["goal_heights_m"][i],
                        purpose=stage["cohort_purposes"][i],
                        profile=stage["identified_profiles"][i],
                        axis=stage["maneuver_tasks"][i],
                        endpoint=True,
                        slot_count=stage["slot_counts"][i],
                        actual_student_policy_samples=samples[i],
                        ended_episodes=int(n),
                        completed_episodes=int(a[1]),
                        pass_fraction=a[2] / n if n else None,
                        quality_pass_fraction=a[3] / n if n else None,
                        progression_pass_fraction=a[2] / n if n else None,
                        mean_dual_support_fraction=v[0],
                        mean_physical_vx=v[1],
                        mean_physical_yaw=v[2],
                        height_MAE_m=v[3],
                        vx_MAE_mps=v[4],
                        yaw_MAE_rad_s=v[5],
                        pitch_abs_mean_rad=v[6],
                        roll_abs_mean_rad=v[7],
                        physical_HF_RMS_rad_s=v[8] ** 0.5 if n else None,
                        mean_uncommanded_lateral_mps=v[9],
                        hold_mean_rolling_forward_mps=v[10:12],
                        opposite_rolling_fraction=v[12],
                        mean_longest_double_support_loss_s=v[13],
                        evidence="completed stochastic training episodes, multiple policy versions",
                        independent_certificate=False,
                    )
                )
                if response_checked:
                    extra = [x / n if n else None for x in responses[i]]
                    required = abs(stage["commands"][i][1]) > 0.05
                    rows[-1].update(
                        yaw_response_required=required,
                        minimum_signed_yaw_gain=MINIMUM_SIGNED_YAW_GAIN if required else None,
                        mean_signed_yaw_gain=extra[4] if required else None,
                        yaw_response_pass_fraction=extra[5] if required else None,
                        signed_roll_mean_rad=extra[0],
                        command_bank_mean_rad=extra[1],
                        measured_motion_bank_mean_rad=extra[2],
                        measured_motion_bank_MAE_rad=extra[3],
                        bank_diagnostic="measured vx*yaw projection; not independent force/curve certification",
                    )
            self.count.zero_()
            self.complete.zero_()
            self.passed.zero_()
            self.quality.zero_()
            self.group_sums.zero_()
            if response_checked:
                self.response_groups.zero_()
            return dict(
                version=version,
                stage_revision=stage["id"],
                through_update=update,
                rows=rows,
                valid_student_steps=int(self.valid_total),
                observed_policy_steps=self.observed_policy_steps,
                denominator="all ended student episodes; absent samples UNKNOWN; early failures included",
            )

    return Episodes
