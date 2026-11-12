"""纯张量事件统计：不改状态、控制、奖励或随机数流。"""

from __future__ import annotations

import torch

VERSION = "motion_C0_force_off_recovery_statistics_v2"
PHASES = (
    "pre_force_on_0p5s",
    "pulse_0p1s",
    "force_off_0_to_0p4s",
    "force_off_0p4_to_2s",
    "force_off_2_to_5s",
)
LEGS = (0, 1, 3, 4)


def handover_purpose(purpose):
    return purpose.startswith(("dq_continuous_handover", "continuous_C0_handover_"))


def configuration():
    return dict(
        version=VERSION,
        policy_period_ms=10,
        pulse_width_ms=100,
        quiet_confirmation_ms=1000,
        phase_names=list(PHASES),
        force_off_windows_ms=[[0, 400], [400, 2000], [2000, 5000]],
        pre_window="500ms before scheduled force-on; excludes pulse",
        no_samples="UNKNOWN/null, never zero",
        HF="HP10 residual; not 10–40Hz band RMS",
        recovery="absolute physical dq RMS<=.25rad/s, HP10 RMS<=.25rad/s, pitch<=.10rad, "
        "roll<=.05rad, dual>2N, horizontal speed<=.30m/s for >=1000ms after force-off",
        decay="event-paired tail/catch HP10 RMS; requires complete windows and catch>.01rad/s; diagnostic only",
        reward_changed=False,
        feedback_changed=False,
    )


def _nullable(values, count):
    """JSON在统计导出边界生成；设备热路径不做同步。"""
    a, n = values.detach().cpu().tolist(), count.detach().cpu().tolist()

    def visit(x, c):
        return (
            [visit(v, w) for v, w in zip(x, c, strict=False)]
            if isinstance(x, list)
            else (x if c > 0 else None)
        )

    return visit(a, n)


class Statistics:
    def __init__(self, envs, stage, *, device):
        self.stage = stage
        self.n, self.c = envs, len(stage["commands"])
        self.device = device
        self.cohort = torch.zeros(envs, device=device, dtype=torch.long)
        self.mode = torch.zeros_like(self.cohort)
        self.sign = torch.ones(envs, device=device)
        self.last_on = torch.full_like(self.cohort, -100000)
        self.force_off = torch.full_like(self.cohort, -100000)
        self.pending = torch.zeros(envs, device=device, dtype=torch.bool)
        self.quiet_start = torch.full_like(self.cohort, -1)
        self.enter_age = self.quiet_start.clone()
        self.confirm_age = self.quiet_start.clone()
        self.previous_target = torch.zeros(envs, 6, device=device)
        self.target_valid = self.pending.clone()
        self.loss_run = torch.zeros(envs, device=device, dtype=torch.long)
        self.event_count = torch.zeros(self.c, device=device, dtype=torch.long)
        self.mode_sign_count = torch.zeros(self.c, 4, 2, device=device, dtype=torch.long)
        self.student_steps = torch.zeros(self.c, device=device, dtype=torch.long)
        self.teacher_steps = torch.zeros(self.c, device=device)
        self.enabled_student_steps = torch.zeros(self.c, device=device)
        self.count = torch.zeros(self.c, len(PHASES), device=device, dtype=torch.long)
        self.metrics = {
            k: torch.zeros_like(self.count, dtype=torch.float64)
            for k in (
                "support",
                "hf2",
                "dq2",
                "pitch",
                "roll",
                "horizontal",
                "leg_target_step2",
                "wheel_target_step2",
            )
        }
        self.target_count = self.count.clone()
        self.loss_max_ms = self.count.clone()
        self.recovered = self.event_count.clone()
        self.aborted = self.event_count.clone()
        self.enter_sum_s = torch.zeros_like(self.event_count, dtype=torch.float64)
        self.enter_max_s = self.enter_sum_s.clone()
        self.confirm_sum_s = self.enter_sum_s.clone()
        self.decay_count = self.event_count.clone()
        self.decay_ratio_sum = self.enter_sum_s.clone()
        self.decay_below_floor = self.event_count.clone()
        self.event_hf2 = torch.zeros(envs, 2, device=device)
        self.event_samples = torch.zeros_like(self.event_hf2, dtype=torch.long)

    def _add(self, destination, value):
        destination.index_add_(0, self.cohort, value.to(destination.dtype))

    def _finalize_decay(self, mask):
        enough = mask & (self.event_samples[:, 0] >= 40) & (self.event_samples[:, 1] >= 300)
        rms = (self.event_hf2 / self.event_samples.clamp_min(1)).sqrt()
        usable = enough & (rms[:, 0] > 0.01)
        self._add(self.decay_count, usable)
        self._add(self.decay_below_floor, enough & ~usable)
        self._add(
            self.decay_ratio_sum, torch.where(usable, rms[:, 1] / rms[:, 0].clamp_min(1e-12), 0.0)
        )

    def reset(self, ids, cohort, mode, sign):
        mask = torch.zeros_like(self.pending)
        mask[ids] = True
        self._finalize_decay(mask)
        self._add(self.aborted, mask & self.pending)
        self.cohort[ids], self.mode[ids], self.sign[ids] = cohort, mode, sign
        self.last_on[ids] = -100000
        self.force_off[ids] = -100000
        self.pending[ids] = False
        self.quiet_start[ids] = -1
        self.enter_age[ids] = -1
        self.confirm_age[ids] = -1
        self.target_valid[ids] = False
        self.loss_run[ids] = 0
        self.event_hf2[ids] = 0
        self.event_samples[ids] = 0

    def policy(self, student, teacher, enabled):
        self._add(self.student_steps, student)
        self._add(self.teacher_steps, teacher)
        self._add(self.enabled_student_steps, student & enabled)

    def begin(self, mask, age):
        self._finalize_decay(mask)
        self._add(self.aborted, mask & self.pending)
        self._add(self.event_count, mask)
        # Fixed-size tensor attribution is captured before a future partial reset.
        flat = self.cohort * 8 + self.mode * 2 + (self.sign > 0).long()
        self.mode_sign_count.view(-1).index_add_(0, flat, mask.to(self.event_count.dtype))
        self.last_on.copy_(torch.where(mask, age, self.last_on))
        self.force_off.copy_(torch.where(mask, age + 100, self.force_off))
        self.pending.logical_or_(mask)
        self.quiet_start.masked_fill_(mask, -1)
        self.enter_age.masked_fill_(mask, -1)
        self.confirm_age.masked_fill_(mask, -1)
        self.event_hf2.masked_fill_(mask[:, None], 0.0)
        self.event_samples.masked_fill_(mask[:, None], 0.0)

    def observe(self, *, age, next_on, active, support, hf2, dq2, pitch, roll, horizontal, targets):
        after = age - self.force_off
        has_event = self.last_on >= 0
        flags = (
            torch.stack(
                (
                    (next_on - age >= 0) & (next_on - age < 500),
                    has_event & (age >= self.last_on) & (after < 0),
                    has_event & (after >= 0) & (after < 400),
                    has_event & (after >= 400) & (after < 2000),
                    has_event & (after >= 2000) & (after < 5000),
                ),
                -1,
            )
            & active[:, None]
        )
        weight = flags.to(self.count.dtype)
        self._add(self.count, weight)
        delta2 = (targets - self.previous_target).square()
        valid_weight = weight * self.target_valid[:, None]
        for key, value in dict(
            support=support.float(),
            hf2=hf2,
            dq2=dq2,
            pitch=pitch.abs(),
            roll=roll.abs(),
            horizontal=horizontal,
        ).items():
            self._add(self.metrics[key], weight * value[:, None])
        self._add(
            self.metrics["leg_target_step2"], valid_weight * delta2[:, LEGS].mean(-1)[:, None]
        )
        self._add(
            self.metrics["wheel_target_step2"], valid_weight * delta2[:, (2, 5)].mean(-1)[:, None]
        )
        self._add(self.target_count, valid_weight)
        self.previous_target.copy_(targets)
        self.target_valid.copy_(active)
        self.loss_run.copy_(torch.where(active & ~support, self.loss_run + 10, 0))
        # Maximum of each continuous loss run in each phase, including across a phase boundary.
        value = flags.long() * self.loss_run[:, None]
        self.loss_max_ms.scatter_reduce_(
            0, self.cohort[:, None].expand_as(value), value, reduce="amax", include_self=True
        )
        event_weight = flags[:, (2, 4)].long()
        self.event_hf2.add_(event_weight * hf2[:, None])
        self.event_samples.add_(event_weight)
        quiet = (
            active
            & self.pending
            & (after >= 0)
            & support
            & (dq2 <= 0.25**2)
            & (hf2 <= 0.25**2)
            & (pitch.abs() <= 0.10)
            & (roll.abs() <= 0.05)
            & (horizontal <= 0.30)
        )
        self.quiet_start.copy_(
            torch.where(quiet, torch.where(self.quiet_start < 0, age, self.quiet_start), -1)
        )
        confirmed = quiet & (age - self.quiet_start >= 1000)
        self.enter_age.copy_(torch.where(confirmed, self.quiet_start, self.enter_age))
        self.confirm_age.copy_(torch.where(confirmed, age, self.confirm_age))
        delay = (self.quiet_start - self.force_off).clamp_min(0).float() * 0.001
        self._add(self.recovered, confirmed)
        self._add(self.enter_sum_s, torch.where(confirmed, delay, 0.0))
        self._add(self.confirm_sum_s, torch.where(confirmed, after.float() * 0.001, 0.0))
        self.enter_max_s.scatter_reduce_(
            0,
            self.cohort,
            torch.where(confirmed, delay, 0.0).to(self.enter_max_s.dtype),
            reduce="amax",
            include_self=True,
        )
        self.pending.logical_and_(~confirmed)

    def summary(self):
        phases = {}
        for key, sums in self.metrics.items():
            count = self.target_count if "target_step" in key else self.count
            values = sums / count.clamp_min(1)
            if key.endswith("2"):
                values = values.sqrt()
            phases[key.removesuffix("2")] = _nullable(values, count)
        pending = torch.zeros_like(self.event_count)
        self._add(pending, self.pending)
        # Include currently complete event windows without consuming or double-counting them.
        rms = (self.event_hf2 / self.event_samples.clamp_min(1)).sqrt()
        enough = (self.event_samples[:, 0] >= 40) & (self.event_samples[:, 1] >= 300)
        usable = enough & (rms[:, 0] > 0.01)
        count, ratios, below = (
            self.decay_count.clone(),
            self.decay_ratio_sum.clone(),
            self.decay_below_floor.clone(),
        )
        self._add(count, usable)
        self._add(below, enough & ~usable)
        self._add(ratios, torch.where(usable, rms[:, 1] / rms[:, 0].clamp_min(1e-12), 0.0))
        return dict(
            version=VERSION,
            configuration=configuration(),
            cohort_keys=self.stage["cohort_keys"],
            phase_names=list(PHASES),
            phase_samples_by_cohort=self.count.cpu().tolist(),
            phase_status_by_cohort=[
                ["OBSERVED" if n else "UNKNOWN" for n in row] for row in self.count.cpu().tolist()
            ],
            phase_measurements=phases,
            phase_HF_metric="HP10 physical residual RMS, rad/s; not a 10–40Hz band",
            band_10_to_40Hz_metric=dict(
                status="NOT_COLLECTED",
                value=None,
                reason="existing 1ms HP10 energy is available; no band-limited physical trace retained by this statistic",
            ),
            phase_fullband_dq_metric="actual physical 4-leg RMS, rad/s",
            phase_dual_support_fraction=phases["support"],
            phase_physical_HF_RMS_rad_s=phases["hf"],
            phase_target_step_valid_samples=self.target_count.cpu().tolist(),
            phase_continuous_support_loss_max_ms=_nullable(self.loss_max_ms, self.count),
            pulse_events=int(self.event_count.sum()),
            pulse_events_by_cohort=self.event_count.cpu().tolist(),
            latest_event_clock=dict(
                origin="student-controlled local milliseconds, reset per episode",
                cohort_index=self.cohort.cpu().tolist(),
                force_on_age_ms=_nullable(self.last_on, (self.last_on >= 0).long()),
                force_off_age_ms=_nullable(self.force_off, (self.last_on >= 0).long()),
                confirmed_quiet_enter_age_ms=_nullable(
                    self.enter_age, (self.enter_age >= 0).long()
                ),
                quiet_confirm_age_ms=_nullable(self.confirm_age, (self.confirm_age >= 0).long()),
            ),
            pulse_events_by_cohort_mode_sign=self.mode_sign_count.cpu().tolist(),
            student_policy_samples_by_cohort=self.student_steps.cpu().tolist(),
            teacher_policy_samples_by_cohort=self.teacher_steps.cpu().tolist(),
            event_enabled_student_samples_by_cohort=self.enabled_student_steps.cpu().tolist(),
            recovery_confirmed_by_cohort=self.recovered.cpu().tolist(),
            recovery_unconfirmed_reset_or_next_event_by_cohort=self.aborted.cpu().tolist(),
            recovery_pending_by_cohort=pending.cpu().tolist(),
            recovery_enter_delay_mean_s_by_cohort=_nullable(
                self.enter_sum_s / self.recovered.clamp_min(1), self.recovered
            ),
            recovery_enter_delay_max_s_by_cohort=_nullable(self.enter_max_s, self.recovered),
            recovery_confirmation_delay_mean_s_by_cohort=_nullable(
                self.confirm_sum_s / self.recovered.clamp_min(1), self.recovered
            ),
            complete_paired_decay_events_by_cohort=count.cpu().tolist(),
            decay_tail_catch_HP10_RMS_ratio_mean_by_cohort=_nullable(
                ratios / count.clamp_min(1), count
            ),
            decay_events_below_catch_floor_by_cohort=below.cpu().tolist(),
            scope="cumulative actual current-policy training; cohort/mode/sign/profile attributed at event; not independent evaluation",
            no_samples="UNKNOWN/null",
            reward_changed=False,
            feedback_changed=False,
        )
