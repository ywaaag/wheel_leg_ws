"""新C0课程的事件适配；继承冻结力路由，仅改matcher与观测统计。"""

from __future__ import annotations

import torch

from native_stand.tasks.motion.transient_statistics import Statistics, handover_purpose


def event_class(base, noise):
    class Events(base):
        def __init__(self, env):
            super().__init__(env)
            self.statistics = Statistics(env.num_envs, env.stage_spec, device=env.device)
            self.teacher = torch.zeros_like(self.active)

        def reset(self, ids):
            e = self.env
            u = noise(e.identified_dr.episodes[ids], ids, 6, 1213)
            index = e.command_index[ids]
            mode = torch.tensor(e.stage_spec["modal_modes"], device=e.device)[index]
            sign = torch.tensor(e.stage_spec["modal_signs"], device=e.device)[index]
            hand = torch.tensor(
                [handover_purpose(p) for p in e.stage_spec["cohort_purposes"]], device=e.device
            )[index]
            self.enabled[ids] = (mode >= 0) | (hand & (u[:, 0] > 0))
            self.mode[ids] = torch.where(mode >= 0, mode, ((u[:, 1] + 1) * 2).long().clamp_max(3))
            self.sign[ids] = torch.where(
                mode >= 0, sign.float(), torch.where(u[:, 2] >= 0, 1.0, -1.0)
            )
            self.direction[ids] = self.basis[self.mode[ids]]
            self.next[ids] = (
                torch.where(hand, 300 + (u[:, 3] + 1) * 250, 2000 + (u[:, 3] + 1) * 1000)
                .round()
                .long()
            )
            self.interval[ids] = (6000 + (u[:, 4] + 1) * 2000).round().long()
            self.aborted[ids] += self.pending[ids].long()
            self.pending[ids] = False
            self.quiet_run[ids] = 0
            self.age[ids] = 0
            self.last[ids] = -100000
            self.current[ids] = 0
            self.active[ids] = False
            self.teacher[ids] = False
            self.statistics.reset(ids, index, self.mode[ids], self.sign[ids])

        def policy(self, student):
            super().policy(student)
            self.teacher.copy_(self.env.r4_owner == 2)
            self.statistics.policy(student, self.teacher, self.enabled)

        def observe(self):
            # Pulses/1ms force generation remain the exact frozen base.physical implementation.
            # Attribute begins once per policy snapshot; no extra kernels in the 1ms force path.
            t = self.statistics
            t.begin(self.active & (self.last >= 0) & (self.last != t.last_on), self.last)
            e = self.env
            s, p = e.recovery_snapshot, e.standing_snapshot
            stats = e.antiosc_physical_stats_post
            t.observe(
                age=self.age,
                next_on=self.next,
                active=self.active & (stats[:, 8] > 0),
                support=(s[:, 10:12] > 2).all(-1),
                hf2=stats[:, 4:8].square().mean(-1),
                dq2=stats[:, :4].square().mean(-1),
                pitch=p[:, 0],
                roll=p[:, 1],
                horizontal=s[:, 7:9].norm(dim=-1),
                targets=e.motion_timing.base.adopted,
            )

        def summary(self):
            return {
                **self.statistics.summary(),
                "pulse_samples_1ms": int(self.samples.sum()),
                "peak_external_torque_norm_Nm": float(self.peak.max()),
                "physical_pulse_events": int(self.count.sum()),
                "force_off_age_ms": "recorded force_on_age+100; local student event clock, reset per episode",
                "motor_limits_raised": False,
                "physical_state_overwritten": False,
                "old_diagnostic_rows_used": False,
            }

    return Events
