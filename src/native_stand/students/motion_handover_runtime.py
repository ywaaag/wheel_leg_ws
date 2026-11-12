"""Opt-in continuous source-policy handover; uses the public MCU plant/PPO.

This module imports no Isaac runtime until the common environment constructs it.
Physical state, adopted targets, sensor packets and motor queues never reset at
the teacher->student boundary. Teacher/initialization transitions remain on the
GAE timeline but cannot optimize either student network.
"""
import torch

from native_stand.recovery.controlled_ppo import pre_action_valid
from native_stand.recovery.domain import Mode


def sample_pd_age(ids, episodes, seed=20261026, maximum=20):
    """Per-env reset stream: batch order and other resets cannot change it."""
    value = (ids.long() * 1103515245 + episodes.long() * 12345 + seed) % 2147483647
    value = (value * 48271) % 2147483647
    if maximum not in (12,16,20):raise ValueError('PD-age curriculum maximum differs')
    return 8 + value % (maximum-7)


def subclass(base, declaration):
    class HandoverMotion(base):
        def __init__(self, *args, **kwargs):
            self.handover_ready = False
            super().__init__(*args, **kwargs)
            from .model import load
            source, _ = load(declaration['source_checkpoint'], 'MotionZeroStudent')
            self.handover_teacher = source.to(self.device).eval().requires_grad_(False)
            self.robust_episode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.prefix_remaining = torch.zeros_like(self.robust_episode)
            self.owner_pre = torch.zeros_like(self.robust_episode)
            self.course_owner_snapshot = self.owner_pre.clone()
            self.absolute_control_intervals = self.owner_pre.clone()
            self.teacher_control_intervals = self.owner_pre.clone()
            self.course_absolute_snapshot = self.owner_pre.clone()
            self.course_teacher_snapshot = self.owner_pre.clone()
            self.course_handoff_snapshot = self.owner_pre.bool()
            self.course_handoff_state_snapshot = torch.zeros(self.num_envs,13+2*len(self.robot.joint_names),device=self.device)
            self.robust_control_counts = torch.zeros(4, device=self.device, dtype=torch.long)
            self.robust_handoffs = torch.zeros((), device=self.device, dtype=torch.long)
            self.robust_age_override = None
            self.robust_prefix_override = None
            # The common physical safety path always runs. Only the task clock
            # waits for the first student-controlled transition; no timer reset
            # or physical reset is needed at the handover itself.
            if hasattr(self.rise_hold,'observe'):
                base_observe=self.rise_hold.observe
                def controlled_observe(*args,active,**kw):
                    return base_observe(*args,active=active&self.student_actor_valid,**kw)
                self.rise_hold.observe=controlled_observe
            self.handover_ready = True

        def configure_mcu_cases(self, cases):
            super().configure_mcu_cases(cases)
            self.robust_age_override = torch.full_like(self.robust_episode, 10)
            self.robust_prefix_override = torch.zeros_like(self.robust_episode)
            for i, case in enumerate(cases):
                block = case.get('_execution_uncertainty')
                if block:
                    ids = slice(i * 8, (i + 1) * 8)
                    # Single-env diagnostics use env0; grouped scoring uses8 per case.
                    if len(cases) == 1:
                        ids = slice(None)
                    self.robust_age_override[ids] = block['PD_age_ms']
                    self.robust_prefix_override[ids] = block['prefix_intervals']

        def _reset_idx(self, ids):
            super()._reset_idx(ids)
            if not self.handover_ready:
                return
            ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
            variants = torch.tensor(self.stage_spec['mcu_variants'], device=self.device)[self.command_index[ids]]
            if not self.curriculum_training:
                variants = torch.zeros_like(variants) if self.mcu_override is None else self.mcu_override[ids]
            self.motion_timing.variant[ids] = variants
            # Finish the actual reset first. The counter/parameter draw happens once
            # per real episode, after all inherited reset callbacks have returned.
            self.motion_timing.reset(ids)
            selected = variants == 3
            picked = ids[selected]
            self.robust_episode[picked] += 1
            if declaration['version'] in ('motion31_mcu_height_pd_parallel_v1','motion31_mcu_highspeed_standing2048_v1'):
                maximum=self.stage_spec['PD_feedback_max_ms']
                self.motion_timing.feedback[picked]=sample_pd_age(picked,self.robust_episode[picked],maximum=maximum)
            else:self.motion_timing.feedback[picked] = sample_pd_age(picked, self.robust_episode[picked])
            self.prefix_remaining[ids] = 0
            self.prefix_remaining[picked] = declaration['configuration']['handover_prefix_policy_intervals']
            if declaration['version'] in ('motion31_mcu_height_pd_parallel_v1','motion31_mcu_highspeed_standing2048_v1'):
                self.prefix_remaining[ids]=torch.tensor(self.stage_spec['prefix_intervals'],device=self.device)[self.command_index[ids]]
            if self.robust_age_override is not None:
                self.motion_timing.feedback[picked] = self.robust_age_override[picked]
                self.prefix_remaining[ids] = self.robust_prefix_override[ids]
            self.owner_pre[ids] = 0
            self.absolute_control_intervals[ids] = 0
            self.teacher_control_intervals[ids] = 0
            if len(picked):
                # Legal root velocity perturbation only; all linkage q/dq and
                # closure geometry remain the declared actual standing reset.
                velocity = self.robot.data.root_state_w.torch[picked,7:13].clone()
                velocity.zero_(); velocity[:, 0] = .05; velocity[:, 4] = .10
                self.robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=picked)
            self._motion_reset_epoch = getattr(self, '_motion_reset_epoch', 0) + 1

        def actor_valid_pre(self):
            ordinary = pre_action_valid(self.fsm.mode == int(Mode.INITIALIZING), self.initialization_checked)
            return ordinary & (self.prefix_remaining == 0) if self.handover_ready else ordinary

        def _pre_physics_step(self, actions):
            if not self.handover_ready:
                return super()._pre_physics_step(actions)
            ordinary = pre_action_valid(self.fsm.mode == int(Mode.INITIALIZING), self.initialization_checked)
            teacher = ordinary & (self.prefix_remaining > 0)
            student = ordinary & ~teacher
            with torch.no_grad():
                # Deterministic mean does not mutate Gaussian cache or policy RNG.
                mean = self.handover_teacher.mean(self.student_obs['student_actor'])
            actual = torch.where(teacher[:, None], mean, actions)
            self.owner_pre.copy_(torch.where(student, 3, torch.where(teacher, 2, 0)))
            handoff = student & (self.course_owner_snapshot == 2)
            self.robust_handoffs += handoff.sum()
            self.course_owner_snapshot.copy_(self.owner_pre)
            self.absolute_control_intervals += ordinary.long()
            self.teacher_control_intervals += teacher.long()
            self.course_absolute_snapshot.copy_(self.absolute_control_intervals)
            self.course_teacher_snapshot.copy_(self.teacher_control_intervals)
            self.course_handoff_snapshot.copy_(handoff)
            state=torch.cat((self.robot.data.root_state_w.torch,self.robot.data.joint_pos.torch,self.robot.data.joint_vel.torch),-1)
            self.course_handoff_state_snapshot.copy_(torch.where(handoff[:,None],state,0.))
            super()._pre_physics_step(actual)
            self.student_actor_valid.copy_(student)
            self.rise_hold.motion.copy_(student)
            self.prefix_remaining -= teacher.long()
            self.robust_control_counts += torch.stack(((~ordinary).sum(), teacher.sum(), student.sum(), handoff.sum()))

        def _get_dones(self):
            result = super()._get_dones()
            if self.handover_ready and getattr(self, 'motion_quality_ready', False):
                from .motion_posture_design import apply_runtime_reward
                apply_runtime_reward(self.student_reward_result, self.standing_snapshot)
            return result

        def robustness_summary(self):
            return dict(version=declaration['version'],
                counts_initial_teacher_student_handoffs=self.robust_control_counts.cpu().tolist(),
                cumulative_handoffs=int(self.robust_handoffs),
                PD_age_min_ms=int(self.motion_timing.feedback.min()), PD_age_max_ms=int(self.motion_timing.feedback.max()),
                actual_parameter_in_critic312=True, frozen_source_sha256=declaration['source_sha256'],
                teacher_controls_only_prefix=True, reset_at_handover=False,
                physical_target_previous='unchanged local5ms adopted target', hypothesis_not_real_alignment=True)

    return HandoverMotion
