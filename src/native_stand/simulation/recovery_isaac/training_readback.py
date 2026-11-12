"""Real rollout engineering evidence; never changes actions, rewards or gates."""

from __future__ import annotations

import copy

import torch

from native_stand.artifacts.io import write_json


class TrainingReadback:
    def __init__(self, context, env, actor):
        self.context = context
        self.env = env
        self.initial = {k: v.detach().clone() for k, v in actor.named_parameters()}
        self.steps = 0
        self.nonzero_reward = False
        self.student = False
        self.prefix = False
        self.done_count = 0
        self.old_prior_rows_seen = 0
        self.handover_events = 0
        self.Swing_recovery_samples = 0
        self.Motion_confirmation_samples = 0
        self.actor_updates = 0
        self.critic_updates = 0
        self.behavior = None
        self.candidate_window = {
            k: dict(complete=0, qualified_pass=0) for k in ("standing", "rise")
        }

    def step(self, outcome):
        self.steps += 1
        if not bool(torch.isfinite(outcome["rewards"]).all()):
            raise RuntimeError("Nonfinite real Recovery reward")
        self.nonzero_reward |= bool((outcome["rewards"].abs() > 0).any())
        done = int(outcome["dones"].sum())
        self.done_count += done
        completed = getattr(getattr(self.env, "env", None), "completed", [])
        for row in completed[-done:] if done else []:
            objective = {"standing_review": "standing", "full_Swing": "rise"}.get(row.get("group"))
            if objective and row.get("nominal_anchor") is True:
                target = self.candidate_window[objective]
                target["complete"] += 1
                target["qualified_pass"] += int(
                    row["success"]
                    and (row.get("Motion_confirmed") is True or row.get("physical_quality") is True)
                )
        tr = outcome["extras"].get("transition")
        if tr:
            self.handover_events += int(tr.get("handover_event", torch.zeros(1)).sum())
            if "initial_Swing" in tr:
                self.Swing_recovery_samples += int((tr["initial_Swing"] & tr["student_mask"]).sum())
            self.Motion_confirmation_samples += int(
                (
                    tr.get("Motion_handover_active", torch.zeros_like(tr["student_mask"]))
                    & ~tr["student_mask"]
                ).sum()
            )
            if "physical_state" in tr:
                from native_stand.learning.ppo.recovery_diagnostics import BehaviorWindow

                if self.behavior is None:
                    self.behavior = BehaviorWindow(
                        len(tr["physical_state"]), tr["physical_state"].device
                    )
                self.behavior.add(tr)
            self.student |= bool(tr["student_mask"].any())
            self.prefix |= bool((~tr["student_mask"]).any())
            if not torch.allclose(tr["motor"], tr["actual_actuator"], atol=1e-4, rtol=0):
                raise RuntimeError("Real actuator/motor ownership differs")
        return outcome

    def before_update(self, ppo):
        n = self.context.updates_used + 1
        self.audit_active = bool(self.context.job.diagnostics.get("recovery_audit_metrics")) and (
            n <= 12 or n % 100 == 0
        )
        ppo.recovery_objective_diagnostics = self.audit_active
        ppo.retention_readback_done = False
        ppo.recovery_objective_audit = None
        self.rollout_audit = None
        if self.audit_active:
            from native_stand.learning.ppo.recovery_diagnostics import rollout

            self.rollout_audit = rollout(ppo)

    def after_update(self, ppo):
        c = self.context
        n = c.updates_used
        self.old_prior_rows_seen = max(
            self.old_prior_rows_seen,
            getattr(ppo, "action_prior_readback", {}).get("total_queried_rows", 0),
        )
        if getattr(self, "audit_active", False):
            write_json(
                c.output / f"optimization_audit_{n:05d}.json",
                dict(
                    update=n,
                    critic_and_sampling=self.rollout_audit,
                    objective_gradient=getattr(ppo, "recovery_objective_audit", None),
                    mask=ppo.mask_summary,
                    action_prior={
                        k: float(v) if torch.is_tensor(v) else v
                        for k, v in getattr(ppo, "action_prior_readback", {}).items()
                    },
                    training_diagnostic_not_acceptance=True,
                ),
            )
        changed = False
        if n <= 5 or n == 12:
            changed = any(
                not torch.equal(p.detach(), self.initial[k])
                for k, p in ppo.actor.named_parameters()
            )
        if n <= 5 and changed:
            raise RuntimeError("Actor changed during first five Critic-only updates")
        self.actor_updates += int(ppo.mask_summary.get("actor_optimizer_steps", 0) > 0)
        self.critic_updates += int(ppo.mask_summary.get("critic_optimizer_steps", 0) > 0)
        if n == 12:
            passed = (
                self.student
                and self.nonzero_reward
                and self.actor_updates > 0
                and self.critic_updates == 12
                and changed
            )
            if getattr(getattr(self.env, "env", None), "handover_mode", False):
                passed = (
                    passed
                    and self.Swing_recovery_samples > 0
                    and self.Motion_confirmation_samples > 0
                    and self.handover_events > 0
                )
                if c.job.assets.get("old_Recovery_action_prior"):
                    passed = passed and self.old_prior_rows_seen > 0
            evidence = dict(
                old_prior_rows_seen=self.old_prior_rows_seen,
                Swing_recovery_samples=self.Swing_recovery_samples,
                Motion_confirmation_samples=self.Motion_confirmation_samples,
                handover_events=self.handover_events,
                actual_updates=n,
                steps=self.steps,
                nonzero_reward=self.nonzero_reward,
                student_control=self.student,
                prefix_control=self.prefix,
                natural_done_count=self.done_count,
                Actor_changed_after_warmup=changed,
                Actor_updates=self.actor_updates,
                Critic_updates=self.critic_updates,
                engineering_smoke_passed=passed,
                full_task_accepted=False,
            )
            write_json(c.output / "engineering_smoke.json", evidence)
            if not passed:
                raise RuntimeError("Recovery first12 engineering evidence incomplete")
            c.verify_wiring(n)
        if n % 100 == 0:
            e = self.env.env
            rows = [copy.deepcopy(course.counts) for course in e.courses]
            write_json(
                c.output / f"learning_statistics_{n:05d}.json",
                dict(
                    update=n,
                    course_counts=rows,
                    course_stage=[x.stage for x in e.courses],
                    Swing_recovery_samples=self.Swing_recovery_samples,
                    Motion_confirmation_samples=self.Motion_confirmation_samples,
                    handover_events=self.handover_events,
                    behavior=self.behavior.summary() if self.behavior is not None else None,
                    real_physics_steps=e.actual_physics_steps,
                    completed_recent=e.completed[-1024:],
                    full_task_accepted=False,
                ),
            )
            self.behavior = None
            # These are on-policy training candidates, never independent best
            # or full-task certificates. No extra optimizer or physics occurs.
            reference = c.latest_reference()
            if reference is not None:
                record = (
                    copy.deepcopy(c.training_candidates)
                    if c.training_candidates
                    else dict(independent_certificate=False, candidates_by_objective={})
                )
                pools = record["candidates_by_objective"]
                for objective, window in self.candidate_window.items():
                    episodes = window["complete"]
                    passed = window["qualified_pass"]
                    if episodes < 32:
                        continue
                    score = passed / episodes
                    old = pools.get(objective, {}).get("nominal_fixed")
                    if old is None or score > old["training_pass_fraction"]:
                        pools[objective] = {
                            "nominal_fixed": dict(
                                reference,
                                training_pass_fraction=score,
                                complete_student_episodes=episodes,
                                cohort="fixed nominal "
                                + objective
                                + " initial pose, no pulse, same source/Plant/std",
                                independent_certificate=False,
                            )
                        }
                standing = pools.get("standing", {}).get("nominal_fixed")
                rise = pools.get("rise", {}).get("nominal_fixed")
                record.update(
                    best_standing_candidate=standing,
                    best_candidate=rise or standing,
                    selection_scope="fixed nominal cohort only, >=32 complete episodes in current100-update window, success AND (Motion confirmation in handover task, otherwise physical quality); no cross-DR/course mixture ranking; NOT independent validation",
                )
                c.publish_training_candidates(record)
            self.candidate_window = {
                k: dict(complete=0, qualified_pass=0) for k in ("standing", "rise")
            }
