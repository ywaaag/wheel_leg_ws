"""Engineering evidence from the public Recovery adapter's real transitions."""

from __future__ import annotations

import torch

from native_stand.artifacts.io import write_json


class TrainingReadback:
    """Observe the first execution block without changing control or optimization."""

    def __init__(self, context, env, actor):
        self.context = context
        self.start_update = context.updates_used
        self.initial = {k: v.detach().clone() for k, v in actor.named_parameters()}
        self.steps = 0
        self.nonzero_reward = False
        self.student = False
        self.prefix = False
        self.done_count = 0
        self.actor_updates = 0
        self.critic_updates = 0

    def step(self, outcome):
        if self.context.updates_used - self.start_update >= 12:
            return
        self.steps += 1
        if not bool(torch.isfinite(outcome["rewards"]).all()):
            raise RuntimeError("Nonfinite real Recovery reward")
        self.nonzero_reward |= bool((outcome["rewards"].abs() > 0).any())
        self.done_count += int(outcome["dones"].sum())
        tr = outcome["extras"].get("transition")
        if tr:
            self.student |= bool(tr["student_mask"].any())
            self.prefix |= bool((~tr["student_mask"]).any())
            if not torch.allclose(tr["motor"], tr["actual_actuator"], atol=1e-4, rtol=0):
                raise RuntimeError("Real actuator/motor ownership differs")

    def before_update(self, ppo, *, allow_actor):
        self.allow_actor = allow_actor
        self.before_parameters = (
            {k: p.detach().clone() for k, p in ppo.actor.named_parameters()}
            if self.context.updates_used - self.start_update < 12
            else None
        )

    def after_update(self, ppo):
        c = self.context
        n = c.updates_used
        segment_updates = n - self.start_update
        if segment_updates > 12:
            return
        changed_this_update = any(
            not torch.equal(p.detach(), self.before_parameters[k])
            for k, p in ppo.actor.named_parameters()
        )
        if not self.allow_actor and changed_this_update:
            raise RuntimeError("Actor changed during a declared Critic-only update")
        actor_steps = int(ppo.mask_summary.get("actor_optimizer_steps", 0))
        critic_steps = int(ppo.mask_summary.get("critic_optimizer_steps", 0))
        if not self.allow_actor and actor_steps:
            raise RuntimeError("Actor optimizer ran during a declared Critic-only update")
        self.actor_updates += int(actor_steps > 0)
        self.critic_updates += int(critic_steps > 0)
        if c.job.diagnostics.get("recovery_audit_metrics"):
            write_json(
                c.output / f"optimization_audit_{n:05d}.json",
                dict(
                    update=n,
                    allow_actor=self.allow_actor,
                    Actor_changed_this_update=changed_this_update,
                    actor_optimizer_steps=actor_steps,
                    critic_optimizer_steps=critic_steps,
                    scope="first12 optimizer/ownership engineering evidence; no gradient decomposition",
                    training_diagnostic_not_acceptance=True,
                ),
            )
        if segment_updates != 12:
            return
        changed = any(
            not torch.equal(p.detach(), self.initial[k]) for k, p in ppo.actor.named_parameters()
        )
        passed = (
            self.student
            and not self.prefix
            and self.nonzero_reward
            and self.done_count > 0
            and self.actor_updates > 0
            and self.critic_updates == 12
            and changed
        )
        write_json(
            c.output / "engineering_smoke.json",
            dict(
                actual_updates=n,
                segment_start_update=self.start_update,
                segment_updates=segment_updates,
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
            ),
        )
        if not passed:
            raise RuntimeError("Recovery first12 engineering evidence incomplete")
