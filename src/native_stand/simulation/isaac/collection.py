"""Motion sampling view over the production environment, without PPO state."""

from __future__ import annotations

import torch


class MotionCollectionEnv:
    """One owned environment; transition metadata belongs to the pre-reset row."""

    def __init__(self, raw, app, job, observation, *, convert, output_contract=None, template=None):
        self.env, self.app, self.job = raw, app, job
        self.num_envs = raw.num_envs
        self.device = raw.device
        self._convert = convert
        self.output_contract, self.template = output_contract, template
        self._obs = observation
        self._closed = False
        self._faulted = False
        self.cumulative_physics_steps = 0
        self._diagnostic = False
        self._snapshot = None
        self._clock = raw.sim.get_physics_step_count
        actor = observation["student_actor"]
        if actor.shape != (self.num_envs, 31) or actor.device != torch.device(self.device):
            raise ValueError("Motion collection requires actual31D rows on the environment device")
        if self.num_envs != job.runtime.envs:
            raise ValueError("Motion collection resources differ from actual observation rows")
        if observation["student_critic"].shape != (self.num_envs, job.model.critic_input):
            raise ValueError("Motion collection Critic differs from the production contract")
        if not hasattr(raw, "r4_owner"):
            raise ValueError("Motion collection requires the actual Handover owner")
        raw.collection_control_owner = torch.full_like(raw.r4_owner, -1)

    def observations(self):
        # No second observation build/noise draw between action and its label.
        return self._obs

    def conditions(self):
        e = self.env
        value = self._obs["student_actor"]
        indices = e.command_index
        stage = e.stage_spec
        # Conditions describe the input row, before step/reset changes cohorts.
        conditions = dict(
            vx=value[:, 22] * 4,
            yaw=value[:, 23] * 4,
            height=value[:, 24] * 0.5,
            target_vx=e.target_xy[:, 0],
            target_yaw=e.target_xy[:, 1],
            cohort=indices,
            profile=e.identified_dr.profiles,
            DR_amplitude=e.identified_dr.amplitude[:, 0],
            modal_mode=e.transient_events.mode,
            modal_active=e.transient_events.active,
            episode_age_s=e.rise_hold.episode.steps.float() * self.job.control.policy_dt_s,
        )
        if "goal_heights_m" in stage:
            conditions["target_height"] = value.new_tensor(stage["goal_heights_m"])[indices]
        else:
            conditions["target_height"] = value.new_tensor(stage["height_targets_m"])[indices]
        if hasattr(e, "motion_flat_terrain"):
            conditions["flat_terrain"] = e.motion_flat_terrain
        for name, tensor in conditions.items():
            if tensor.shape != (self.num_envs,):
                raise ValueError("Motion routing condition shape differs: " + name)
        return {k: v.detach().clone() for k, v in conditions.items()}

    def step(self, action):
        if self._closed or self._faulted:
            raise RuntimeError("Closed/faulted Motion collection cannot continue")
        e = self.env
        conditions = self.conditions()
        before = self._clock()
        e.collection_control_owner.fill_(-1)
        self._snapshot = None
        try:
            raw, reward, terminated, truncated, extras = e.step(action)
            self._obs = self._convert(e, raw)
            owner = e.collection_control_owner
            if bool(((owner != 0) & (owner != 2) & (owner != 3)).any()):
                raise ValueError("Actual Motion transition owner was not captured")
            # Public collection ABI: 0 Student, 1 Teacher, 2 initialization.
            owner = torch.where(owner == 3, 0, torch.where(owner == 2, 1, 2))
            conditions["control_owner"] = owner.detach().clone()
            if self._diagnostic and self._snapshot is None:
                raise ValueError("Motion diagnostic omitted the pre-reset physical snapshot")
        except BaseException as failure:
            self._faulted = True
            failure.actual_physics_steps = [self._clock() - before] * self.num_envs
            raise
        finally:
            elapsed = self._clock() - before
            if elapsed < 0:
                self._faulted = True
                raise RuntimeError("Motion physics counter regressed")
            self.cumulative_physics_steps += elapsed * self.num_envs
        return dict(
            observations=self._obs,
            rewards=reward,
            terminated=terminated,
            truncated=truncated,
            extras={
                **extras,
                "actual_physics_steps": [elapsed] * self.num_envs,
                "control_owner": owner,
                "collection_conditions": conditions,
                "terminal_critic": e.student_terminal_critic,
                "physics_count_semantics": "engine step calls; faulting last-step completion UNKNOWN",
                **({"diagnostic": self._snapshot} if self._diagnostic else {}),
            },
        )

    def configure_diagnostic(self, protocol):
        from .collection_diagnostic import configure

        return configure(self, protocol)

    def capture_diagnostic(self):
        e = self.env
        t = e.motion_timing
        d = e.robot.data
        self._snapshot = {
            "q": d.joint_pos.torch[:, e.ids].detach().clone(),
            "dq": d.joint_vel.torch[:, e.ids].detach().clone(),
            "motor": t.base.motor_actual.detach().clone(),
            "NET": t.last_external.detach().clone(),
            "state": d.root_state_w.torch.detach().clone(),
            "command": e.command.detach().clone(),
            "adopted": t.base.adopted.detach().clone(),
            "PD_q_dq": t.base.last_pd_sensors.detach().clone(),
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.env.close()
        finally:
            if self.app is not None:
                close_application(self.app, self.job, failed=self._faulted)
                self.app = None


def close_application(app, job, *, failed):
    if job.assets.get("application_lifecycle", {}).get("owner") == "worker_atexit":
        import atexit

        atexit.register(lambda: app.close(exit_code=1 if failed else 0))
    else:
        app.close()


def make_environment(job, *, resources, session_factory, environment_factory):
    """Resources come from collection.runtime, never evaluation.envs."""
    if resources != job.runtime:
        raise ValueError("Motion collection resources differ from the explicit runtime request")
    context = app = raw = None
    try:
        context, app = session_factory(job, train=True)
        raw, obs = environment_factory(
            job,
            context["spec"],
            context["config"],
            context["prefix"],
            context["source"],
            context["environment"],
            context["observations"],
            train=True,
        )
        return MotionCollectionEnv(
            raw,
            app,
            job,
            obs,
            convert=context["observations"],
            output_contract=context["source"]["output_contract"],
            template=context.get("template"),
        )
    except BaseException as failure:
        try:
            if raw is not None:
                raw.close()
        except Exception as cleanup:
            failure.add_note("Motion collection cleanup failed: " + repr(cleanup))
        finally:
            if app is not None:
                close_application(app, job, failed=True)
        raise
