"""Planned877 plus optional actual-contact881, using the same terminal encoder."""

from __future__ import annotations

import torch

from native_stand.tasks.motion.planned_value import DIM, VERSION, append

from . import maneuver_runtime


def environment_class(base):
    class PlannedValueEnvironment(base):
        def _get_observations(self):
            result = super()._get_observations()
            if getattr(self, "planned_value_ready", False) and "student_critic" in result:
                result = dict(result)
                result["student_critic"] = append(result["student_critic"], self.target_xy)
                contact = getattr(
                    getattr(getattr(self, "motion_timing", None), "dr", None), "contact", None
                )
                if contact is not None:
                    from native_stand.tasks.motion.contact_randomization import append_context

                    result["student_critic"] = append_context(
                        result["student_critic"], contact.actual, contact.config
                    )
                self.student_obs = result
            return result

        def _get_dones(self):
            result = super()._get_dones()
            if getattr(self, "planned_value_ready", False):
                self.student_terminal_critic = None
                terminated, truncated = result
                # The ordinary post-step observation will build the same877
                # once. Only ended rows need an additional PRE-RESET snapshot.
                if bool((terminated | truncated).any()):
                    terminal = self._get_observations()["student_critic"]
                    self.student_terminal_critic = terminal.detach().clone()
                capture = getattr(self, "collection_snapshot", None)
                if capture is not None:
                    capture()
            return result

    return PlannedValueEnvironment


def attach(env, config, *, reward_config=None, command_program_version=None):
    maneuver_runtime.attach(
        env, config, reward_config=reward_config, command_program_version=command_program_version
    )
    env.planned_value_ready = True


def readback(env, obs):
    value = obs["student_critic"]
    contact = getattr(env.motion_timing.dr, "contact", None)
    dim = 881 if contact is not None else DIM
    if value.shape != (env.num_envs, dim):
        raise RuntimeError(f"Declared{dim} Critic did not reach actual observations")
    base = dict(obs)
    base["student_critic"] = value[:, :875]
    result = maneuver_runtime.readback(env, base)
    check = bool(torch.allclose(value[:, 875:877], env.target_xy / 4.0, atol=1e-7, rtol=0))
    if not check:
        raise RuntimeError("Planned command endpoints missing from actual Critic")
    result.update(
        version=VERSION,
        Critic_dim=dim,
        planned_command_readback=True,
        terminal_bootstrap_contract=f"same{dim} pre-reset successor construction",
        Actor_dim=31,
        planned_context_is_future_state=False,
    )
    if contact is not None:
        from native_stand.tasks.motion.contact_randomization import CRITIC_VERSION, append_context

        expected = append_context(value[:, :877], contact.actual, contact.config)
        if not torch.equal(expected[:, 877:], value[:, 877:]):
            raise RuntimeError("Actual contact values missing from ordinary/terminal Critic")
        result.update(version=CRITIC_VERSION, wheel_contact=contact.last_readback)
    if getattr(env, "maneuver_command_program_version", None) is not None:
        from native_stand.tasks.motion.maneuver_program import (
            EXTENDED_VERSION,
            extended_tensor_program,
            tensor_program,
        )
        from native_stand.tasks.motion.maneuver_program import VERSION as PROGRAM

        age = env.rise_hold.episode.steps.float() * 0.01
        fn = (
            extended_tensor_program
            if env.maneuver_command_program_version == EXTENDED_VERSION
            else tensor_program
        )
        expected, rate = fn(age, env.target_xy, env.maneuver_goal_height)
        if env.maneuver_command_program_version not in (PROGRAM, EXTENDED_VERSION):
            raise RuntimeError("Shared command schedule did not reach actual environment")
        finite = bool(torch.isfinite(expected).all() and torch.isfinite(rate).all())
        actual = bool(torch.allclose(env.motion_observation_command, expected, atol=1e-7, rtol=0))
        if not finite or not actual:
            raise RuntimeError("Actual Actor command does not follow the shared timing schedule")
        result.update(
            command_program_version=env.maneuver_command_program_version,
            current_schedule_finite=finite,
            actual_observed_command_verified=actual,
        )
    return result
