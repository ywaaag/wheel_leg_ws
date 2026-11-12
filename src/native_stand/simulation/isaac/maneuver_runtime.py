"""New-instance adapters for the V4 height/banking task, no frozen-file edits."""

from __future__ import annotations

import torch

from native_stand.tasks.motion.maneuver_course import CRITIC_VERSION, DIM
from native_stand.tasks.motion.maneuver_reward import apply, quality_inputs
from native_stand.tasks.motion.maneuver_targets import mode_masks, roll_target


def program(age, target_xy, goals):
    changing = (goals - 0.34).abs() > 1e-6
    h = 0.34 + (goals - 0.34) * ((age - 1.0) / 2.0).clamp(0, 1)
    vfactor = torch.where(
        changing, ((age - 6.0) / 2.0).clamp(0, 1), ((age - 0.5) / 1.0).clamp(0, 1)
    )
    mixed = (target_xy[:, 0] != 0) & (target_xy[:, 1] != 0)
    wfactor = torch.where(
        mixed,
        torch.where(changing, ((age - 11.0) / 2.0).clamp(0, 1), ((age - 5.0) / 2.0).clamp(0, 1)),
        vfactor,
    )
    cmd = torch.cat(
        (target_xy[:, 0:1] * vfactor[:, None], target_xy[:, 1:2] * wfactor[:, None], h[:, None]), -1
    )
    rate = torch.where((age > 1) & (age < 3), (goals - 0.34) / 2.0, torch.zeros_like(goals))
    return cmd, rate


class QualityAdapter:
    def __init__(self, original, env):
        self.original = original
        self.env = env

    def __getattr__(self, name):
        return getattr(self.original, name)

    def observe(self, **kw):
        actual = kw["posture"]
        p, w = quality_inputs(
            actual,
            kw["wheel_offset"],
            self.env.motion_action_command,
            is_flat=getattr(self.env, "motion_flat_terrain", None),
        )
        measured_height = kw["height"]
        if getattr(self.env, "extended_height_quality", False):
            measured_height = measured_height + 0.34 - self.env.motion_action_command[:, 2]
        decision = self.original.observe(
            **{**kw, "posture": p, "wheel_offset": w, "height": measured_height}
        )
        # Banking never removes the original absolute attitude safety bound.
        hard = (actual[:, 1].abs() > 0.50) & kw["controlled"] & kw["valid"]
        decision["causes"][:, 1] |= hard
        decision["terminated"] |= hard
        self.original.sent |= hard
        return decision


def append_critic(env, value):
    if value.shape[-1] != 863:
        raise RuntimeError("Expected actual V4 causal863 prefix")
    value = value.clone()
    cmd = env.command
    ref = roll_target(cmd)
    from ground_height import from_env

    height_error = from_env(env) - cmd[:, 2]
    # Named original fields: base511 height is raw metres; reward kinematics
    # uses .04m and .05rad. Do not retain contradictory fixed .34 references.
    value[:, env.maneuver_critic_columns["height"]] = height_error
    k = env.maneuver_critic_columns["kinematics"]
    value[:, k] = height_error / 0.04
    value[:, k + 2] = (env.standing_snapshot[:, 1] - ref) / 0.05
    mode = mode_masks(cmd, is_flat=getattr(env, "motion_flat_terrain", None))
    age = env.rise_hold.episode.steps.float() * 0.01
    quiet, ramp = (age >= 3) & (age < 6), (age >= 6) & (age < 8)
    if (
        getattr(env, "maneuver_command_program_version", None)
        == "motion48_height020_rate_limited_schedule_v3"
    ):
        from native_stand.tasks.motion.maneuver_program import extended_phase_flags

        quiet, ramp = extended_phase_flags(age, env.maneuver_goal_height)
    extra = torch.stack(
        (
            cmd[:, 2] / 0.5,
            env.maneuver_goal_height / 0.5,
            env.maneuver_height_rate / 0.1,
            (env.maneuver_height_rate != 0).to(cmd),
            quiet.to(cmd),
            ramp.to(cmd),
            ref / 0.30,
            mode["symmetry_required"].to(cmd),
            mode["standing"].to(cmd),
            mode["straight"].to(cmd),
            mode["spin"].to(cmd),
            mode["travel_turn"].to(cmd),
        ),
        -1,
    )
    result = torch.cat((value, extra), -1)
    if result.shape[-1] != DIM:
        raise RuntimeError("Maneuver Critic fields do not match declaration")
    return result


def environment_class(base, config, *, reward_config=None):
    class ManeuverEnvironment(base):
        def _reset_idx(self, ids):
            result = super()._reset_idx(ids)
            if getattr(self, "maneuver_ready", False):
                chosen = torch.arange(self.num_envs, device=self.device) if ids is None else ids
                override = getattr(self, "maneuver_goal_override", None)
                self.maneuver_goal_height[chosen] = (
                    self.maneuver_goal_table[self.command_index[chosen]]
                    if override is None
                    else override[chosen]
                )
            return result

        def _get_observations(self):
            result = super()._get_observations()
            if getattr(self, "maneuver_ready", False) and "student_critic" in result:
                result = dict(result)
                result["student_critic"] = append_critic(self, result["student_critic"])
                self.student_obs = result
            return result

        def _get_dones(self):
            result = super()._get_dones()
            if getattr(self, "maneuver_ready", False):
                if (
                    reward_config is not None
                    and reward_config.get("version") != "motion48_height_bank_mode_reward_v1"
                ):
                    from native_stand.tasks.motion.maneuver_retention_reward import (
                        apply as retained_apply,
                    )

                    retained_apply(self, config, reward_config)
                else:
                    apply(self, config)
                self.onpolicy_episodes.observe_final(result)
            return result

    return ManeuverEnvironment


def attach(env, config, *, reward_config=None, command_program_version=None):
    from critic_candidate import FIELDS as MEMORY_FIELDS
    from critic_tensor import FIELDS as BASE_FIELDS

    def offset(fields, name):
        return sum(w for n, w in fields[: next(i for i, x in enumerate(fields) if x[0] == name)])

    env.maneuver_critic_columns = dict(
        height=offset(BASE_FIELDS, "height_error_m"),
        kinematics=511 + offset(MEMORY_FIELDS, "salient_reward_kinematics"),
    )
    env.maneuver_goal_table = torch.tensor(env.stage_spec["goal_heights_m"], device=env.device)
    env.maneuver_goal_height = env.maneuver_goal_table[env.command_index].clone()
    env.maneuver_height_rate = torch.zeros_like(env.maneuver_goal_height)
    env.maneuver_swing_difference = torch.zeros_like(env.maneuver_goal_height)
    env.quality_monitor = QualityAdapter(env.quality_monitor, env)
    command_function = program
    if command_program_version is not None:
        from native_stand.tasks.motion.maneuver_program import (
            EXTENDED_VERSION,
            VERSION,
            extended_tensor_program,
            tensor_program,
        )

        if command_program_version not in (VERSION, EXTENDED_VERSION):
            raise ValueError("Unknown maneuver timing version")
        command_function = (
            extended_tensor_program
            if command_program_version == EXTENDED_VERSION
            else tensor_program
        )
    env.maneuver_command_program_version = command_program_version

    def command(target, age, training):
        cmd, rate = command_function(age, target, env.maneuver_goal_height)
        env.maneuver_height_rate.copy_(rate)
        env.height_adapter.target.copy_(cmd[:, 2])
        env.height_adapter.rate.copy_(rate)
        return cmd

    env.height_adapter.command = command
    env.maneuver_ready = True


def readback(env, obs):
    from ground_height import from_env

    value = obs["student_critic"]
    cols = env.maneuver_critic_columns
    checks = dict(
        width=value.shape == (env.num_envs, DIM),
        finite=bool(torch.isfinite(value).all()),
        actual_height_error=bool(
            torch.allclose(
                value[:, cols["height"]], from_env(env) - env.command[:, 2], atol=1e-7, rtol=0
            )
        ),
        actual_command_height=bool(
            torch.allclose(value[:, 863], env.command[:, 2] / 0.5, atol=1e-7, rtol=0)
        ),
        no_teacher_prefix=not bool(env.r4_prefix_remaining.any()),
        actual_caps=bool(
            torch.allclose(
                env.motion_timing.dr.caps[:, [2, 5]], value.new_full((env.num_envs, 2), 4.8)
            )
        ),
    )
    if not all(checks.values()):
        raise RuntimeError("Maneuver actual consumer readback: " + str(checks))
    return dict(
        status="PASSED",
        version=CRITIC_VERSION,
        checks=checks,
        Actor_dim=31,
        physical_state_resume=False,
        height_first_no_mid_episode_reset=True,
        real_dq_and_PD_D=True,
        body_roll_reference="soft only; absolute .5rad safety retained",
    )
