"""Leg-led Recovery math: causal sequence, real 1ms motion costs, no actuator clamp."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

VERSION = "recovery_leg_driven_low_wheel_v1"
COURSE_VERSION = "recovery_leg_driven_course_v1"
CRITIC_VERSION = "recovery_leg_driven_causal942_v1"
CRITIC_WIDTH = 942
FRACTIONS = ((0.30, 0.25, 0.25, 0.20), (0.70, 0.10, 0.10, 0.10), (0.85, 0.05, 0.05, 0.05))
WHEEL_SCALE = (4.0, 2.0, 1.0, 0.5, 0.5, 0.5)
WHEEL_PEAK = (6.0, 4.0, 3.0, 2.0, 2.0, 2.0)
GROUPS = ("full_Swing", "shortened", "rotated", "lifted")
COST_WEIGHTS = (0.28, 0.12, 0.20, 0.08, 0.05, 0.04, 0.05, 0.10, 0.03, 0.05)


def anti_prone_config(value=None):
    """Validate the optional reward recipe at the external configuration boundary."""
    if value is None:
        return None
    expected = {"version", "grace_s", "ramp_s", "max_rate", "height_m"}
    if set(value) != expected or value["version"] != "recovery_anti_prone_v1":
        raise ValueError("Explicit anti-prone reward recipe required")
    result = dict(value)
    for key in expected - {"version"}:
        if isinstance(value[key], bool):
            raise ValueError("Finite positive anti-prone parameters required")
        result[key] = float(value[key])
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError("Finite positive anti-prone parameters required")
    if result["height_m"] >= 0.32:
        raise ValueError("Anti-prone height must stay below standing entry")
    return result


def anti_prone_density(recipe, *, phase, progress, stage_age, height, gate):
    """No motion credit: joint twitching and high wheel speed cannot cancel cost."""
    if recipe is None:
        return torch.zeros_like(height)
    ramp = ((stage_age - recipe["grace_s"]) / recipe["ramp_s"]).clamp(0, 1)
    low = ((recipe["height_m"] - height) / 0.04).clamp(0, 1)
    active = (phase < 2) & ~gate
    # Near-target states retain a cost until they actually meet the phase gate.
    return -recipe["max_rate"] * ramp * low * (1 - 0.5 * progress.clamp(0, 1)) * active


def progression_config(value=None):
    """Optional revision; old frozen rewards keep their original behavior."""
    if value is None:
        return None
    keys = {
        "version",
        "remaining_rate",
        "completion_bonus",
        "failure_tail_rate",
        "terminate_invalid_sequence",
    }
    if set(value) != keys or value["version"] != "recovery_stage_progress_v1":
        raise ValueError("Explicit stage progression recipe required")
    result = copy.deepcopy(value)
    for key in ("remaining_rate", "failure_tail_rate"):
        if isinstance(value[key], bool) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError("Positive finite progression rates required")
    bonus = value["completion_bonus"]
    if len(bonus) != 3 or any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in bonus):
        raise ValueError("Three positive phase completion bonuses required")
    if value["terminate_invalid_sequence"] is not True:
        raise ValueError("Progression recipe ends irreversibly invalid sequences")
    return result


def course_fractions(stages=()):
    if not stages:
        return FRACTIONS
    if len(stages) != 3:
        raise ValueError("Three explicit leg-driven sampling stages required")
    fractions = []
    for row in stages:
        if set(row) != {"groups", "fractions"} or tuple(row["groups"]) != GROUPS:
            raise ValueError("Explicit ordered leg-driven groups required")
        values = tuple(row["fractions"])
        if (
            len(values) != 4
            or any(not math.isfinite(v) or v <= 0 for v in values)
            or abs(sum(values) - 1) > 1e-8
        ):
            raise ValueError("Positive leg-driven group fractions must sum to one")
        fractions.append(values)
    return tuple(fractions)


def angle_difference(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


class Course:
    """Per-env sampling debt. Promotion is global, from complete training windows."""

    def __init__(self, seed, *, fractions=FRACTIONS):
        self.fractions = fractions
        self.seed = seed
        self.stage = 0
        self.episodes = 0
        self.counts = {g: dict(student_seconds=0.0, episodes=0, passed=0) for g in GROUPS}

    def choose(self):
        fractions = getattr(self, "fractions", FRACTIONS)[min(self.stage, 2)]
        total = sum(r["student_seconds"] for r in self.counts.values())
        index = (
            0
            if self.episodes == 0
            else int(
                np.argmax(
                    [
                        fractions[i] * (total + 1) - self.counts[g]["student_seconds"]
                        for i, g in enumerate(GROUPS)
                    ]
                )
            )
        )
        nominal = self.stage < 4 or self.episodes % 2 == 0
        return dict(
            group=GROUPS[index],
            standing=index == 3,
            prefix_s=0.0,
            pulse=False,
            DR_strength=0.0 if nominal else (0.1 if self.stage == 4 else 0.25),
            nominal_anchor=nominal,
            height_m=0.34,
            height_dynamic=False,
            pose_name=("Swing", "shortened", "rotated", "standing_h0.34")[index],
            initial_phase=index,
            course_level=self.stage,
        )

    def record(self, group, *, student_seconds, teacher_seconds, passed, **kwargs):
        if teacher_seconds != 0:
            raise ValueError("Leg-led Recovery never borrows Teacher-controlled time")
        row = self.counts[group]
        row["student_seconds"] += student_seconds
        row["episodes"] += 1
        row["passed"] += int(passed)
        self.episodes += 1


class Lifecycle:
    """All state used by reward/certification is captured, including substep history."""

    def __init__(self, n, device, geometry, gamma, *, anti_prone=None, progression=None):
        self.anti_prone = anti_prone_config(anti_prone)
        self.progression = progression_config(progression)
        if self.progression is not None:
            added = self.anti_prone["max_rate"] if self.anti_prone is not None else 0.0
            if (
                self.progression["failure_tail_rate"]
                < 4 + self.progression["remaining_rate"] + added
            ):
                raise ValueError("Failure tail must cover all maximum running costs")
        self.n = n
        self.device = device
        self.max_s = 6.0
        self.gamma = gamma
        self.geometry = copy.deepcopy(geometry)
        z = torch.zeros(n, device=device)
        for k in (
            "h0",
            "prefix_s",
            "dwell",
            "hold",
            "last_phi",
            "elapsed",
            "total_reward",
            "reward",
            "tilt_peak",
            "level_time",
            "duration",
            "tilt_bad_time",
            "fast_time",
            "last_vz",
            "stage_start",
            "step_cost",
            "step_count",
            "progress",
        ):
            setattr(self, k, z.clone())
        self.entry = z - 1
        self.success = z.bool()
        self.phase = z.long()
        self.initial_phase = z.long()
        self.course_level = z.long()
        self.sequence_valid = torch.ones(n, device=device, dtype=torch.bool)
        self.valid_target = z.bool()
        self.last_target = z[:, None].expand(-1, 6).clone()
        self.last_delta = self.last_target.clone()
        self.initial_geometry = z[:, None].expand(-1, 4).clone()
        self.last_geometry = self.initial_geometry.clone()
        self.geometry_speed = self.initial_geometry.clone()
        self.wheel_square = z[:, None].expand(-1, 2).clone()
        self.wheel_peak = self.wheel_square.clone()
        self.current_wheel = self.wheel_square.clone()
        self.step_terms = z[:, None].expand(-1, 10).clone()
        self.last_result = {}
        self._state_names = tuple(k for k, v in vars(self).items() if torch.is_tensor(v))

    def reset(self, ids, h0, prefix_s):
        # Existing environment calls configure_reset immediately after physical FK.
        for k in self._state_names:
            v = getattr(self, k)
            if torch.is_tensor(v):
                v[ids] = -1 if k == "entry" else True if k == "sequence_valid" else 0
        self.h0[ids] = h0
        self.prefix_s[ids] = prefix_s

    def configure_reset(self, ids, geometry, phases, levels):
        self.initial_geometry[ids] = geometry
        self.last_geometry[ids] = geometry
        self.phase[ids] = phases
        self.initial_phase[ids] = phases
        self.course_level[ids] = levels
        # Auxiliary reset is not completion of preceding stages.
        self.last_phi[ids] = phases.to(self.last_phi.dtype) / 4

    def substep(self, geometry, wheel_velocity, gravity, angular, vz, motor, caps):
        dt = 0.001
        self.current_wheel.copy_(wheel_velocity)
        levels = self.course_level
        scale = wheel_velocity.new_tensor(WHEEL_SCALE)[levels]
        peak = wheel_velocity.new_tensor(WHEEL_PEAK)[levels]
        speed = (geometry - self.last_geometry) / dt
        speed[:, 2:] = angle_difference(geometry[:, 2:], self.last_geometry[:, 2:]) / dt
        self.geometry_speed.copy_(speed)
        self.last_geometry.copy_(geometry)
        tilt = torch.acos((-gravity[:, 1]).clamp(-1, 1))
        absolute = wheel_velocity.abs()
        self.wheel_square += wheel_velocity.square() * dt
        self.wheel_peak = torch.maximum(self.wheel_peak, absolute)
        self.tilt_peak = torch.maximum(self.tilt_peak, tilt)
        self.level_time += (tilt <= math.radians(10)) * dt
        self.duration += dt
        self.tilt_bad_time = torch.where(tilt > math.radians(25), self.tilt_bad_time + dt, 0.0)
        self.fast_time = torch.where(absolute.amax(-1) > 8, self.fast_time + dt, 0.0)
        short = geometry.new_tensor(self.geometry["short_length_m"])
        fraction = (
            (self.initial_geometry[:, :2] - geometry[:, :2])
            / (self.initial_geometry[:, :2] - short).clamp_min(0.01)
        ).clamp(0, 1)
        short_angle = geometry.new_tensor(self.geometry["short_swing_rad"])
        reference = self.initial_geometry[:, 2:] + fraction * angle_difference(
            short_angle, self.initial_geometry[:, 2:]
        )
        swing_change = angle_difference(geometry[:, 2:], reference).abs()
        premature_turn = ((swing_change - 0.10).clamp_min(0) / 0.20).amax(-1)
        premature_extension = ((geometry[:, :2] - short - 0.015).clamp_min(0) / 0.03).amax(-1)
        mirror_length = (geometry[:, 0] - geometry[:, 1]).abs() / 0.03
        order = torch.where(
            self.phase == 0,
            premature_turn,
            torch.where(self.phase == 1, premature_extension, mirror_length),
        )
        # Sequence validity is an episode certificate, not an action projection.
        self.sequence_valid &= ~(
            ((self.phase == 0) & (premature_turn > 1))
            | ((self.phase == 1) & (premature_extension > 1))
        )
        acceleration = (vz - self.last_vz) / dt
        self.last_vz.copy_(vz)
        impact = torch.where(
            self.phase < 2,
            (vz.clamp_min(0) / 0.15).square(),
            ((vz.abs() - 0.30).clamp_min(0) / 0.30).square(),
        )
        impact = torch.maximum(impact, ((acceleration.abs() - 10).clamp_min(0) / 10).square())
        geometric_speed = torch.maximum(
            (speed[:, :2].abs() / 0.30).amax(-1), (speed[:, 2:].abs() / 2.0).amax(-1)
        )
        # Costs are bounded individually; raw extrema above remain unbounded evidence.
        terms = torch.stack(
            (
                (wheel_velocity / scale[:, None]).square().clamp_max(1).mean(-1),
                ((absolute - peak[:, None]).clamp_min(0) / peak[:, None]).square().amax(-1),
                (tilt / math.radians(10)).square(),
                (angular[:, [0, 2]].norm(dim=-1) / 0.5).square(),
                (geometric_speed - 1).clamp_min(0).square(),
                torch.zeros_like(tilt),
                impact,
                order.square(),
                (motor[:, [0, 1, 3, 4]] / caps[:, [0, 1, 3, 4]]).square().mean(-1),
                torch.ones_like(tilt),
            ),
            -1,
        ).clamp(0, 1)
        self.step_terms += terms
        self.step_count += 1
        self.step_cost += (terms * terms.new_tensor(COST_WEIGHTS)).sum(-1)

    def observe(
        self,
        *,
        time_s,
        state,
        target,
        motor,
        student,
        terminal,
        truncated,
        hp10,
        angular_velocity,
        geometry,
        dt_s=0.01,
    ):
        if not bool(torch.isfinite(torch.cat((state, target, geometry), -1)).all()):
            raise ValueError("Finite leg-driven reward boundary required")
        if not bool((self.step_count == 10).all()):
            raise ValueError("Leg-driven reward requires ten actual 1ms substeps")
        short = geometry.new_tensor(self.geometry["short_length_m"])
        turn = geometry.new_tensor(self.geometry["support_swing_rad"])
        length = geometry[:, :2]
        turn_error = angle_difference(geometry[:, 2:], turn).abs().amax(-1)
        tilt = torch.acos(state[:, 12].clamp(-1, 1))
        angular_ok = angular_velocity[:, [0, 2]].norm(dim=-1) <= 0.5
        posture_ok = (tilt <= math.radians(10)) & angular_ok
        shortened = (length <= short + 0.01).all(-1) & (
            self.geometry_speed[:, :2].abs().amax(-1) <= 0.05
        )
        if self.progression is not None:
            short_angle = geometry.new_tensor(self.geometry["short_swing_rad"])
            shortened &= angle_difference(geometry[:, 2:], short_angle).abs().amax(-1) <= 0.10
        rotated = (
            (length <= short + 0.015).all(-1)
            & (turn_error <= 0.10)
            & (self.geometry_speed[:, 2:].abs().amax(-1) <= 0.5)
        )
        supported = (state[:, 9:11] > 2).all(-1)
        lifted = (abs(state[:, 3] - 0.34) <= 0.02) & supported & posture_ok
        ready = lifted & (abs(self.last_vz) <= 0.05)
        # Final instantaneous wheel condition uses last physical substep, not network mean.
        ready &= (self.last_wheel_speed() <= 1).all(-1)
        gate = torch.where(
            self.phase == 0,
            shortened,
            torch.where(self.phase == 1, rotated, torch.where(self.phase == 2, lifted, ready)),
        )
        gate &= student & ~terminal & (state[:, 11] <= 0.005)
        self.dwell = torch.where(gate, self.dwell + dt_s, 0.0)
        thresholds = torch.where(self.phase == 3, 0.5, 0.1)
        event = gate & (self.dwell >= thresholds - 1e-6)
        old_phase = self.phase.clone()
        failed = terminal | (self.tilt_bad_time >= 0.03 - 1e-6) | (self.fast_time >= 0.02 - 1e-6)
        if self.progression is not None:
            failed |= ~self.sequence_valid
        event &= ~failed
        progressed = event & (old_phase < 3)
        self.phase += progressed.long()
        self.stage_start = torch.where(progressed, time_s, self.stage_start)
        self.dwell = torch.where(progressed, 0.0, self.dwell)
        entered = progressed & (old_phase == 2)
        self.entry = torch.where(entered, time_s, self.entry)
        self.hold = torch.where(self.phase == 3, self.dwell, 0.0)
        initial_length = (self.initial_geometry[:, :2] - short).clamp_min(0.01)
        p0 = ((self.initial_geometry[:, :2] - length) / initial_length).clamp(0, 1).amin(-1)
        turn_start = self.initial_geometry[:, 2:]
        if self.progression is not None:
            turn_start = torch.where(
                (self.initial_phase < 1)[:, None],
                geometry.new_tensor(self.geometry["short_swing_rad"]),
                turn_start,
            )
        initial_error = angle_difference(turn_start, turn).abs().amax(-1).clamp_min(0.1)
        p1 = (1 - turn_error / initial_error).clamp(0, 1)
        p2 = ((state[:, 3] - self.h0) / (0.34 - self.h0).clamp_min(0.02)).clamp(0, 1)
        p3 = (self.hold / 0.5).clamp(0, 1)
        self.progress = torch.where(
            self.phase == 0,
            p0,
            torch.where(self.phase == 1, p1, torch.where(self.phase == 2, p2, p3)),
        )
        phi = (self.phase.to(state.dtype) + self.progress) / 4
        wheel_rms = torch.sqrt(self.wheel_square / self.duration.clamp_min(0.001)[:, None])
        quality_scale = state.new_tensor(WHEEL_SCALE)[self.course_level]
        quality_peak = state.new_tensor(WHEEL_PEAK)[self.course_level]
        quality = (wheel_rms.amax(-1) <= quality_scale) & (self.wheel_peak.amax(-1) <= quality_peak)
        quality &= (
            (self.tilt_peak <= math.radians(15))
            & (self.level_time / self.duration.clamp_min(0.001) >= 0.95)
            & self.sequence_valid
        )
        succeeded = event & (old_phase == 3) & quality & ~failed
        timeout = (time_s >= self.max_s - 1e-6) | truncated
        done = succeeded | failed | timeout
        delta = torch.where(self.valid_target[:, None], target - self.last_target, 0.0)
        target_cost = (delta[:, [0, 1, 3, 4]] / 0.05).square().clamp_max(1).mean(-1)
        terms = self.step_terms / self.step_count[:, None]
        terms[:, 5] = target_cost
        cost = (terms * terms.new_tensor(COST_WEIGHTS)).sum(-1)
        potential = 2 * (self.gamma * torch.where(done, 0.0, phi) - self.last_phi)
        bonuses = (
            [2.0, 2.0, 4.0, 0.0]
            if self.progression is None
            else [*self.progression["completion_bonus"], 0.0]
        )
        bonus = torch.where(progressed, state.new_tensor(bonuses)[old_phase], 0.0)
        failure = done & ~succeeded
        prone_density = anti_prone_density(
            self.anti_prone,
            phase=self.phase,
            progress=self.progress,
            stage_age=(time_s - self.stage_start).clamp_min(0),
            height=state[:, 3],
            gate=gate,
        )
        # Failure cannot erase the future added running cost. This conservative
        # tail applies to every failed initial group, not only low prone states.
        prone_failure_tail = (
            self.anti_prone["max_rate"] * (self.max_s - time_s).clamp_min(0) * failure
            if self.anti_prone is not None
            else torch.zeros_like(time_s)
        )
        remaining_density = torch.zeros_like(time_s)
        base_failure_tail = 4 * (self.max_s - time_s).clamp_min(0) * failure
        if self.progression is not None:
            remaining_density = -self.progression["remaining_rate"] * (1 - self.progress) * ~gate
            # Total tail includes the base/prone running costs, replacing rather
            # than adding another duplicate failure payment.
            base_failure_tail = (
                self.progression["failure_tail_rate"] * (self.max_s - time_s).clamp_min(0) * failure
            )
            prone_failure_tail = torch.zeros_like(time_s)
        reward = (
            potential
            + bonus
            - dt_s * 4 * cost
            + dt_s * (prone_density + remaining_density)
            - prone_failure_tail
            + 100 * succeeded
            - 20 * failure
            - base_failure_tail
        )
        self.last_phi.copy_(phi)
        self.last_delta.copy_(delta)
        self.last_target.copy_(target)
        self.valid_target.fill_(True)
        self.elapsed.copy_(time_s)
        self.reward.copy_(reward)
        self.total_reward += reward
        self.success |= succeeded
        density = {
            k: -4 * w * terms[:, i]
            for i, (k, w) in enumerate(
                zip(
                    (
                        "actual_wheel_speed",
                        "wheel_overspeed",
                        "whole_tilt",
                        "angular_speed",
                        "leg_speed",
                        "leg_target_delta",
                        "lift_impact",
                        "sequence",
                        "leg_effort",
                        "time",
                    ),
                    COST_WEIGHTS,
                    strict=True,
                )
            )
        }
        density["anti_prone"] = prone_density
        density["stage_remaining"] = remaining_density
        self.last_result = dict(
            anti_prone_failure_tail=prone_failure_tail,
            failure_tail=base_failure_tail + prone_failure_tail,
            phase_bonus=bonus,
            reward=reward,
            entered=entered,
            success=succeeded,
            terminated=succeeded | failed,
            truncated=timeout & ~succeeded & ~failed,
            done=done,
            density=density,
            potential=potential,
            leg_driven_quality=quality,
            entry_eligible=lifted,
            hold_quality=ready,
        )
        self.step_count.zero_()
        self.step_terms.zero_()
        self.step_cost.zero_()
        return self.last_result

    def last_wheel_speed(self):
        return self.current_wheel.abs()

    def extra(self, state, clearance, restricted, initialized):
        phase3 = self.phase.clamp_max(2)
        return dict(
            initial_root_height=self.h0[:, None] / 0.5,
            height_progress=((state[:, 3] - self.h0) / (0.34 - self.h0).clamp_min(0.02)).clamp(
                0, 1
            )[:, None],
            chassis_clearance=clearance[:, None] / 0.5,
            closure=state[:, 11:12] / 0.005,
            restricted_contact=restricted[:, None] / 100,
            stage_onehot=torch.nn.functional.one_hot(phase3, 3).to(state.dtype),
            entry_latched=(self.entry >= 0)[:, None].to(state.dtype),
            entry_age=torch.where(self.entry >= 0, self.elapsed - self.entry, 0.0)[:, None] / 10,
            hold_age=self.hold[:, None] / 10,
            student_age=(self.elapsed - self.prefix_s).clamp_min(0)[:, None] / 23,
            time_remaining=(6 - self.elapsed).clamp_min(0)[:, None] / 23,
            entry_dwell_progress=(self.dwell / 0.1).clamp_max(1)[:, None],
            initialization_complete=initialized[:, None].to(state.dtype),
        )

    def critic(self, base):
        wheel_rms = (self.wheel_square / self.duration.clamp_min(0.001)[:, None]).sqrt()
        # 38 actual causal columns, added to the unchanged 904 prefix.
        extra = torch.cat(
            (
                torch.nn.functional.one_hot(self.phase, 4).to(base.dtype),
                self.dwell[:, None],
                self.initial_geometry,
                self.last_geometry,
                self.geometry_speed,
                self.course_level[:, None].to(base.dtype),
                self.initial_phase[:, None].to(base.dtype),
                self.sequence_valid[:, None].to(base.dtype),
                self.wheel_square,
                self.wheel_peak,
                self.duration[:, None],
                self.level_time[:, None],
                self.tilt_peak[:, None],
                self.tilt_bad_time[:, None],
                self.fast_time[:, None],
                self.last_vz[:, None],
                self.stage_start[:, None],
                self.progress[:, None],
                wheel_rms,
                self.current_wheel,
                self.step_count[:, None],
                self.step_cost[:, None],
            ),
            -1,
        )
        if base.shape[1] != 904 or extra.shape[1] != 38:
            raise ValueError("Explicit leg-driven Critic 904+38 columns required")
        return torch.cat((base, extra), -1)

    def episode_record(self, i):
        return dict(
            course_level=int(self.course_level[i]),
            initial_phase=int(self.initial_phase[i]),
            final_phase=int(self.phase[i]),
            sequence_valid=bool(self.sequence_valid[i]),
            progress=float(self.progress[i]),
            length_m=self.last_geometry[i, :2].tolist(),
            swing_rad=self.last_geometry[i, 2:].tolist(),
            phase_age_s=float(self.elapsed[i] - self.stage_start[i]),
            wheel_rms=(self.wheel_square[i] / self.duration[i].clamp_min(0.001)).sqrt().tolist(),
            wheel_peak=self.wheel_peak[i].tolist(),
            tilt_peak_rad=float(self.tilt_peak[i]),
            level_fraction=float(self.level_time[i] / self.duration[i].clamp_min(0.001)),
            quality=bool(self.last_result["leg_driven_quality"][i]),
        )

    def state_dict(self):
        return copy.deepcopy({k: getattr(self, k) for k in (*self._state_names, "last_result")})

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError("Leg-driven lifecycle snapshot fields differ")
        for k, v in state.items():
            if torch.is_tensor(v):
                if v.shape != getattr(self, k).shape:
                    raise ValueError("Leg-driven lifecycle row shape")
                getattr(self, k).copy_(v.to(self.device))
            else:
                setattr(self, k, copy.deepcopy(v))


def review_window(rows, level, history):
    """Per-condition promotion; UNKNOWN coverage is never a pass."""
    selected = [r for r in rows if r["condition"].get("course_level") == level]
    sufficient = bool(selected) and all(
        r["ended"] >= (256 if r["objective"] == "Swing" else 64) for r in selected
    )
    groups = {r["condition"]["group"] for r in selected}
    sufficient &= groups == set(GROUPS)
    if level >= 4:
        sufficient &= all(
            {r["condition"]["DR_strength"] for r in selected if r["condition"]["group"] == g}
            == {0.0, 0.1 if level == 4 else 0.25}
            for g in GROUPS
        )
    passed = sufficient and all(
        r["student_success"] / r["ended"] >= (0.95 if r["objective"] == "Swing" else 0.9)
        and r["quality"] / r["ended"] >= 0.9
        and r.get("sequence_valid", 0) / r["ended"] >= 0.98
        for r in selected
    )
    full = [r for r in selected if r["objective"] == "Swing"]
    score = min(
        (
            r["student_success"] / r["ended"]
            + r["quality"] / r["ended"]
            + r.get("sequence_valid", 0) / r["ended"]
            + 0.1 * r.get("continuous_progress_sum", r.get("phase_progress_sum", 0)) / r["ended"]
            - 0.01 * r.get("wheel_rms_sum", 0) / r["ended"]
            for r in full
            if r["ended"] > 0
        ),
        default=0.0,
    )
    record = dict(
        level=level,
        coverage_sufficient=bool(sufficient),
        passed=bool(passed),
        score=score,
        full_success=min(
            (r["student_success"] / r["ended"] for r in full if r["ended"] > 0), default=0.0
        ),
    )
    recent = [r for r in history if r["level"] == level]
    advance = level < 5 and len(recent) >= 2 and passed and all(r["passed"] for r in recent[-2:])
    plateau = (
        len(recent) >= 5
        and sufficient
        and max(r["score"] for r in recent[-5:])
        <= max((r["score"] for r in recent[:-5]), default=-1.0)
        and score <= max(r["score"] for r in recent[-5:])
        and not advance
    )
    lost_retention = (
        level > 0
        and sufficient
        and bool(recent)
        and record["full_success"] < 0.8
        and recent[-1].get("full_success", 1.0) < 0.8
    )
    return record, advance, plateau or lost_retention
