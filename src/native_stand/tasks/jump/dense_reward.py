"""Bounded progress rewards and physical event accounting for direct-raw Jump.

This scorer never changes the controller phase, actuator targets or physics.
Packets are explicit simulator truth for rewards/Critic, never Actor inputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import ClassVar

import numpy as np

from .phases import Phase

VERSION = "jump_bounded_progress_reward_v1"
LEGS = (0, 1, 3, 4)


def unit(value):
    return min(max(float(value), 0.0), 1.0)


def policy_substep_reward(total, shaping, *, index, dt_s, discount_rate_s):
    """Aggregate potential shaping at the same discount as the policy step.

    Sum gamma_ms**j * (gamma_ms*Phi[j+1]-Phi[j]) telescopes to
    gamma_policy*Phi[end]-Phi[start]. Other physical costs remain unchanged.
    """
    return total + (math.exp(-discount_rate_s * dt_s * index) - 1.0) * shaping


@dataclass(frozen=True)
class RewardConfig:
    crouch_leg_m: float = 0.145
    crouch_span_m: float = 0.20
    extension_leg_m: float = 0.36
    airborne_dwell_s: float = 0.004
    takeoff_min_vz_m_s: float = 0.10
    stable_duration_s: float = 0.20
    stable_tilt_rad: float = math.radians(5)
    stable_omega_rad_s: float = 0.30
    stable_vz_m_s: float = 0.10
    stable_vxy_m_s: float = 0.15
    stable_height_error_m: float = 0.035
    force_reference_bodyweights: float = 3.0
    w_crouch: float = 4.0
    w_push: float = 5.0
    w_extension: float = 0.5
    w_air_rise: float = 1.0
    w_air_shape: float = 1.0
    w_cushion: float = 2.0
    w_settle: float = 2.0
    w_takeoff: float = 2.0
    w_apex: float = 3.0
    w_landing: float = 6.0
    w_handover: float = 4.0
    w_time: float = 0.5
    w_posture: float = 1.0
    w_angular: float = 0.05
    w_symmetry: float = 0.2
    w_torque: float = 0.02
    w_target_change: float = 0.05
    w_drift: float = 0.2
    w_down_speed: float = 0.5
    w_slip: float = 0.2
    w_force_peak: float = 0.25
    w_stroke: float = 0.2
    w_settle_height: float = 0.5
    w_settle_leg_speed: float = 0.1
    failure_cost: float = 20.0
    timeout_cost: float = 8.0
    require_crouch: bool = True
    goal_gate_landing: bool = False
    no_takeoff_deadline_s: float = 0.0
    w_velocity_potential: float = 0.0
    potential_events: bool = False
    discount_rate_s: float = 0.2

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in asdict(self).values()):
            raise ValueError("finite nonnegative reward parameters required")
        if not 0.126 < self.crouch_leg_m <= 0.15 < self.extension_leg_m < 0.373:
            raise ValueError("reward geometry must respect the existing C guards")
        if min(self.crouch_span_m, self.airborne_dwell_s, self.stable_duration_s) <= 0:
            raise ValueError("positive progress span and dwell required")
        if (
            min(
                self.stable_tilt_rad,
                self.stable_omega_rad_s,
                self.stable_vz_m_s,
                self.stable_vxy_m_s,
                self.stable_height_error_m,
                self.failure_cost,
                self.timeout_cost,
            )
            <= 0
        ):
            raise ValueError("positive physical tolerances and terminal costs required")


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    horizon_s: float
    mass_kg: float
    target_rise_m: float = 0.05
    target_clearance_m: float = 0.02
    air_leg_m: float = 0.18
    entry_height_m: float = 0.34

    def __post_init__(self):
        if self.kind not in ("ground_active_jump", "landing_only"):
            raise ValueError("explicit reset provenance required")
        values = [getattr(self, k) for k in TaskSpec.__dataclass_fields__ if k != "kind"]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("finite positive physical task parameters required")
        if not 0.126 < self.air_leg_m < 0.373 or self.target_clearance_m <= 0.002:
            raise ValueError("legal retraction target and genuine tire clearance required")


@dataclass(frozen=True)
class Packet:
    valid_phases: ClassVar[tuple] = tuple(range(9))
    dt_s: float
    phase: int
    owner: int
    COM_height_m: float
    COM_velocity_m_s: tuple
    root_height_m: float
    tilt_rad: tuple
    omega_rad_s: tuple
    leg_length_m: tuple
    leg_swing_rad: tuple
    spring_length_m: tuple
    contact: tuple
    tire_clearance_m: tuple
    wheel_normal_force_N: tuple
    contact_slip_m_s: tuple
    actual_motor_Nm: tuple
    actual_dq_rad_s: tuple
    adopted_target: tuple
    force_valid: bool
    motion_confirmed: bool = False
    fault: bool = False
    external_takeoff_assist: bool = False

    def __post_init__(self):
        finite = [self.COM_height_m, self.root_height_m]
        sizes = dict(
            COM_velocity_m_s=3,
            tilt_rad=2,
            omega_rad_s=3,
            leg_length_m=2,
            leg_swing_rad=2,
            spring_length_m=2,
            contact=2,
            tire_clearance_m=2,
            wheel_normal_force_N=2,
            contact_slip_m_s=2,
            actual_motor_Nm=6,
            actual_dq_rad_s=6,
            adopted_target=6,
        )
        for name, size in sizes.items():
            value = tuple(getattr(self, name))
            if len(value) != size:
                raise ValueError("finite explicitly aligned packet field " + name)
            if name != "contact":
                value = tuple(float(x) for x in value)
            object.__setattr__(self, name, value)
            finite.extend(value)
        if self.phase not in self.valid_phases:
            raise ValueError("未知C阶段")
        if self.owner not in range(5) or not 0 <= self.dt_s <= 0.02:
            raise ValueError("known owner and actual interval <=20ms required")
        if not np.isfinite(finite).all():
            if not np.isfinite([self.COM_height_m, self.root_height_m]).all():
                raise ValueError("finite measured heights required")
            for name in sizes:
                if not np.isfinite(getattr(self, name)).all():
                    raise ValueError("finite explicitly aligned packet field " + name)
        if any(type(x) not in (bool, np.bool_) for x in self.contact):
            raise ValueError("contact truth must be explicit booleans")
        object.__setattr__(self, "contact", tuple(bool(x) for x in self.contact))
        if min(*self.wheel_normal_force_N, *self.contact_slip_m_s) < 0:
            raise ValueError("normal force and slip magnitude must be nonnegative")


@dataclass
class DenseReward:
    spec: TaskSpec
    config: RewardConfig
    previous: Packet
    elapsed_s: float = 0.0
    jump_control_s: float = 0.0
    had_support: bool = False
    crouch_legal: bool = False
    push_seen: bool = False
    assisted: bool = False
    off_s: float = 0.0
    air_origin_COM_m: float = 0.0
    air_origin_vz_m_s: float = 0.0
    takeoff_COM_m: float = 0.0
    takeoff_vz_m_s: float = 0.0
    peak_COM_m: float = 0.0
    peak_tire_clearance_m: float = 0.0
    has_taken_off: bool = False
    had_airborne_start: bool = False
    has_landed: bool = False
    apex_paid: bool = False
    landing_paid: bool = False
    handover_paid: bool = False
    failure_paid: bool = False
    timeout_paid: bool = False
    landing_age_s: float = 0.0
    touchdown_down_v_m_s: float = 0.2
    stable_s: float = 0.0
    force_peak_score: float = 0.0
    best: dict = field(
        default_factory=lambda: dict(
            crouch=0.0,
            push=0.0,
            extension=0.0,
            air_rise=0.0,
            air_shape=0.0,
            cushion=0.0,
            settle=0.0,
        )
    )

    @classmethod
    def start(cls, spec, initial, config=None):
        config = config or RewardConfig()
        if initial.dt_s != 0:
            raise ValueError("initial packet must be a reset snapshot, not a rewarded step")
        memory = cls(spec, config, initial, had_support=bool(all(initial.contact)))
        memory.best["crouch"] = memory.crouch_score(initial)
        memory.best["extension"] = memory.extension_score(initial)
        memory.assisted = initial.external_takeoff_assist
        memory.had_airborne_start = spec.kind == "landing_only" and (
            not any(initial.contact) and min(initial.tire_clearance_m) > 0.002
        )
        return memory

    def crouch_score(self, p):
        return unit(
            1
            - np.mean(abs(np.asarray(p.leg_length_m) - self.config.crouch_leg_m))
            / self.config.crouch_span_m
        )

    def extension_score(self, p):
        c = self.config
        return unit(
            (np.mean(p.leg_length_m) - c.crouch_leg_m) / (c.extension_leg_m - c.crouch_leg_m)
        )

    def progress(self, terms, name, score, weight):
        score = unit(score)
        terms[name + "_progress"] = weight * max(score - self.best[name], 0.0)
        self.best[name] = max(self.best[name], score)

    @property
    def rise_m(self):
        return max(self.peak_COM_m - self.takeoff_COM_m, 0.0) if self.has_taken_off else 0.0

    @property
    def active_eligible(self):
        return self.spec.kind == "ground_active_jump" and not self.assisted

    def event_potential(self):
        c = self.config
        if not self.active_eligible:
            return 0.0
        return (
            c.w_takeoff * self.has_taken_off
            + c.w_air_rise * self.best["air_rise"]
            + c.w_apex * self.apex_paid
            + c.w_landing * self.landing_paid
        )

    def step(self, p):
        c, s, previous = self.config, self.spec, self.previous
        previous_events = self.event_potential() if c.potential_events else 0.0
        if p.dt_s == 0 and not p.fault:
            raise ValueError("zero interval is only legal for a pre-integration fault")
        if p.dt_s > 0 and not p.force_valid and not p.fault:
            raise ValueError("solved contact force/slip data must be wired, never defaulted")
        self.elapsed_s += p.dt_s
        self.assisted |= p.external_takeoff_assist
        active_control = p.owner == 2 and Phase.CROUCH <= p.phase <= Phase.SETTLE
        self.jump_control_s += p.dt_s if active_control else 0.0
        both, any_contact = all(p.contact), any(p.contact)
        vz = p.COM_velocity_m_s[2]
        air = not any_contact and min(p.tire_clearance_m) > getattr(
            s, "takeoff_clearance_m", 0.002
        )
        terms = {
            name: 0.0
            for name in (
                "crouch_progress",
                "push_progress",
                "extension_progress",
                "air_rise_progress",
                "air_shape_progress",
                "cushion_progress",
                "settle_progress",
                "takeoff",
                "apex",
                "landing_stable",
                "handover",
                "time",
                "posture",
                "angular",
                "symmetry",
                "torque",
                "target_change",
                "drift",
                "down_speed",
                "slip",
                "force_peak",
                "stroke_margin",
                "settle_height",
                "settle_leg_speed",
                "failure",
                "timeout",
            )
        }
        self.had_support |= bool(both)
        if active_control and p.phase == Phase.CROUCH:
            self.crouch_legal |= min(p.leg_length_m) > 0.126 and max(p.leg_length_m) <= 0.15
            self.progress(terms, "crouch", self.crouch_score(p), c.w_crouch)
        if active_control and p.phase == Phase.PUSH:
            if not self.push_seen:
                self.best["extension"] = self.extension_score(previous)
            self.push_seen = True
            if any_contact and not self.has_taken_off:
                target_vz = math.sqrt(2 * 9.81 * s.target_rise_m)
                self.progress(terms, "push", max(vz, 0.0) / target_vz, c.w_push)
                if vz > 0:
                    self.progress(terms, "extension", self.extension_score(p), c.w_extension)

        if air:
            if self.off_s == 0:
                self.air_origin_COM_m = p.COM_height_m
                self.air_origin_vz_m_s = vz
            self.off_s += p.dt_s
        else:
            self.off_s = 0.0
        eligible = (
            self.active_eligible
            and self.had_support
            and (not c.require_crouch or self.crouch_legal)
            and self.push_seen
        )
        if (
            eligible
            and active_control
            and p.phase in (Phase.PUSH, Phase.FLIGHT)
            and self.off_s >= c.airborne_dwell_s - 1e-10
            and not self.has_taken_off
            and self.air_origin_vz_m_s > c.takeoff_min_vz_m_s
        ):
            self.has_taken_off = True
            self.takeoff_COM_m = self.air_origin_COM_m
            self.takeoff_vz_m_s = self.air_origin_vz_m_s
            self.peak_COM_m = p.COM_height_m
            terms["takeoff"] = c.w_takeoff
        flight_proven = self.has_taken_off or self.had_airborne_start
        if self.has_taken_off and self.active_eligible and air and not self.has_landed:
            self.peak_COM_m = max(self.peak_COM_m, p.COM_height_m)
            self.peak_tire_clearance_m = max(self.peak_tire_clearance_m, min(p.tire_clearance_m))
            if not self.has_landed:
                self.progress(terms, "air_rise", self.rise_m / s.target_rise_m, c.w_air_rise)
                shape = 0.5 * unit(min(p.tire_clearance_m) / s.target_clearance_m)
                shape += 0.5 * unit(
                    1 - np.mean(abs(np.asarray(p.leg_length_m) - s.air_leg_m)) / 0.20
                )
                self.progress(terms, "air_shape", shape, c.w_air_shape)
            if previous.COM_velocity_m_s[2] > 0 >= vz and not self.apex_paid:
                self.apex_paid = True
                terms["apex"] = c.w_apex * unit(self.rise_m / s.target_rise_m)
        if flight_proven and any_contact and not self.has_landed:
            self.has_landed = True
            self.touchdown_down_v_m_s = max(-previous.COM_velocity_m_s[2], 0.2)
            self.best["cushion"] = self.best["settle"] = 0.0
        if self.has_landed:
            self.landing_age_s += p.dt_s
        settling = (
            self.has_landed and both and (p.phase == Phase.SETTLE or self.landing_age_s >= 0.05)
        )
        if self.has_landed and any_contact:
            self.progress(
                terms, "cushion", 1 - max(-vz, 0.0) / self.touchdown_down_v_m_s, c.w_cushion
            )
            force_score = min(
                max(
                    sum(p.wheel_normal_force_N) / (s.mass_kg * 9.81)
                    - c.force_reference_bodyweights,
                    0.0,
                )
                ** 2,
                16.0,
            )
            terms["force_peak"] = -c.w_force_peak * max(force_score - self.force_peak_score, 0.0)
            self.force_peak_score = max(self.force_peak_score, force_score)
        if settling:
            score = np.mean(
                [
                    unit(1 - abs(p.root_height_m - s.entry_height_m) / 0.10),
                    unit(1 - abs(vz)),
                    unit(1 - np.linalg.norm(p.COM_velocity_m_s[:2]) / 0.5),
                    unit(1 - np.linalg.norm(p.omega_rad_s) / 2),
                    unit(1 - max(abs(x) for x in p.tilt_rad) / 0.35),
                ]
            )
            self.progress(terms, "settle", score, c.w_settle)
        stable = (
            self.has_landed
            and both
            and max(abs(x) for x in p.tilt_rad) < c.stable_tilt_rad
            and np.linalg.norm(p.omega_rad_s) < c.stable_omega_rad_s
            and abs(vz) < c.stable_vz_m_s
            and np.linalg.norm(p.COM_velocity_m_s[:2]) < c.stable_vxy_m_s
            and abs(p.root_height_m - s.entry_height_m) < c.stable_height_error_m
        )
        self.stable_s = self.stable_s + p.dt_s if stable else 0.0
        goal = (
            not c.goal_gate_landing
            or s.kind != "ground_active_jump"
            or (
                self.rise_m + 1e-9 >= s.target_rise_m
                and self.peak_tire_clearance_m + 1e-9 >= s.target_clearance_m
            )
        )
        if goal and self.stable_s >= c.stable_duration_s - 1e-10 and not self.landing_paid:
            self.landing_paid = True
            terms["landing_stable"] = c.w_landing
        if p.motion_confirmed and self.landing_paid and not self.handover_paid:
            self.handover_paid = True
            terms["handover"] = c.w_handover

        if active_control:
            dt = p.dt_s
            terms["time"] = -c.w_time * dt
            terms["posture"] = (
                -c.w_posture * min(sum((x / 0.35) ** 2 for x in p.tilt_rad), 4.0) * dt
            )
            terms["angular"] = (
                -c.w_angular
                * min(sum(x * x for x in p.omega_rad_s[:2]) + 0.1 * p.omega_rad_s[2] ** 2, 16.0)
                * dt
            )
            dl = (p.leg_length_m[0] - p.leg_length_m[1]) / 0.03
            da = math.remainder(p.leg_swing_rad[0] - p.leg_swing_rad[1], 2 * math.pi) / 0.15
            terms["symmetry"] = -c.w_symmetry * min(dl * dl + da * da, 4.0) * dt
            relax = 0.05 if p.phase == Phase.PUSH else 1.0 if p.phase == Phase.SETTLE else 0.3
            terms["torque"] = (
                -c.w_torque
                * relax
                * min(np.mean((np.asarray(p.actual_motor_Nm)[list(LEGS)] / 40) ** 2), 1.0)
                * dt
            )
            # A transport adoption is a discrete event. No dt factor: splitting
            # an unchanged target into 1/5/10ms observations cannot change cost.
            delta = (np.asarray(p.adopted_target) - previous.adopted_target) / [
                0.15,
                0.15,
                40,
                0.15,
                0.15,
                40,
            ]
            terms["target_change"] = -c.w_target_change * relax * min(np.mean(delta * delta), 4.0)
            margin = np.minimum(
                np.asarray(p.spring_length_m) - 0.155, 0.215 - np.asarray(p.spring_length_m)
            )
            terms["stroke_margin"] = (
                -c.w_stroke * min(np.mean((np.maximum(0.002 - margin, 0.0) / 0.002) ** 2), 4.0) * dt
            )
            if air:
                terms["drift"] = (
                    -c.w_drift * min(np.linalg.norm(p.COM_velocity_m_s[:2]) ** 2 / 0.5**2, 4.0) * dt
                )
            if any_contact:
                slip = np.asarray(p.contact_slip_m_s)[np.asarray(p.contact, bool)]
                terms["slip"] = -c.w_slip * min(np.mean(slip * slip), 4.0) * dt
            if self.has_landed and any_contact:
                terms["down_speed"] = -c.w_down_speed * min(max(-vz - 0.2, 0.0) ** 2, 9.0) * dt
            if settling:
                terms["settle_height"] = (
                    -c.w_settle_height
                    * min(((p.root_height_m - s.entry_height_m) / 0.1) ** 2, 4.0)
                    * dt
                )
                terms["settle_leg_speed"] = (
                    -c.w_settle_leg_speed
                    * min(np.mean((np.asarray(p.actual_dq_rad_s)[list(LEGS)] / 5) ** 2), 4.0)
                    * dt
                )
        mechanical_fault = (
            min(p.leg_length_m) <= 0.126
            or max(p.leg_length_m) >= 0.373
            or min(p.spring_length_m) < 0.155 - 1e-5
            or max(p.spring_length_m) > 0.215 + 1e-5
        )
        fault = p.fault or p.phase == Phase.FAULT or mechanical_fault
        if fault and not self.failure_paid:
            terms["failure"] = -c.failure_cost
            self.failure_paid = True
        full_success = (
            self.active_eligible
            and self.has_taken_off
            and self.apex_paid
            and self.rise_m + 1e-9 >= s.target_rise_m
            and self.peak_tire_clearance_m + 1e-9 >= s.target_clearance_m
            and self.landing_paid
            and self.handover_paid
            and not self.failure_paid
        )
        if c.goal_gate_landing and not full_success:
            terms["handover"] = 0.0
        objective = getattr(s, "objective", 2)
        course_success = (
            self.active_eligible
            and not self.failure_paid
            and (
                self.has_taken_off
                if objective == 0
                else self.has_taken_off and self.apex_paid and self.landing_paid
                if objective == 1
                else full_success
            )
        )
        drop_success = (
            s.kind == "landing_only"
            and self.had_airborne_start
            and self.landing_paid
            and not self.failure_paid
        )
        expired = self.elapsed_s >= s.horizon_s - 1e-10 or (
            c.no_takeoff_deadline_s > 0
            and s.kind == "ground_active_jump"
            and not self.has_taken_off
            and self.elapsed_s >= c.no_takeoff_deadline_s - 1e-10
        )
        if (
            expired
            and not (course_success or drop_success or self.failure_paid)
            and not self.timeout_paid
        ):
            terms["timeout"] = -c.timeout_cost
            self.timeout_paid = True
        if c.w_velocity_potential:

            def potential(packet):
                if packet.owner != 2 or packet.phase != Phase.PUSH or not any(packet.contact):
                    return 0.0
                return c.w_velocity_potential * unit(
                    packet.COM_velocity_m_s[2] / math.sqrt(2 * 9.81 * s.target_rise_m)
                )

            # gamma(dt)*Phi(next)-Phi(previous), with terminal Phi=0.
            # Oscillating/holding a partial pose cannot accumulate progress pay.
            next_value = 0.0 if (fault or expired or course_success or drop_success) else potential(p)
            terms["velocity_potential"] = math.exp(
                -c.discount_rate_s * p.dt_s
            ) * next_value - potential(previous)
        if c.potential_events and s.kind == "ground_active_jump":
            for name in ("takeoff", "air_rise_progress", "apex", "landing_stable"):
                terms[name] = 0.0
            terminal = fault or expired or course_success or drop_success
            next_events = 0.0 if terminal else self.event_potential()
            terms["progress_potential"] = (
                math.exp(-c.discount_rate_s * p.dt_s) * next_events - previous_events
            )
        self.previous = p
        total = float(sum(terms.values()))
        if not math.isfinite(total):
            raise ValueError("nonfinite Jump reward")
        return dict(
            total=total,
            terms=terms,
            terminated=bool(fault or expired or course_success or drop_success),
            truncated=False,
            timeout_bootstrap_allowed=False,
            active_jump_success=full_success,
            course_success=course_success,
            landing_only_success=drop_success,
            actual_takeoff=self.has_taken_off,
            landing_stable=self.landing_paid,
            COM_rise_m=self.rise_m,
            jump_control_s=self.jump_control_s,
        )

    def state_dict(self):
        return dict(version=VERSION, **asdict(self))

    @classmethod
    def restore(cls, value):
        data = dict(value)
        if data.pop("version") != VERSION:
            raise ValueError("reward state version mismatch")
        data["spec"] = TaskSpec(**data["spec"])
        data["config"] = RewardConfig(**data["config"])
        data["previous"] = Packet(**data["previous"])
        return cls(**data)

    def critic_fields(self):
        """All reward memory and per-episode goals, with explicit SI scales."""
        fields = {}

        def add(name, value, scale=1.0, unit="bool or normalized scalar"):
            fields["dense." + name] = (np.asarray(value, float), scale, unit)

        add("kind", [self.spec.kind == "ground_active_jump", self.spec.kind == "landing_only"])
        for name, value in vars(self.spec).items():
            if name != "kind":
                add(
                    "goal." + name,
                    value,
                    1.0
                    if name == "objective"
                    else 25.0
                    if name == "mass_kg"
                    else 10.0
                    if name == "horizon_s"
                    else 0.5,
                    "0=takeoff,1=stable_landing,2=complete_jump"
                    if name == "objective"
                    else "kg, seconds or metres by field",
                )
        state = vars(self)
        for name, value in state.items():
            if name in ("spec", "config", "previous", "best"):
                continue
            add(
                name,
                value,
                10.0 if name.endswith("_s") else 1.0,
                "seconds or SI scalar/bool by field",
            )
        for name, value in self.best.items():
            add("best." + name, value)
        packet_scales = dict(
            COM_velocity_m_s=3.0,
            omega_rad_s=4.0,
            actual_dq_rad_s=100.0,
            actual_motor_Nm=[40, 40, 2.44694786377709, 40, 40, 2.44694786377709],
            adopted_target=[1.5, 1.5, 100, 1.5, 1.5, 100],
            wheel_normal_force_N=250.0,
        )
        for name, value in vars(self.previous).items():
            if name in ("phase", "owner"):
                add("previous." + name, np.eye(9 if name == "phase" else 5)[value])
            else:
                add(
                    "previous." + name,
                    value,
                    packet_scales.get(name, 1.0),
                    "explicit previous physical packet SI/bool",
                )
        return fields
