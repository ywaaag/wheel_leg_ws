"""Batched ABI4 reward memory on the physical tensor device.

The scalar UnifiedReward remains the checkpoint/terminal reference. This
module consumes explicit truth tensors; it never controls phases or motors.
"""

import math
from dataclasses import fields

import numpy as np
import torch

from .dense_reward import DenseReward, TaskSpec
from .unified_reward import UnifiedPacket

BEST = ("crouch", "push", "extension", "air_rise", "air_shape", "cushion", "settle")
MEMORY = tuple(
    f.name for f in fields(DenseReward) if f.name not in ("spec", "config", "previous", "best")
)
PACKET = tuple(f.name for f in fields(UnifiedPacket))
GOALS = tuple(f.name for f in fields(TaskSpec) if f.name != "kind")


class DeviceReward:
    def __init__(self, rows, device):
        self.n, self.device = len(rows), device
        self.c = rows[0].config
        if any(row.config != self.c for row in rows):
            raise ValueError("One explicit reward configuration per batch required")
        self.goal_fields = tuple(k for k in vars(rows[0].spec) if k != "kind")
        if any(tuple(k for k in vars(row.spec) if k != "kind") != self.goal_fields for row in rows):
            raise ValueError("One explicit task goal layout per batch required")
        self.state, self.previous, self.goals, self.best = {}, {}, {}, {}
        for name in MEMORY:
            self.state[name] = torch.empty(self.n, dtype=torch.float64, device=device)
        for name in BEST:
            self.best[name] = torch.empty(self.n, dtype=torch.float64, device=device)
        for name in self.goal_fields:
            self.goals[name] = torch.empty(self.n, dtype=torch.float64, device=device)
        self.active = torch.empty(self.n, dtype=torch.bool, device=device)
        for name in PACKET:
            size = np.asarray(getattr(rows[0].previous, name)).shape
            self.previous[name] = torch.empty((self.n, *size), dtype=torch.float64, device=device)
        self.reset(list(range(self.n)), rows)
        self.run = (
            torch.compile(self.step, fullgraph=True, dynamic=False)
            if str(device).startswith("cuda")
            else self.step
        )

    def reset(self, ids, rows):
        index = torch.as_tensor(ids, device=self.device)
        for names, target, source in (
            (MEMORY, self.state, lambda r, k: getattr(r, k)),
            (BEST, self.best, lambda r, k: r.best[k]),
            (self.goal_fields, self.goals, lambda r, k: getattr(r.spec, k)),
            (PACKET, self.previous, lambda r, k: getattr(r.previous, k)),
        ):
            for name in names:
                target[name][index] = torch.as_tensor(
                    np.asarray([source(rows[i], name) for i in ids]),
                    device=self.device,
                    dtype=torch.float64,
                )
        self.active[index] = torch.tensor(
            [rows[i].spec.kind == "ground_active_jump" for i in ids], device=self.device
        )

    def step(self, p, enabled):
        c, s, m, b, old = self.c, self.goals, self.state, self.best, self.previous
        # Double arithmetic preserves the scalar Python reward's accumulation
        # and event thresholds; only the public PPO reward is float32.
        p = {k: v.double() for k, v in p.items()}
        terms = {
            k: torch.zeros_like(m["elapsed_s"])
            for k in (
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
        # Keep finished rows exactly at their first terminal physical tick.
        before = {k: v.clone() for k, v in m.items()}
        before_best = {k: v.clone() for k, v in b.items()}
        dt = p["dt_s"]
        phase = p["phase"]
        own = p["owner"] == 2
        control = own & (phase >= 1) & (phase <= 5)

        def flag(k):
            return m[k] != 0

        def put(k, value):
            return m.__setitem__(k, value.double())

        def clip(v):
            return v.clamp(0, 1)

        def progress(name, score, weight, mask):
            score = clip(score)
            terms[name + "_progress"] = torch.where(
                mask, weight * (score - b[name]).clamp_min(0), 0.0
            )
            b[name] = torch.where(mask, torch.maximum(b[name], score), b[name])

        def event_potential():
            return torch.where(
                self.active & ~flag("assisted"),
                c.w_takeoff * m["has_taken_off"]
                + c.w_air_rise * b["air_rise"]
                + c.w_apex * m["apex_paid"]
                + c.w_landing * m["landing_paid"],
                0.0,
            )

        previous_events = event_potential() if c.potential_events else 0.0
        put("elapsed_s", m["elapsed_s"] + dt)
        put("assisted", flag("assisted") | p["external_takeoff_assist"].bool())
        put("jump_control_s", m["jump_control_s"] + torch.where(control, dt, 0.0))
        both = p["contact"].bool().all(-1)
        contact = p["contact"].bool().any(-1)
        mechanical = (
            (p["leg_length_m"].amin(-1) <= 0.126)
            | (p["leg_length_m"].amax(-1) >= 0.373)
            | (p["spring_length_m"].amin(-1) < 0.155 - 1e-5)
            | (p["spring_length_m"].amax(-1) > 0.215 + 1e-5)
        )
        fault = p["fault"].bool() | (phase == 8) | mechanical
        valid_event = torch.ones_like(contact)
        if "objective" in s:
            valid_event = ~(fault | flag("failure_paid"))
        vz = p["COM_velocity_m_s"][:, 2]
        air = ~contact & (p["tire_clearance_m"].amin(-1) > s.get("takeoff_clearance_m", 0.002))
        put("had_support", flag("had_support") | both)
        crouch = control & (phase == 1)
        put(
            "crouch_legal",
            flag("crouch_legal")
            | (
                crouch & (p["leg_length_m"].amin(-1) > 0.126) & (p["leg_length_m"].amax(-1) <= 0.15)
            ),
        )
        progress(
            "crouch",
            1 - (p["leg_length_m"] - c.crouch_leg_m).abs().mean(-1) / c.crouch_span_m,
            c.w_crouch,
            crouch,
        )

        def extension(x):
            return clip((x.mean(-1) - c.crouch_leg_m) / (c.extension_leg_m - c.crouch_leg_m))

        push = control & (phase == 2)
        b["extension"] = torch.where(
            push & ~flag("push_seen"), extension(old["leg_length_m"]), b["extension"]
        )
        put(
            "push_seen",
            flag("push_seen") | (push & contact & valid_event if "objective" in s else push),
        )
        progress(
            "push",
            vz.clamp_min(0) / torch.sqrt(2 * 9.81 * s["target_rise_m"]),
            c.w_push,
            push & contact & ~flag("has_taken_off"),
        )
        progress(
            "extension",
            extension(p["leg_length_m"]),
            c.w_extension,
            push & contact & ~flag("has_taken_off") & (vz > 0),
        )
        put(
            "air_origin_COM_m",
            torch.where(air & (m["off_s"] == 0), p["COM_height_m"], m["air_origin_COM_m"]),
        )
        put("air_origin_vz_m_s", torch.where(air & (m["off_s"] == 0), vz, m["air_origin_vz_m_s"]))
        put("off_s", torch.where(air, m["off_s"] + dt, 0.0))
        eligible = self.active & ~flag("assisted")
        takeoff = (
            eligible
            & valid_event
            & flag("had_support")
            & (flag("crouch_legal") | (not c.require_crouch))
            & flag("push_seen")
            & control
            & ((phase == 2) | (phase == 3))
            & (m["off_s"] >= c.airborne_dwell_s - 1e-10)
            & ~flag("has_taken_off")
            & (m["air_origin_vz_m_s"] > c.takeoff_min_vz_m_s)
        )
        put("has_taken_off", flag("has_taken_off") | takeoff)
        put("takeoff_COM_m", torch.where(takeoff, m["air_origin_COM_m"], m["takeoff_COM_m"]))
        put("takeoff_vz_m_s", torch.where(takeoff, m["air_origin_vz_m_s"], m["takeoff_vz_m_s"]))
        put("peak_COM_m", torch.where(takeoff, p["COM_height_m"], m["peak_COM_m"]))
        terms["takeoff"] = takeoff.double() * c.w_takeoff
        flight = flag("has_taken_off") | flag("had_airborne_start")
        rising = flag("has_taken_off") & eligible & valid_event & air & ~flag("has_landed")
        put(
            "peak_COM_m",
            torch.where(rising, torch.maximum(m["peak_COM_m"], p["COM_height_m"]), m["peak_COM_m"]),
        )
        put(
            "peak_tire_clearance_m",
            torch.where(
                rising,
                torch.maximum(m["peak_tire_clearance_m"], p["tire_clearance_m"].amin(-1)),
                m["peak_tire_clearance_m"],
            ),
        )
        rise = torch.where(
            flag("has_taken_off"), (m["peak_COM_m"] - m["takeoff_COM_m"]).clamp_min(0), 0.0
        )
        progress("air_rise", rise / s["target_rise_m"], c.w_air_rise, rising)
        shape = 0.5 * clip(p["tire_clearance_m"].amin(-1) / s["target_clearance_m"]) + 0.5 * clip(
            1 - (p["leg_length_m"] - s["air_leg_m"][:, None]).abs().mean(-1) / 0.20
        )
        progress("air_shape", shape, c.w_air_shape, rising)
        apex = rising & (old["COM_velocity_m_s"][:, 2] > 0) & (vz <= 0) & ~flag("apex_paid")
        put("apex_paid", flag("apex_paid") | apex)
        terms["apex"] = apex.double() * c.w_apex * clip(rise / s["target_rise_m"])
        landing = valid_event & flight & contact & ~flag("has_landed")
        put("has_landed", flag("has_landed") | landing)
        put(
            "touchdown_down_v_m_s",
            torch.where(
                landing, (-old["COM_velocity_m_s"][:, 2]).clamp_min(0.2), m["touchdown_down_v_m_s"]
            ),
        )
        for k in ("cushion", "settle"):
            b[k] = torch.where(landing, 0.0, b[k])
        put("landing_age_s", m["landing_age_s"] + torch.where(flag("has_landed"), dt, 0.0))
        settling = flag("has_landed") & both & ((phase == 5) | (m["landing_age_s"] >= 0.05))
        landed = flag("has_landed") & contact
        progress("cushion", 1 - (-vz).clamp_min(0) / m["touchdown_down_v_m_s"], c.w_cushion, landed)
        force = (
            (
                p["wheel_normal_force_N"].sum(-1) / (s["mass_kg"] * 9.81)
                - c.force_reference_bodyweights
            )
            .clamp_min(0)
            .square()
            .clamp_max(16)
        )
        terms["force_peak"] = torch.where(
            landed, -c.w_force_peak * (force - m["force_peak_score"]).clamp_min(0), 0.0
        )
        put(
            "force_peak_score",
            torch.where(landed, torch.maximum(m["force_peak_score"], force), m["force_peak_score"]),
        )
        vxy = torch.linalg.vector_norm(p["COM_velocity_m_s"][:, :2], dim=-1)
        omega = torch.linalg.vector_norm(p["omega_rad_s"], dim=-1)
        tilt = p["tilt_rad"].abs().amax(-1)
        dh = (p["root_height_m"] - s["entry_height_m"]).abs()
        score = (
            clip(1 - dh / 0.10)
            + clip(1 - vz.abs())
            + clip(1 - vxy / 0.5)
            + clip(1 - omega / 2)
            + clip(1 - tilt / 0.35)
        ) / 5
        progress("settle", score, c.w_settle, settling)
        stable = (
            valid_event
            & flag("has_landed")
            & both
            & (tilt < c.stable_tilt_rad)
            & (omega < c.stable_omega_rad_s)
            & (vz.abs() < c.stable_vz_m_s)
            & (vxy < c.stable_vxy_m_s)
            & (dh < c.stable_height_error_m)
        )
        put("stable_s", torch.where(stable, m["stable_s"] + dt, 0.0))
        physical_goal = (rise + 1e-9 >= s["target_rise_m"]) & (
            m["peak_tire_clearance_m"] + 1e-9 >= s["target_clearance_m"]
        )
        goal = (not c.goal_gate_landing) | ~self.active | physical_goal
        skill_landing = torch.ones_like(contact)
        if "objective" in s:
            skill_landing = valid_event & eligible & flag("apex_paid")
        paid = (
            goal
            & skill_landing
            & (m["stable_s"] >= c.stable_duration_s - 1e-10)
            & ~flag("landing_paid")
        )
        put("landing_paid", flag("landing_paid") | paid)
        terms["landing_stable"] = paid.double() * c.w_landing
        hand = p["motion_confirmed"].bool() & flag("landing_paid") & ~flag("handover_paid")
        if "objective" in s:
            hand &= skill_landing & physical_goal
        put("handover_paid", flag("handover_paid") | hand)
        terms["handover"] = hand.double() * c.w_handover
        terms["time"] = torch.where(control, -c.w_time * dt, 0.0)
        terms["posture"] = torch.where(
            control, -c.w_posture * (p["tilt_rad"] / 0.35).square().sum(-1).clamp_max(4) * dt, 0.0
        )
        terms["angular"] = torch.where(
            control,
            -c.w_angular
            * (
                p["omega_rad_s"][:, :2].square().sum(-1) + 0.1 * p["omega_rad_s"][:, 2].square()
            ).clamp_max(16)
            * dt,
            0.0,
        )
        dl = (p["leg_length_m"][:, 0] - p["leg_length_m"][:, 1]) / 0.03
        da = p["leg_swing_rad"][:, 0] - p["leg_swing_rad"][:, 1]
        da = (da - 2 * math.pi * torch.round(da / (2 * math.pi))) / 0.15
        terms["symmetry"] = torch.where(
            control, -c.w_symmetry * (dl.square() + da.square()).clamp_max(4) * dt, 0.0
        )
        relax = torch.where(
            phase == 2,
            dt.new_tensor(0.05),
            torch.where(phase == 5, dt.new_tensor(1.0), dt.new_tensor(0.3)),
        )
        legs = [0, 1, 3, 4]
        terms["torque"] = torch.where(
            control,
            -c.w_torque
            * relax
            * (p["actual_motor_Nm"][:, legs] / 40).square().mean(-1).clamp_max(1)
            * dt,
            0.0,
        )
        delta = (p["adopted_target"] - old["adopted_target"]) / p["adopted_target"].new_tensor(
            [0.15, 0.15, 40, 0.15, 0.15, 40]
        )
        terms["target_change"] = torch.where(
            control, -c.w_target_change * relax * delta.square().mean(-1).clamp_max(4), 0.0
        )
        margin = torch.minimum(p["spring_length_m"] - 0.155, 0.215 - p["spring_length_m"])
        terms["stroke_margin"] = torch.where(
            control,
            -c.w_stroke
            * ((0.002 - margin).clamp_min(0) / 0.002).square().mean(-1).clamp_max(4)
            * dt,
            0.0,
        )
        terms["drift"] = torch.where(
            control & air, -c.w_drift * (vxy.square() / 0.5**2).clamp_max(4) * dt, 0.0
        )
        slip = (p["contact_slip_m_s"].square() * p["contact"]).sum(-1) / p["contact"].sum(
            -1
        ).clamp_min(1)
        terms["slip"] = torch.where(control & contact, -c.w_slip * slip.clamp_max(4) * dt, 0.0)
        terms["down_speed"] = torch.where(
            control & landed,
            -c.w_down_speed * (-vz - 0.2).clamp_min(0).square().clamp_max(9) * dt,
            0.0,
        )
        terms["settle_height"] = torch.where(
            control & settling, -c.w_settle_height * (dh / 0.1).square().clamp_max(4) * dt, 0.0
        )
        terms["settle_leg_speed"] = torch.where(
            control & settling,
            -c.w_settle_leg_speed
            * (p["actual_dq_rad_s"][:, legs] / 5).square().mean(-1).clamp_max(4)
            * dt,
            0.0,
        )
        terms["failure"] = torch.where(fault & ~flag("failure_paid"), -c.failure_cost, 0.0)
        put("failure_paid", flag("failure_paid") | fault)
        success = (
            eligible
            & flag("has_taken_off")
            & flag("apex_paid")
            & (rise + 1e-9 >= s["target_rise_m"])
            & (m["peak_tire_clearance_m"] + 1e-9 >= s["target_clearance_m"])
            & flag("landing_paid")
            & flag("handover_paid")
            & ~flag("failure_paid")
        )
        if c.goal_gate_landing or "objective" in s:
            terms["handover"] = torch.where(success, terms["handover"], 0.0)
        if "objective" in s:
            for name in ("takeoff", "air_rise_progress", "apex", "landing_stable", "handover"):
                terms[name] = torch.where(valid_event, terms[name], 0.0)
            partial_scale = torch.where(s["objective"] == 2, c.complete_partial_reward_scale, 1.0)
            for name in ("takeoff", "air_rise_progress", "apex", "landing_stable"):
                terms[name] *= partial_scale
        course_success = success
        if "objective" in s:
            course_success = (
                eligible
                & ~flag("failure_paid")
                & torch.where(
                    s["objective"] == 0,
                    flag("has_taken_off"),
                    torch.where(
                        s["objective"] == 1,
                        flag("has_taken_off") & flag("apex_paid") & flag("landing_paid"),
                        success,
                    ),
                )
            )
        drop = (
            ~self.active & flag("had_airborne_start") & flag("landing_paid") & ~flag("failure_paid")
        )
        expired = (m["elapsed_s"] >= s["horizon_s"] - 1e-10) | (
            (c.no_takeoff_deadline_s > 0)
            & self.active
            & ~flag("has_taken_off")
            & (m["elapsed_s"] >= c.no_takeoff_deadline_s - 1e-10)
        )
        timeout = expired & ~(course_success | drop | flag("failure_paid")) & ~flag("timeout_paid")
        terms["timeout"] = timeout.double() * -c.timeout_cost
        put("timeout_paid", flag("timeout_paid") | timeout)
        if c.w_velocity_potential:

            def potential(packet):
                return torch.where(
                    (packet["owner"] == 2)
                    & (packet["phase"] == 2)
                    & packet["contact"].bool().any(-1),
                    c.w_velocity_potential
                    * clip(
                        packet["COM_velocity_m_s"][:, 2] / torch.sqrt(2 * 9.81 * s["target_rise_m"])
                    ),
                    0.0,
                )

            next_value = torch.where(fault | expired | course_success | drop, 0.0, potential(p))
            terms["velocity_potential"] = (-c.discount_rate_s * dt).exp() * next_value - potential(
                old
            )
        if c.potential_events:
            for name in ("takeoff", "air_rise_progress", "apex", "landing_stable"):
                terms[name] = torch.where(self.active, 0.0, terms[name])
            next_events = torch.where(
                fault | expired | course_success | drop, 0.0, event_potential()
            )
            terms["progress_potential"] = (
                -c.discount_rate_s * dt
            ).exp() * next_events - previous_events
        # Unified ABI4 additions, in the same summation order as scalar.
        terms["settle_height"] = torch.where(
            own & (phase == 5),
            -c.w_settle_height * (dh / 0.1).square().clamp_max(4) * dt,
            terms["settle_height"],
        )
        balance = own & (phase == 9)
        put("jump_control_s", m["jump_control_s"] + torch.where(balance, dt, 0.0))
        terms["balance_height"] = torch.where(
            balance, -0.5 * (dh / 0.05).square().clamp_max(4) * dt, 0.0
        )
        terms["balance_velocity"] = torch.where(
            balance, -0.5 * (vxy / 0.3).square().clamp_max(4) * dt, 0.0
        )
        terms["balance_posture"] = torch.where(
            balance, -(p["tilt_rad"] / 0.35).square().sum(-1).clamp_max(4) * dt, 0.0
        )
        terms["time"] -= torch.where(balance, c.w_time * dt, 0.0)
        terms["own_completion"] = terms.pop("handover")
        total = torch.zeros_like(dt)
        for value in terms.values():
            total = total + value
        for k in MEMORY:
            m[k] = torch.where(enabled, m[k], before[k])
        for k in BEST:
            b[k] = torch.where(enabled, b[k], before_best[k])
        for k, value in p.items():
            mask = enabled.reshape(self.n, *([1] * (value.ndim - 1)))
            self.previous[k] = torch.where(mask, value, old[k])
        return dict(
            total=torch.where(enabled, total, 0.0),
            shaping=torch.where(
                enabled,
                terms.get("velocity_potential", torch.zeros_like(dt))
                + terms.get("progress_potential", torch.zeros_like(dt)),
                0.0,
            ),
            terminated=enabled & (fault | expired | course_success | drop),
            active_jump_success=success,
            course_success=course_success,
            landing_only_success=drop,
        )

    def critic_values(self):
        out = {"dense.kind": torch.stack((self.active, ~self.active), -1)}
        out.update({"dense.goal." + k: v for k, v in self.goals.items()})
        out.update({"dense." + k: v for k, v in self.state.items()})
        out.update({"dense.best." + k: v for k, v in self.best.items()})
        out.update(
            {
                "dense.previous." + k: (
                    torch.nn.functional.one_hot(v.long(), 10 if k == "phase" else 5)
                    if k in ("phase", "owner")
                    else v
                )
                for k, v in self.previous.items()
            }
        )
        return out

    def export(self, ids, rows):
        # Serialization/normal completed episodes only, never a per-tick path.
        columns = [*self.state.values(), *self.best.values()]
        values = torch.stack(columns, -1)[ids].cpu().numpy()
        packets = {k: v[ids].cpu().numpy() for k, v in self.previous.items()}
        for j, i in enumerate(ids):
            for k, value in zip((*MEMORY, *BEST), values[j], strict=True):
                if k in self.best:
                    rows[i].best[k] = float(value)
                else:
                    setattr(
                        rows[i],
                        k,
                        bool(value) if isinstance(getattr(rows[i], k), bool) else float(value),
                    )
            data = {k: packets[k][j].tolist() for k in PACKET}
            for k in ("phase", "owner"):
                data[k] = int(data[k])
            for k in ("force_valid", "motion_confirmed", "fault", "external_takeoff_assist"):
                data[k] = bool(data[k])
            data["contact"] = tuple(bool(x) for x in data["contact"])
            rows[i].previous = UnifiedPacket(**data)
