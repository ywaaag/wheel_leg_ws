"""Episode-boundary Jump curriculum driven by actual Jump control seconds.

No iteration-based advancement, hidden height command, extra evaluator or DR
mutation. A cohort is explicitly fixed to one plant/DR identity per controller.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import numpy as np

VERSION = "jump_landing_and_active_time_course_v1"


@dataclass(frozen=True)
class Level:
    name: str
    active_time_share: float
    rise_m: float
    clearance_m: float
    air_leg_m: float


LEVELS = (
    Level("landing_and_first_jump", 0.50, 0.05, 0.02, 0.18),
    Level("complete_low_jump", 0.75, 0.05, 0.02, 0.18),
    Level("rise_10cm", 0.85, 0.10, 0.04, 0.17),
    Level("rise_15cm", 0.85, 0.15, 0.06, 0.16),
    Level("rise_20cm_research_target", 0.85, 0.20, 0.08, 0.16),
)


@dataclass(frozen=True)
class CourseConfig:
    window_episodes: int = 32
    pass_fraction: float = 0.8
    maximum_fault_fraction: float = 0.05
    promote_windows: int = 2
    demote_windows: int = 2
    demote_below_fraction: float = 0.4
    landing_pass_fraction: float = 0.8
    active_horizon_s: float = 16.0
    landing_horizon_s: float = 2.5
    reservation_active_s: float = 1.5
    reservation_landing_s: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("finite curriculum parameters required")
        if any(
            type(getattr(self, name)) is not int
            for name in ("window_episodes", "promote_windows", "demote_windows")
        ):
            raise ValueError("integer statistical windows required")
        if min(self.window_episodes, self.promote_windows, self.demote_windows) < 1:
            raise ValueError("positive statistical windows required")
        if not 0 < self.demote_below_fraction < self.pass_fraction <= 1:
            raise ValueError("separate advancement and fallback thresholds required")
        if not 0 <= self.maximum_fault_fraction < 1 or not 0 < self.landing_pass_fraction <= 1:
            raise ValueError("valid physical quality fractions required")
        if (
            min(
                self.active_horizon_s,
                self.landing_horizon_s,
                self.reservation_active_s,
                self.reservation_landing_s,
            )
            <= 0
        ):
            raise ValueError("positive episode and reservation durations required")


class AdaptiveCourse:
    def __init__(self, *, seed, plant_id, DR_id, config=None):
        if not plant_id or not DR_id:
            raise ValueError("explicit plant/DR cohort identity required")
        self.config = config or CourseConfig()
        self.plant_id, self.DR_id = str(plant_id), str(DR_id)
        self.rng = np.random.default_rng(seed)
        self.level = self.generation = self.serial = 0
        self.pending = {}
        self.control_s = dict(ground_active_jump=0.0, landing_only=0.0)
        self.buffers = dict(ground_active_jump=[], landing_only=[])
        self.windows = dict(ground_active_jump=[], landing_only=[])
        self.history = []
        self.episodes = 0

    def next_episode(self):
        level, c = LEVELS[self.level], self.config
        effective = dict(self.control_s)
        for request in self.pending.values():
            if request["generation"] == self.generation:
                effective[request["kind"]] += request["reserved_jump_s"]
        total = sum(effective.values())
        # Deficit scheduling corrects actual student-control duration, instead
        # of counting ten seconds of upstream Motion as Jump training samples.
        active_deficit = level.active_time_share * (total + 1.0) - effective["ground_active_jump"]
        landing_deficit = (1 - level.active_time_share) * (total + 1.0) - effective["landing_only"]
        choose_active = active_deficit > landing_deficit
        if abs(active_deficit - landing_deficit) < 1e-12:
            choose_active = bool(self.rng.integers(2))
        kind = "ground_active_jump" if choose_active else "landing_only"
        self.serial += 1
        request = dict(
            request_id=self.serial,
            generation=self.generation,
            level=self.level,
            plant_id=self.plant_id,
            DR_id=self.DR_id,
            kind=kind,
            target_rise_m=level.rise_m,
            target_clearance_m=level.clearance_m,
            air_leg_m=level.air_leg_m,
            entry_height_m=0.34,
            leg_length_m=0.28,
            drop_m=0.0 if choose_active else float(self.rng.uniform(0.02, 0.05)),
            seconds=c.active_horizon_s if choose_active else c.landing_horizon_s,
            reserved_jump_s=c.reservation_active_s if choose_active else c.reservation_landing_s,
        )
        self.pending[self.serial] = copy.deepcopy(request)
        return request

    def record(self, request, *, jump_control_s, success, fault):
        identity = request.get("request_id")
        if identity not in self.pending or self.pending[identity] != request:
            raise ValueError("unknown, modified or already settled curriculum request")
        if not math.isfinite(jump_control_s) or jump_control_s < 0:
            raise ValueError("actual nonnegative Jump control seconds required")
        if success and (fault or jump_control_s <= 0):
            raise ValueError("success requires effective Jump control and no fault")
        self.pending.pop(identity)
        self.episodes += 1
        if request["generation"] != self.generation:
            return dict(stale_generation=True, transition=None)
        kind = request["kind"]
        self.control_s[kind] += jump_control_s
        self.buffers[kind].append(dict(success=bool(success), fault=bool(fault)))
        c = self.config
        transition = None
        if len(self.buffers[kind]) == c.window_episodes:
            rows = self.buffers[kind]
            window = dict(
                success_fraction=sum(x["success"] for x in rows) / len(rows),
                fault_fraction=sum(x["fault"] for x in rows) / len(rows),
                episodes=len(rows),
            )
            self.windows[kind].append(window)
            self.windows[kind] = self.windows[kind][-max(c.promote_windows, c.demote_windows) :]
            self.buffers[kind] = []
            active = self.windows["ground_active_jump"]
            landing = self.windows["landing_only"]
            if (
                len(active) >= c.demote_windows
                and self.level > 0
                and all(
                    w["success_fraction"] < c.demote_below_fraction
                    for w in active[-c.demote_windows :]
                )
            ):
                transition = self._move(
                    self.level - 1, "two_complete_active_windows_below_fallback"
                )
            elif (
                len(active) >= c.promote_windows
                and landing
                and self.level < len(LEVELS) - 1
                and all(
                    w["success_fraction"] >= c.pass_fraction
                    and w["fault_fraction"] < c.maximum_fault_fraction
                    for w in active[-c.promote_windows :]
                )
                and landing[-1]["success_fraction"] >= c.landing_pass_fraction
                and landing[-1]["fault_fraction"] < c.maximum_fault_fraction
            ):
                transition = self._move(self.level + 1, "active_complete_and_landing_retained")
        return dict(stale_generation=False, transition=transition)

    def _move(self, level, reason):
        event = dict(
            from_level=self.level, to_level=level, generation=self.generation, reason=reason
        )
        self.history.append(event)
        self.history = self.history[-100:]
        self.level = level
        self.generation += 1
        self.control_s = dict(ground_active_jump=0.0, landing_only=0.0)
        self.buffers = dict(ground_active_jump=[], landing_only=[])
        self.windows = dict(ground_active_jump=[], landing_only=[])
        # In-flight old requests remain valid for settlement; their results
        # cannot fill a new visit's statistical window or control quota.
        return event

    def metrics(self):
        active = self.windows["ground_active_jump"]
        landing = self.windows["landing_only"]
        total = sum(self.control_s.values())
        return dict(
            current_curriculum_stage=self.level,
            curriculum_generation=self.generation,
            jump_success_rate=active[-1]["success_fraction"] if active else None,
            landing_success_rate=landing[-1]["success_fraction"] if landing else None,
            observed_active_control_share=self.control_s["ground_active_jump"] / total
            if total
            else None,
            target_active_control_share=LEVELS[self.level].active_time_share,
            pending_episodes=len(self.pending),
            recorded_episodes=self.episodes,
        )

    def state_dict(self):
        return copy.deepcopy(
            dict(
                version=VERSION,
                config=asdict(self.config),
                plant_id=self.plant_id,
                DR_id=self.DR_id,
                levels=[asdict(x) for x in LEVELS],
                rng=self.rng.bit_generator.state,
                level=self.level,
                generation=self.generation,
                serial=self.serial,
                pending=list(self.pending.values()),
                control_s=self.control_s,
                buffers=self.buffers,
                windows=self.windows,
                history=self.history,
                episodes=self.episodes,
            )
        )

    @classmethod
    def restore(cls, value):
        data = copy.deepcopy(value)
        if data.pop("version") != VERSION or data.pop("levels") != [asdict(x) for x in LEVELS]:
            raise ValueError("curriculum version/level identity mismatch")
        rng_state = data.pop("rng")
        pending = data.pop("pending")
        restored = cls(
            seed=0,
            plant_id=data.pop("plant_id"),
            DR_id=data.pop("DR_id"),
            config=CourseConfig(**data.pop("config")),
        )
        restored.__dict__.update(data)
        restored.pending = {x["request_id"]: x for x in pending}
        restored.rng.bit_generator.state = rng_state
        return restored
