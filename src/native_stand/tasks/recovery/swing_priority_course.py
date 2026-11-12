"""Swing-first own-control course; nominal/DR and height are separate axes."""

import numpy as np

from .cubic_course import GROUPS
from .cubic_course import RecoveryCourse as Base

VERSION = "recovery_Swing_priority_own_hold_v2"
FRACTIONS = (0.15, 0.075, 0.075, 0.70)


def standing_context(episode, seed):
    # Every height/schedule occurs in consecutive nominal/stress pairs.
    bucket = (episode // 2 + seed) % 6
    return dict(height_m=[0.30, 0.32, 0.34, 0.36, 0.38, 0.34][bucket], height_dynamic=bucket == 5)


class RecoveryCourse(Base):
    def choose(self):
        total = sum(v["student_seconds"] for v in self.counts.values())
        group = (
            3
            if total == 0
            else int(
                np.argmax(
                    np.asarray(FRACTIONS) * (total + 1)
                    - [self.counts[g]["student_seconds"] for g in GROUPS]
                )
            )
        )
        nominal = self.episodes % 2 == 0
        spec = dict(
            group=GROUPS[group],
            standing=group != 3,
            prefix_s=0.0,
            pulse=group in (1, 2),
            DR_strength=0.0 if nominal else 0.25,
            nominal_anchor=nominal,
        )
        spec.update(
            standing_context(self.episodes, self.seed)
            if group != 3
            else dict(height_m=0.34, height_dynamic=False)
        )
        return spec

    def __init__(self, seed=87001):
        super().__init__(seed)
        self.seed = seed

    def record(
        self,
        group,
        *,
        student_seconds,
        teacher_seconds,
        passed,
        nominal=None,
        physical_quality=None,
    ):
        if teacher_seconds > 1e-7:
            raise ValueError("Swing-first Recovery cannot count teacher-controlled time")
        row = self.counts[group]
        row["student_seconds"] += student_seconds
        row["episodes"] += 1
        row["passed"] += int(passed)
        self.episodes += 1
        # No standing or HF prerequisite can remove Swing exposure.
        # DR strength remains explicit .25, not an unvalidated auto-escalation.
