"""Standing-prepared Recovery owns the complete rise and ten-second hold.

No Motion prefix or downstream controller. Standing retention is present at all
stages; stage0 learns own standing first, then Swing is introduced. HF is
reported separately and is not the standing-to-Swing promotion gate.
"""

import numpy as np

from .cubic_course import GROUPS
from .cubic_course import RecoveryCourse as Base

VERSION = "recovery_standalone_standing_then_rise_course_v1"
FRACTIONS = ((0.6, 0.2, 0.2, 0.0), (0.3, 0.2, 0.1, 0.4), (0.2, 0.1, 0.1, 0.6))


class RecoveryCourse(Base):
    def choose(self):
        weights = np.asarray(FRACTIONS[self.stage])
        total = sum(v["student_seconds"] for v in self.counts.values())
        group = (
            int(
                np.argmax(
                    np.where(
                        weights > 0,
                        weights * (total + 1) - [self.counts[g]["student_seconds"] for g in GROUPS],
                        -np.inf,
                    )
                )
            )
            if total
            else int(self.rng.choice(4, p=weights))
        )
        nominal = self.episodes % 2 == 0
        return dict(
            group=GROUPS[group],
            standing=group != 3,
            prefix_s=0.0,
            pulse=group in (1, 2),
            DR_strength=0.0 if nominal else [0.25, 0.5, 1.0][self.stage],
            nominal_anchor=nominal,
        )

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
            raise ValueError("Standalone course cannot count Motion-controlled time")
        row = self.counts[group]
        row["student_seconds"] += student_seconds
        row["episodes"] += 1
        row["passed"] += int(passed)
        self.episodes += 1
        if nominal is None:
            return
        self.pending[group].append(
            dict(passed=passed is True, nominal=nominal, HF_diagnostic=physical_quality is True)
        )
        active = [g for g, w in zip(GROUPS, FRACTIONS[self.stage], strict=False) if w > 0]
        if any(
            len(self.pending[g]) < 32
            or not any(x["nominal"] for x in self.pending[g])
            or not any(not x["nominal"] for x in self.pending[g])
            for g in active
        ):
            return
        rates = {
            g: dict(
                nominal=float(np.mean([x["passed"] for x in self.pending[g] if x["nominal"]])),
                stress=float(np.mean([x["passed"] for x in self.pending[g] if not x["nominal"]])),
                HF_diagnostic=float(np.mean([x["HF_diagnostic"] for x in self.pending[g]])),
            )
            for g in active
        }
        good = (
            min(x["nominal"] for x in rates.values()) >= 0.9
            and min(x["stress"] for x in rates.values()) >= 0.75
        )
        self.pass_streak = self.pass_streak + 1 if good else 0
        if self.pass_streak >= 2 and self.stage < 2:
            self.stage += 1
            self.pass_streak = 0
        self.windows.append(
            dict(
                groups=rates,
                good=good,
                stage=self.stage,
                success="Recovery own entry and ten-second hold",
                Motion_control=False,
            )
        )
        self.pending = {g: [] for g in GROUPS}
