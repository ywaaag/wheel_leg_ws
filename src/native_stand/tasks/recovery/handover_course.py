"""Swing exposure from the first reset; downstream Motion owns stability."""

import numpy as np

from .cubic_course import GROUPS
from .cubic_course import RecoveryCourse as ExposureCourse

VERSION = "recovery_Swing_Motion_handover_course_v1"
FRACTIONS = ((0.1, 0.1, 0.2, 0.6), (0.1, 0.1, 0.1, 0.7), (0.05, 0.05, 0.1, 0.8))


class RecoveryCourse(ExposureCourse):
    def choose(self):
        weights = np.asarray(FRACTIONS[self.stage])
        total = sum(v["student_seconds"] for v in self.counts.values())
        if total:
            # Balance MEASURED Recovery-controlled seconds. Dividing by the
            # downstream Motion episode duration overselects short Recovery
            # stand entries and starves Swing during Actor updates.
            deficit = weights * (total + 1.0) - np.array(
                [self.counts[g]["student_seconds"] for g in GROUPS]
            )
            group = int(np.argmax(deficit))
        else:
            group = int(self.rng.choice(4, p=weights))
        nominal = self.episodes % 2 == 0
        return dict(
            group=GROUPS[group],
            standing=group != 3,
            prefix_s=2.0 if group == 2 else 0.0,
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
        row = self.counts[group]
        row["student_seconds"] += student_seconds
        row["teacher_seconds"] += teacher_seconds
        row["episodes"] += 1
        row["passed"] += int(passed)
        self.episodes += 1
        if nominal is None:
            return
        self.pending[group].append(
            dict(passed=passed is True, nominal=nominal, quality=physical_quality is True)
        )
        if any(
            len(self.pending[g]) < 32
            or not any(r["nominal"] for r in self.pending[g])
            or not any(not r["nominal"] for r in self.pending[g])
            for g in GROUPS
        ):
            return
        groups = {}
        for g in GROUPS:
            rows = self.pending[g]
            nom = [r for r in rows if r["nominal"]]
            stress = [r for r in rows if not r["nominal"]]
            groups[g] = dict(
                complete_student_episodes=len(rows),
                nominal_pass_fraction=float(np.mean([r["passed"] for r in nom])),
                stress_pass_fraction=float(np.mean([r["passed"] for r in stress])),
                HF_quality_diagnostic_fraction=float(np.mean([r["quality"] for r in rows])),
            )
        result = self.complete_window(groups)
        self.pending = {g: [] for g in GROUPS}
        return result

    def complete_window(self, groups):
        nominal = min(r["nominal_pass_fraction"] for r in groups.values())
        stress = min(r["stress_pass_fraction"] for r in groups.values())
        good = nominal >= 0.9 and stress >= 0.75
        self.pass_streak = self.pass_streak + 1 if good else 0
        self.failure_streak = self.failure_streak + 1 if nominal < 0.7 else 0
        self.progression_frozen = self.failure_streak >= 2
        promoted = self.pass_streak >= 2 and self.stage < 2
        if promoted:
            self.stage += 1
            self.pass_streak = 0
        self.windows.append(
            dict(
                groups=groups,
                good=good,
                stage=self.stage,
                success_definition="Recovery entry + frozen Motion ten-second confirmation",
                HF_gate=False,
            )
        )
        return dict(promoted=promoted, stage=self.stage, Swing_always_present=True)
