"""Recovery exposure course, explicit counted student time and nominal anchors."""

from __future__ import annotations

import numpy as np

VERSION = "recovery_cubic_R0_R1_R2_recoverable_v2"
GROUPS = ("standing_review", "perturbed_standing", "dynamic_catch", "full_Swing")
FRACTIONS = ((0.5, 0.2, 0.3, 0.0), (0.2, 0.1, 0.5, 0.2), (0.1, 0.1, 0.2, 0.6))


class RecoveryCourse:
    def __init__(self, seed=87001):
        self.stage = 0
        self.episodes = 0
        self.counts = {
            g: dict(student_seconds=0.0, teacher_seconds=0.0, episodes=0, passed=0) for g in GROUPS
        }
        self.pending = {g: [] for g in GROUPS}
        self.progression_frozen = False
        self.windows = []
        self.pass_streak = 0
        self.failure_streak = 0
        self.rng = np.random.default_rng(seed)

    def choose(self):
        weights = np.asarray(FRACTIONS[self.stage])
        durations = np.array([23.0, 23.0, 21.0, 23.0])
        available = weights > 0
        total = sum(v["student_seconds"] for v in self.counts.values())
        if total:
            deficit = weights * (total + 23) - np.array(
                [self.counts[g]["student_seconds"] for g in GROUPS]
            )
            group = int(np.argmax(np.where(available, deficit / durations, -np.inf)))
        else:
            group = int(self.rng.choice(4, p=weights))
        strength = [0.25, 0.5, 1.0][self.stage]
        # Nominal anchors remain deterministic every other episode. This
        # schedule does not advance the separate physical sampling RNG.
        nominal = self.episodes % 2 == 0
        return dict(
            group=GROUPS[group],
            standing=group != 3,
            prefix_s=2.0 if group == 2 else 0.0,
            pulse=group in (1, 2),
            DR_strength=0.0 if nominal else strength,
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
            return None  # Exposure-only records are not gate evidence.
        self.pending[group].append(
            dict(passed=passed is True, nominal=nominal, quality=physical_quality is True)
        )
        active = [g for g, w in zip(GROUPS, FRACTIONS[self.stage], strict=False) if w > 0]
        if any(
            len(self.pending[g]) < 32
            or not any(r["nominal"] for r in self.pending[g])
            or not any(not r["nominal"] for r in self.pending[g])
            for g in active
        ):
            return None
        groups = {}
        for g in active:
            rows = self.pending[g]
            nom = [r for r in rows if r["nominal"]]
            stress = [r for r in rows if not r["nominal"]]
            groups[g] = dict(
                complete_student_episodes=len(rows),
                nominal_pass_fraction=np.mean([r["passed"] for r in nom]).item(),
                stress_pass_fraction=np.mean([r["passed"] for r in stress]).item(),
                physical_quality_retained=bool([r for r in nom if r["passed"]])
                and all(r["quality"] for r in nom if r["passed"]),
            )
        result = self.complete_window(groups)
        self.pending = {g: [] for g in GROUPS}
        return result

    def complete_window(self, groups):
        # Group records must be complete measured episodes, not worker update
        # counts or teacher-controlled samples. Unknown quality never passes.
        qualified = all(row["complete_student_episodes"] >= 32 for row in groups.values())
        if not qualified:
            return dict(promoted=False, reason="Insufficient complete student episodes")
        nominal = min(row["nominal_pass_fraction"] for row in groups.values())
        stress = min(row["stress_pass_fraction"] for row in groups.values())
        good = (
            nominal >= 0.9
            and stress >= 0.75
            and all(row["physical_quality_retained"] is True for row in groups.values())
        )
        self.pass_streak = self.pass_streak + 1 if good else 0
        self.failure_streak = self.failure_streak + 1 if nominal < 0.7 else 0
        # Freeze only while the measured failure streak persists. Later two
        # good windows must be allowed to resume curriculum progression.
        self.progression_frozen = self.failure_streak >= 2
        promoted = self.pass_streak >= 2 and self.stage < 2 and not self.progression_frozen
        if promoted:
            self.stage += 1
            self.pass_streak = 0
        self.windows.append(dict(groups=groups, good=good, stage=self.stage))
        return dict(
            promoted=promoted, stop_increasing_difficulty=self.failure_streak >= 2, stage=self.stage
        )
