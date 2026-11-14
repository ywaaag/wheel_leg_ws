"""Recovery uses the common PPO loop; only ended training episodes drive nominees."""

from __future__ import annotations

import copy
import os
from contextlib import nullcontext

from native_stand.artifacts.io import file_sha256, json_digest, write_json
from native_stand.simulation.engine import GenericPPOAdapter


class RecoveryAdapter(GenericPPOAdapter):
    def build(self, context, stage, resume_from):
        super().build(context, stage, resume_from)
        self.selection = context.job.diagnostics["training_candidate_selection"]
        self.leg_driven = context.job.reward.version == "recovery_leg_driven_low_wheel_v1"
        payload0 = resume_from.payload if resume_from is not None else {}
        self.leg_review_history = copy.deepcopy(payload0.get("leg_driven_review_history", []))
        recipe = context.job.as_run_record()
        self.domain = json_digest(
            dict(
                contracts={
                    k: recipe[k]
                    for k in (
                        "model",
                        "observation",
                        "action",
                        "control",
                        "reward",
                        "curriculum",
                        "ppo",
                    )
                },
                course=recipe["assets"].get("recovery_standalone_course"),
                selection=self.selection,
                dependencies={
                    key: recipe["assets"][key]["sha256"]
                    for key in (
                        "recovery_training_contract",
                        "recovery_isaac_runtime",
                        "qualified_physical_BC_anchor",
                        "old_Recovery_action_prior",
                    )
                    if key in recipe["assets"]
                },
            )
        )
        payload = resume_from.payload if resume_from is not None else {}
        self.legacy_unscoped = copy.deepcopy(
            payload.get("legacy_unscoped_completed_statistics", [])
        )
        if resume_from is not None and "recovery_episode_cursor" not in payload:
            # The old schema has no window boundaries. Preserve its ended rows
            # as UNKNOWN history, never silently assign them to a new window.
            self.legacy_unscoped = copy.deepcopy(self.env.env.completed)
            self.cursor = self.env.env.completed_count
        else:
            self.cursor = payload.get("recovery_episode_cursor", 0)
        self.window = copy.deepcopy(payload.get("recovery_training_window", {}))
        self.pending = copy.deepcopy(payload.get("recovery_pending_publication"))
        indexed = context.refs.candidate_index(context.run_dir.name)
        restored = payload.get("training_candidates")
        self.candidates = (
            indexed
            or copy.deepcopy(restored)
            or dict(
                version="recovery_own_hold_training_candidates_v1",
                comparison_sha256=self.domain,
                label="训练统计候选",
                independent_certificate=False,
                extra_simulation_steps=0,
                candidates_by_objective={"Swing": {}, "standing": {}},
                best_candidate=None,
                best_standing_candidate=None,
            )
        )
        if self.candidates["comparison_sha256"] != self.domain:
            raise ValueError("Recovery同run候选比较域不一致")
        for pool in self.candidates["candidates_by_objective"].values():
            for row in pool.values():
                if file_sha256(row["checkpoint"]) != row["sha256"]:
                    raise ValueError("Recovery候选身份不同")
        if (
            indexed is None
            and restored is not None
            and self.pending is None
            and any(self.candidates["candidates_by_objective"].values())
        ):
            # A relocated checkpoint carries its mapped candidate paths. Acquire
            # their references before any later checkpoint rotation can run.
            self.candidates = context.refs.publish_candidates(context.run_dir.name, self.candidates)
        if self.pending is not None:
            self.after_save(
                context,
                dict(
                    checkpoint=str(resume_from.path),
                    sha256=resume_from.sha256,
                    update=self.pending["end_update"],
                ),
            )

    def after_update(self, context, index, metrics):
        timer = getattr(self, "measurement", None)
        with timer.section("course_statistics", synchronize=False) if timer else nullcontext():
            return self._consume_completed(context, index, metrics)

    def _consume_completed(self, context, index, metrics):
        env = self.env.env
        first = env.completed_count - len(env.completed)
        if self.cursor < first or self.cursor > env.completed_count:
            raise ValueError("Recovery结束统计游标越界；不能丢失行为证据")
        for row in env.completed[self.cursor - first :]:
            import math

            if (
                any(
                    not math.isfinite(row[k]) or row[k] < 0
                    for k in ("student_seconds", "teacher_seconds", "DR_strength")
                )
                or row["steps"] <= 0
            ):
                raise ValueError("Recovery训练回合统计损坏")
            objective = "Swing" if row["group"] == "full_Swing" else "standing"
            key = json_digest(
                [
                    objective,
                    row["group"],
                    row["nominal_anchor"],
                    row["command_height_m"],
                    row["height_dynamic"],
                    row["DR_strength"],
                ]
            )
            if self.leg_driven:
                key = json_digest(
                    [key, row["leg_driven"]["course_level"], row["leg_driven"]["initial_phase"]]
                )
            aggregate = self.window.setdefault(
                key,
                dict(
                    objective=objective,
                    ended=0,
                    student_success=0,
                    student_seconds=0.0,
                    teacher_seconds=0.0,
                    quality=0,
                    condition={
                        k: row[k]
                        for k in (
                            "group",
                            "nominal_anchor",
                            "command_height_m",
                            "height_dynamic",
                            "DR_strength",
                        )
                    },
                ),
            )
            if self.leg_driven:
                info = row["leg_driven"]
                aggregate["condition"].update(
                    course_level=info["course_level"], initial_phase=info["initial_phase"]
                )
                aggregate["phase_progress_sum"] = aggregate.get("phase_progress_sum", 0.0) + max(
                    0, info["final_phase"] - info["initial_phase"]
                )
                aggregate["wheel_rms_sum"] = aggregate.get("wheel_rms_sum", 0.0) + max(
                    info["wheel_rms"]
                )
                aggregate["sequence_valid"] = aggregate.get("sequence_valid", 0) + int(
                    info["sequence_valid"]
                )
                aggregate["wheel_peak_max"] = max(
                    aggregate.get("wheel_peak_max", 0.0), max(info["wheel_peak"])
                )
                aggregate["wheel_rms_max"] = max(
                    aggregate.get("wheel_rms_max", 0.0), max(info["wheel_rms"])
                )
                aggregate["tilt_peak_max_rad"] = max(
                    aggregate.get("tilt_peak_max_rad", 0.0), info["tilt_peak_rad"]
                )
            aggregate["ended"] += 1
            aggregate["student_success"] += int(
                row["success"] is True and row["teacher_seconds"] == 0
            )
            aggregate["student_seconds"] += row["student_seconds"]
            aggregate["teacher_seconds"] += row["teacher_seconds"]
            aggregate["quality"] += int(
                row["leg_driven"]["quality"] if self.leg_driven else row["physical_quality"] is True
            )
        self.cursor = env.completed_count
        if context.updates_used % self.selection["every_updates"] == 0:
            self.pending = dict(
                end_update=context.updates_used,
                episode_cursor=self.cursor,
                comparison_sha256=self.domain,
                rows=copy.deepcopy(self.window),
            )
            if self.leg_driven:
                from native_stand.tasks.recovery.leg_driven import review_window

                level = env.courses[0].stage
                record, advance, plateau = review_window(
                    list(self.window.values()), level, self.leg_review_history
                )
                record["through_update"] = context.updates_used
                self.leg_review_history.append(record)
                metrics["leg_driven_review"] = record
                write_json(
                    context.run_dir / "leg_driven_course.json",
                    dict(history=self.leg_review_history, level=level + int(advance)),
                )
                if advance:
                    for course in env.courses:
                        course.stage = level + 1
                if plateau:
                    from native_stand.experiments.state import write_stop_request

                    write_stop_request(
                        context.run_dir,
                        reason="leg_driven_five_windows_without_improvement",
                        requester="Recovery course",
                    )
            self.window = {}

    def capture(self, context, counters):
        result = super().capture(context, counters)
        result.update(
            legacy_unscoped_completed_statistics=copy.deepcopy(self.legacy_unscoped),
            recovery_episode_cursor=self.cursor,
            recovery_training_window=copy.deepcopy(self.window),
            recovery_pending_publication=copy.deepcopy(self.pending),
            training_candidates=copy.deepcopy(self.candidates),
            leg_driven_review_history=copy.deepcopy(self.leg_review_history),
        )
        return result

    def after_save(self, context, reference):
        timer = getattr(self, "measurement", None)
        with (
            timer.section("candidate_references_views", synchronize=False)
            if timer
            else nullcontext()
        ):
            return self._publish_candidates(context, reference)

    def _publish_candidates(self, context, reference):
        if self.pending is None:
            return
        if reference["update"] != self.pending["end_update"]:
            raise ValueError("Recovery窗口与checkpoint更新不同")
        current = context.refs.candidate_index(context.run_dir.name) or self.candidates
        if current.get("last_review", {}).get("window") != self.pending:
            result = copy.deepcopy(current)
            for key, row in self.pending["rows"].items():
                if (
                    row["ended"] < self.selection["minimum_ended"]
                    or not row["student_success"]
                    or row["teacher_seconds"]
                ):
                    continue
                if self.leg_driven and (
                    row["objective"] != "Swing"
                    or row["student_success"] / row["ended"] < 0.95
                    or row["quality"] / row["ended"] < 0.9
                    or row.get("sequence_valid", 0) / row["ended"] < 0.98
                ):
                    continue
                rank = [row["student_success"] / row["ended"], row["quality"] / row["ended"]]
                if self.leg_driven:
                    rank = [row["condition"]["course_level"], *rank]
                pool = result["candidates_by_objective"][row["objective"]]
                if key not in pool or rank > pool[key]["evidence"]["rank"]:
                    pool[key] = dict(
                        reference,
                        objective=row["objective"],
                        label="训练统计候选",
                        independent_certificate=False,
                        evidence=dict(row, rank=rank, window=self.pending),
                    )
            for objective, field in [
                ("Swing", "best_candidate"),
                ("standing", "best_standing_candidate"),
            ]:
                pool = result["candidates_by_objective"][objective]
                result[field] = (
                    max(pool.values(), key=lambda x: x["evidence"]["rank"]) if pool else None
                )
            result["last_review"] = dict(
                through_update=self.pending["end_update"], window=self.pending, checkpoint=reference
            )
            current = context.refs.publish_candidates(context.run_dir.name, result)
        self.candidates = current
        write_json(context.run_dir / "training_candidates.json", current)
        context.store.update(writer_pid=os.getpid(), training_candidates=current)
        self.pending = None
