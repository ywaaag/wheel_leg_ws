"""Jump post-update evidence and publication; rollout/math stay in common adapter."""

import copy
import os
from dataclasses import asdict
from pathlib import Path

from native_stand.artifacts.io import json_digest, write_json
from native_stand.simulation.engine import GenericPPOAdapter
from native_stand.tasks.jump.unified import VERSION


class JumpAdapter(GenericPPOAdapter):
    def build(self, context, stage, resume_from):
        self.pending = None
        super().build(context, stage, resume_from)
        from native_stand.artifacts.io import file_sha256
        from native_stand.artifacts.model_refs import ModelReference

        for env in self.env.envs:
            binding = env.plant.binding
            for path, sha in binding["sources"].items():
                context.refs.register(
                    ModelReference(sha, path, ("control",), context.run_dir.name, kind="C_source")
                )
            context.refs.register(
                ModelReference(
                    binding["sha256"],
                    binding["path"],
                    ("control",),
                    context.run_dir.name,
                    kind="C_library",
                )
            )
            manifest = str(Path(binding["path"]).with_suffix(".json"))
            context.refs.register(
                ModelReference(
                    file_sha256(manifest),
                    manifest,
                    ("control",),
                    context.run_dir.name,
                    kind="C_build",
                )
            )
        self.domain = json_digest(
            dict(
                contract=context.job.assets["jump_contract_data"],
                schema=context.job.assets["jump_critic_schema"],
                execution=context.job.assets["jump_bindings"],
                course=context.job.assets["jump_course"],
            )
        )
        self.candidates = context.refs.candidate_index(context.run_dir.name) or dict(
            version=VERSION,
            comparison_sha256=self.domain,
            label="训练统计候选",
            independent_certificate=False,
            extra_simulation_steps=0,
            candidates_by_objective={"balance": {}, "landing_only": {}, "active_jump": {}},
            best_candidate=None,
            best_standing_candidate=None,
        )
        if self.candidates["comparison_sha256"] != self.domain:
            raise ValueError("Jump候选比较域不同")
        for pool in self.candidates["candidates_by_objective"].values():
            for row in pool.values():
                if file_sha256(row["checkpoint"]) != row["sha256"]:
                    raise ValueError("Jump候选文件身份不同")
        write_json(context.run_dir / "training_candidates.json", self.candidates)
        context.store.update(writer_pid=os.getpid(), training_candidates=self.candidates)
        self.cursor = 0
        payload = getattr(resume_from, "payload", resume_from) or {}
        if resume_from is not None:
            self.cursor = payload["jump_consumed_episode"]
            self.pending = payload.get("jump_pending_publication")
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
        if context.updates_used % 100:
            return
        rows = [r for r in self.env.completed if r["id"] > self.cursor]
        self.pending = dict(
            start_episode=self.cursor + 1,
            end_episode=self.env.serial,
            end_update=context.updates_used,
            rows=copy.deepcopy(rows),
            comparison_sha256=self.domain,
            selection=asdict(self.env.envs[0].course.config),
        )
        self.cursor = self.env.serial
        # Pending publication now owns the consumed summaries; unconsumed rows
        # remain in the batch until this exact window has been captured.
        self.env.completed[:] = [r for r in self.env.completed if r["id"] > self.cursor]

    def capture(self, context, counters):
        value = super().capture(context, counters)
        value.update(
            jump_consumed_episode=self.cursor,
            jump_pending_publication=copy.deepcopy(self.pending),
            training_candidates=copy.deepcopy(self.candidates),
        )
        return value

    def after_save(self, context, reference):
        if self.pending is None:
            return
        if self.pending["end_update"] != reference["update"]:
            raise ValueError("Jump统计窗口/保存点不同")
        current = context.refs.candidate_index(context.run_dir.name) or self.candidates
        last = current.get("last_review")
        if last and last.get("window") == self.pending:
            self.candidates = current
        else:
            self.candidates = consider(current, self.pending, reference)
            self.candidates = context.refs.publish_candidates(context.run_dir.name, self.candidates)
        write_json(context.run_dir / "training_candidates.json", self.candidates)
        context.store.update(writer_pid=os.getpid(), training_candidates=self.candidates)
        self.pending = None


def consider(index, window, reference):
    result = copy.deepcopy(index)
    config = window["selection"]
    groups = {
        tuple(r["request"][key] for key in ("plant_id", "DR_id", "level", "generation"))
        for r in window["rows"]
    }
    for group in groups:
        grouped = [
            r
            for r in window["rows"]
            if tuple(r["request"][key] for key in ("plant_id", "DR_id", "level", "generation"))
            == group
        ]
        for kind, objective in [
            ("landing_only", "landing_only"),
            ("ground_active_jump", "active_jump"),
            (None, "balance"),
        ]:
            rows = [r for r in grouped if kind is None or r["request"]["kind"] == kind]
            if len(rows) < config["window_episodes"]:
                continue
            if objective == "balance":
                measured = [r for r in rows if r["balance_seconds"] >= 1.0]
                if len(measured) < config["window_episodes"]:
                    continue
                fraction = sum(
                    r["balance_good_seconds"] / r["balance_seconds"] for r in measured
                ) / len(measured)
                rows = measured
            else:
                fraction = sum(r["success"] for r in rows) / len(rows)
            fault = sum(r["fault"] for r in rows) / len(rows)
            threshold = (
                config["pass_fraction"]
                if objective == "active_jump"
                else config["landing_pass_fraction"]
            )
            if fraction < threshold or fault >= config["maximum_fault_fraction"]:
                continue
            rank = [fraction, -fault]
            key = json_digest(group)
            pool = result["candidates_by_objective"][objective]
            if key not in pool or rank > pool[key]["rank"]:
                pool[key] = dict(
                    reference,
                    rank=rank,
                    window=window,
                    cohort=list(group),
                    label="训练统计候选",
                    independent_certificate=False,
                    extra_simulation_steps=0,
                )
    result["last_review"] = dict(
        through_update=reference["update"], checkpoint=reference, window=window
    )
    return result
