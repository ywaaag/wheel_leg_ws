"""run worker：执行阶段、回报实际update、写终态。

约定：
- 配置只来自run.json里的解析后记录，不再读原始配置文件；
- 预算账本是消耗唯一权威，恢复不重置已用额度；
- 阶段之间用上一阶段的最终完整状态接续，不重新随机初始化；
- 任何结束路径都保存最新完整状态一次。
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from native_stand.artifacts import evaluation as evaluation_index
from native_stand.artifacts import source_identity
from native_stand.artifacts.checkpoint import save_full
from native_stand.artifacts.io import file_sha256, new_dir, read_json, write_json
from native_stand.artifacts.metrics import MetricsWriter
from native_stand.artifacts.model_refs import ModelIndex, ModelReference, register_model
from native_stand.artifacts.report import write_report
from native_stand.artifacts.retention import (
    SavePolicy,
    disk_usage,
    prune,
    read_inventory,
    write_inventory,
)
from native_stand.integrations import hapi
from native_stand.simulation import dispatch
from native_stand.simulation.engine import Backend, run_stage
from native_stand.tasks.motion.spec import semantics_record as motion_semantics

from .budget import Budget
from .config import JobConfig, job_from_record
from .state import RunStateStore

STOP_REQUESTED = False
MAX_OUTPUT_REFUSAL = "max_output_bytes"


def _signal(_number, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def build_task(job: JobConfig):
    """正式入口：与dispatch消费同一份registry定义。"""
    from native_stand.tasks.registry import build_task_from_registry

    return build_task_from_registry(job)


def resolve_backend(job: JobConfig) -> Backend:
    return dispatch.backend_for(job)


def _send(fd: int | None, payload: dict[str, Any]) -> None:
    if fd is None:
        return
    with contextlib.suppress(OSError):
        os.write(fd, (json.dumps(payload) + "\n").encode())


@dataclass
class RunContext:
    """引擎可用的服务集合；引擎不直接读写状态与预算文件。"""

    job: JobConfig
    task: Any
    run_dir: Path
    stage: str
    attempt_id: str
    budget: Budget
    store: RunStateStore
    metrics: MetricsWriter
    policy: SavePolicy
    output: Path
    refs: ModelIndex
    source_identity: dict[str, Any] = field(default_factory=dict)
    updates_used: int = 0
    samples_used: int = 0
    stage_updates_done: int = 0
    actor_changed: bool = False
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    protected: set[str] = field(default_factory=set)
    startup_fd: int | None = None
    startup_sent: bool = False
    claimed: bool = False
    resume_verified: bool = False
    save_failures: list[str] = field(default_factory=list)
    curriculum: dict[str, Any] = field(default_factory=dict)
    limit_reason: str | None = None

    # ---- 生命周期 --------------------------------------------------
    def startup_stage(self, stage: str) -> None:
        """接管状态写入权，并在真正开始后发一次握手。"""
        if not self.claimed:
            if self.resume_verified:
                self.store.reopen_for_resume(
                    pid=os.getpid(), attempt_id=self.attempt_id, stage=stage
                )
            self.store.claim(pid=os.getpid(), attempt_id=self.attempt_id, stage=stage)
            self.claimed = True
        self.stage = stage
        if self.startup_sent:
            return
        _send(
            self.startup_fd,
            {
                "status": "RUNNING",
                "stage": stage,
                "run": str(self.run_dir),
                "worker_pid": os.getpid(),
                "attempt_id": self.attempt_id,
                "first_update_verified": False,
            },
        )
        self.startup_sent = True

    def prepare_resume(self, source) -> None:
        """校验完整恢复点后，才允许在启动边界重新打开已结算终态。"""
        from native_stand.artifacts.handoff import validate_resume_identity

        validate_resume_identity(source, self.run_dir)
        state = self.store.read()
        counter = "new_run_updates" if "motion_course_state" in source.payload else "updates"
        updates = source.payload["counters"][counter]
        if updates != self.updates_used or state["updates_used"] != self.updates_used:
            raise ValueError("恢复点、运行状态与预算累计更新不一致")
        if state["status"] not in ("COMPLETED", "STOPPED_BOUNDED"):
            raise ValueError(f"状态{state['status']}不能按已结算完整状态恢复")
        self.resume_verified = True

    def verify_first_update(self) -> None:
        """只由真实完成的一次optimizer update调用。"""
        _send(
            self.startup_fd,
            {
                "status": "FIRST_UPDATE",
                "stage": self.stage,
                "run": str(self.run_dir),
                "worker_pid": os.getpid(),
                "attempt_id": self.attempt_id,
                "first_update_verified": True,
                "actual_updates": self.updates_used,
            },
        )

    def stop_requested(self) -> bool:
        if STOP_REQUESTED or (
            getattr(self, "stop_at_update", None) is not None
            and self.updates_used >= self.stop_at_update
        ):
            return True
        # stop_request.json是owner.stop的标准协议；文件请求与信号同等生效。
        return (self.run_dir / "stop_request.json").is_file()

    # ---- 计数与预算 ------------------------------------------------
    def charge(self, updates: int = 1) -> int:
        total = self.budget.charge(self.attempt_id, self.output, updates=updates)
        self.updates_used += updates
        self.stage_updates_done += updates
        self.samples_used += updates * self.job.runtime.samples_per_update
        self.store.update(
            writer_pid=os.getpid(),
            status="RUNNING",
            stage=self.stage,
            updates_used=self.updates_used,
            samples_used=self.samples_used,
        )
        if self.stage_updates_done == 1:
            self.verify_first_update()
        return total

    def stage_remaining(self, stage: str) -> int:
        limits = self.budget.limits()
        used = self.budget.used()
        return int(limits.get(stage, 0)) - int(used.get(stage, 0))

    def total_updates_before(self, stage: str) -> int:
        used = self.budget.used()
        return int(used.get("smoke", 0)) if stage == "formal" else 0

    def warmup_offset(self, stage: str) -> int:
        """本阶段第一条update在整run中的序号前缀。"""
        used = self.budget.used()
        return int(used.get("smoke", 0)) if stage == "formal" else 0

    def wall_budget_left(self) -> float | None:
        """累计已用执行时间来自已结算attempt；阶段/attempt切换不刷新额度。"""
        limit = self.job.limits.chain_wall_s
        if limit is None:
            return None
        used_before = sum(
            float(row.get("wall_s") or 0.0)
            for row in self.budget.rows().values()
            if row.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED")
        )
        return max(0.0, float(limit) - used_before - (time.monotonic() - self._started))

    def output_bytes_used(self) -> int:
        """只统计本run产物目录，不做全树扫描。"""
        total = disk_usage(self.run_dir / "checkpoints")
        for name in ("metrics.jsonl",):
            path = self.run_dir / name
            if path.is_file():
                total += path.stat().st_size
        for sub in ("evaluations", "traces"):
            directory = self.run_dir / sub
            if directory.is_dir():
                for entry in os.scandir(directory):
                    if entry.is_file():
                        total += entry.stat().st_size
        return total

    def output_bytes_left(self) -> int | None:
        limit = self.job.storage.max_output_bytes
        return None if limit is None else int(limit)

    # ---- 记录与保存 ------------------------------------------------
    def record(self, row: dict[str, Any]) -> bool:
        return self.metrics.write(row)

    def save_due(self, update: int, *, stage: str) -> bool:
        return self.policy.due(update)

    def save(
        self,
        state: dict[str, Any],
        *,
        update: int,
        actor_changed: bool = False,
        stage: str,
        final: bool = False,
    ) -> dict | None:
        """写完整恢复checkpoint；失败时保留上一个有效状态并记录。"""
        path = self.run_dir / "checkpoints" / f"model_update_{update}.pt"
        payload = dict(state)
        payload.setdefault("counters", {})
        names = (
            "source_anchor_identity",
            "recovery_initial_actor",
            "recovery_training_contract",
            "recovery_isaac_runtime",
            "qualified_physical_BC_anchor",
            "old_Recovery_action_prior",
        )
        payload["required_dependencies"] = {
            k: self.job.assets[k] for k in names if k in self.job.assets
        }
        for k in ("bank", "bank_metadata"):
            ref = self.job.assets.get("source_policy_retention", {}).get(k)
            if ref is not None:
                payload["required_dependencies"][k] = ref
        payload["update"] = update
        payload["stage"] = stage
        payload["final"] = bool(final)
        payload["interface_version"] = self.job.model.interface
        payload["action_version"] = self.job.action.version
        payload["source_identity"] = self.source_identity
        payload["run_identity"] = str(self.run_dir.resolve())
        from native_stand.artifacts.io import json_digest

        recipe = self.job.as_run_record()
        payload["config_contract_sha256"] = json_digest(
            {
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
            }
        )
        record = read_json(self.run_dir / "run.json", {})
        if record.get("logical_run_id") is not None:
            payload["logical_run_id"] = record["logical_run_id"]
        try:
            save_full(path, payload, exclusive=True)
        except BaseException as exc:
            self.save_failures.append(f"update {update}: {exc!r}")
            write_json(
                self.run_dir / "save_error.json",
                {
                    "update": update,
                    "error": repr(exc),
                    "last_valid": self.latest_reference(),
                    "updates_completed": self.updates_used,
                },
            )
            return None
        reference = {
            "checkpoint": str(path.resolve()),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "update": update,
            "stage": stage,
        }
        keys = {Path(row["checkpoint"]).resolve() for row in self.checkpoints}
        if path.resolve() not in keys:
            self.checkpoints.append(reference)
        else:
            self.checkpoints = [
                reference if Path(row["checkpoint"]).resolve() == path.resolve() else row
                for row in self.checkpoints
            ]
        self.actor_changed = self.actor_changed or actor_changed
        self.store.update(
            writer_pid=os.getpid(),
            status="RUNNING",
            stage=stage,
            updates_used=self.updates_used,
            samples_used=self.samples_used,
            latest_model=reference,
        )
        write_inventory(self.run_dir, self.checkpoints)
        self.refs.register_output(
            ModelReference(
                reference["sha256"], reference["checkpoint"], (), owner=str(self.run_dir.resolve())
            ),
            consumer=self.run_dir.name,
        )
        return reference

    def latest_reference(self) -> dict[str, Any] | None:
        rows = [row for row in self.checkpoints if Path(row["checkpoint"]).is_file()]
        if not rows:
            rows = [
                row for row in read_inventory(self.run_dir) if Path(row["checkpoint"]).is_file()
            ]
        if not rows:
            return None
        return max(rows, key=lambda row: int(row.get("update") or -1))

    def mark_stage_continuation(self, reference: dict[str, Any] | None, *, stage: str) -> None:
        """本阶段最终状态是下一阶段的恢复来源，登记为消费者。"""
        if not reference:
            return
        self.protected.add(reference["checkpoint"])
        register_model(
            self.refs.path,
            reference["checkpoint"],
            roles=("resume",),
            run=self.run_dir.name,
            reason=f"{stage}阶段最终完整状态",
        )
        self.store.update(writer_pid=os.getpid(), continuation_model=reference)

    def mark_best(self, reference: dict[str, Any]) -> None:
        self.best = dict(reference)
        self.protected.add(reference["checkpoint"])
        register_model(
            self.refs.path,
            reference["checkpoint"],
            roles=("best",),
            run=self.run_dir.name,
            reason="独立评估最佳",
        )
        self.store.update(writer_pid=os.getpid(), best_model=self.best)

    def mark_selected(self, reference: dict[str, Any]) -> None:
        self.selected = dict(reference)
        self.protected.add(reference["checkpoint"])
        register_model(
            self.refs.path,
            reference["checkpoint"],
            roles=("selected",),
            run=self.run_dir.name,
            reason="显式选中的交付候选",
        )
        self.store.update(writer_pid=os.getpid(), selected_model=self.selected)

    def register_evaluation(self, kind: str, path: str | Path, result: dict[str, Any]) -> None:
        entry = evaluation_index.register(
            self.run_dir / "evaluations", kind=kind, path=path, result=result
        )
        self.evaluations.append(entry)

    def curriculum_state(self) -> dict[str, Any]:
        return dict(self.curriculum)

    def restore_runtime_state(self, restored: dict[str, Any]) -> None:
        self.curriculum = dict(restored.get("curriculum") or {})
        counters = restored.get("counters") or {}
        self.stage_updates_done = int(counters.get("stage_updates", 0))

    def prune_checkpoints(self) -> dict[str, Any]:
        registered = self.refs.consumer_paths()
        for reference in (self.best, self.selected):
            if reference:
                registered.add(reference["checkpoint"])
        report = prune(
            self.run_dir,
            self.policy,
            protected=self.protected,
            registered=registered,
            index=self.refs,
        )
        self.checkpoints = [
            row for row in read_inventory(self.run_dir) if Path(row["checkpoint"]).is_file()
        ]
        return report

    best: dict[str, Any] | None = None
    selected: dict[str, Any] | None = None
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    _started: float = field(default_factory=time.monotonic)


def _new_context(
    run_dir: Path,
    stage: str,
    *,
    suffix: str | None = None,
    attempt_id: str | None = None,
) -> RunContext:
    """创建attempt上下文；reserve成功后的初始化失败只结算本次attempt。"""
    record = read_json(run_dir / "run.json")
    if record is None:
        raise FileNotFoundError(f"run.json不存在: {run_dir}")
    from native_stand.artifacts.runtime_paths import execution_job

    job = execution_job(job_from_record(record["config"]))
    task = build_task(job)
    store = RunStateStore(run_dir)
    budget = Budget(run_dir / "budget.json")
    attempt_id = attempt_id or uuid.uuid4().hex
    if suffix is None:
        suffix = f"{time.strftime('%H%M%S')}_{attempt_id[:6]}"
    output = run_dir / (f"{stage}_{suffix}" if suffix else stage)
    new_dir(output)
    budget.reserve(
        attempt_id,
        stage,
        output,
        finalize_only=stage == "distill"
        and job.supervision is not None
        and budget.used().get("distill", 0) == budget.limits().get("distill"),
    )
    try:
        return _finish_new_context(run_dir, job, task, stage, attempt_id, output, store, budget)
    except BaseException as exc:
        # reserve成功后的初始化失败：只结算本次attempt；不碰其他任务。
        import traceback

        with contextlib.suppress(Exception):
            budget.finalize(
                attempt_id,
                output,
                "FAILED",
                reason=f"init: {exc!r}",
                wall_s=0.0,
            )
        write_json(
            run_dir / "context_init_failure.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise


def _finish_new_context(run_dir, job, task, stage, attempt_id, output, store, budget):
    """reserve之后的上下文构建；任何异常由调用方结算本次attempt。"""
    metrics = MetricsWriter(
        run_dir / "metrics.jsonl",
        every_updates=job.storage.metrics_every_updates,
        tensorboard=job.storage.tensorboard and stage == "formal",
        stage=stage,
    )
    try:
        return _build_context(run_dir, job, task, stage, attempt_id, output, store, budget, metrics)
    except BaseException as exc:
        # 后续构造失败：关闭已创建的metrics；清理异常作附加信息，不覆盖原始错误。
        try:
            metrics.close()
        except Exception as close_exc:
            exc.close_error = repr(close_exc)  # 附加信息随异常链上抛
        raise


def _build_context(run_dir, job, task, stage, attempt_id, output, store, budget, metrics):
    policy = SavePolicy(
        every_updates=job.storage.checkpoint_every_updates,
        keep_recent=job.storage.keep_recent_checkpoints,
        milestones=tuple(job.storage.milestones),
    )
    context = RunContext(
        job=job,
        task=task,
        run_dir=run_dir,
        stage=stage,
        attempt_id=attempt_id,
        budget=budget,
        store=store,
        metrics=metrics,
        policy=policy,
        output=output,
        refs=ModelIndex(
            (read_json(run_dir / "run.json") or {}).get(
                "model_index", run_dir.parent / "models.json"
            )
        ),
        source_identity=source_identity.read_reference(run_dir) or {},
    )
    context.checkpoints = read_inventory(run_dir)
    settled = budget.used()
    baseline = int(settled.get("smoke", 0)) + int(settled.get("formal", 0))
    context.updates_used = baseline
    context.samples_used = baseline * job.runtime.samples_per_update
    previous = store.read()
    for key in ("continuation_model", "selected_model", "best_model"):
        entry = previous.get(key) or {}
        if entry.get("checkpoint"):
            context.protected.add(entry["checkpoint"])
    # 引用索引里的消费者模型同样受保护。
    context.protected |= context.refs.consumer_paths(excluding=("training_candidate",))
    # Motion专用语义记录；所有受支持任务都返回有效上下文（Recovery等不再得到None）。
    if job.task == "motion":
        write_json(run_dir / "source" / "motion_semantics.json", motion_semantics(task))
    return context


def _plan_stages(run_dir: Path, *, only: str | None = None) -> list[str]:
    """只读计划：读run记录与预算，决定真实要执行的阶段。"""
    record = read_json(run_dir / "run.json")
    if record is None:
        raise FileNotFoundError(f"run.json不存在: {run_dir}")
    job = job_from_record(record["config"])
    budget = Budget(run_dir / "budget.json")
    stages = dispatch.stage_order(job, budget, only=only)
    if not stages:
        raise ValueError("没有剩余可执行阶段；预算已耗尽，不会刷新额度")
    return stages


def _load_resume_source(path: str | Path):
    """统一加载入口：返回带身份的来源对象。"""
    from native_stand.artifacts.checkpoint import load_source

    return load_source(path)


def _load_resume_state(path: str | Path):
    """兼容旧调用点：返回统一的来源对象，不再返回裸 payload。"""
    return _load_resume_source(path)


def _execute_stage(context, stage: str, payload: dict[str, Any] | None):
    """计划、校验与实际执行只走同一个解析结果：execution_service。"""
    service = dispatch.execution_service(context.job)
    if service["service"] == "production_motion":
        module = importlib.import_module(service["module"])
        module.validate_job(context.job)
        return module.run(context, stage=stage, resume_from=payload)
    if service["service"] == "generic":
        path = (
            getattr(payload, "path", None)
            if payload is not None
            else context.job.initialization.checkpoint
        )
        if path is not None:
            register_model(
                context.refs.path,
                path,
                roles=("resume" if payload is not None else "initialization",),
                run=context.run_dir.name,
            )
        if context.job.backend == "isaac_recovery":
            from native_stand.artifacts.model_refs import register_dependencies

            names = (
                "recovery_initial_actor",
                "recovery_training_contract",
                "recovery_isaac_runtime",
                "qualified_physical_BC_anchor",
                "old_Recovery_action_prior",
                "recovery_legacy_reward",
                "recovery_leg_driven_geometry",
                "training_authorization",
                "training_allocation",
                "recovery_cubic_preparation",
            )
            register_dependencies(
                context.refs,
                consumer=context.run_dir.name,
                references={
                    name: context.job.assets[name] for name in names if name in context.job.assets
                },
            )
            from native_stand.artifacts.runtime_paths import execution_json

            runtime = execution_json(
                context.job.assets["recovery_isaac_runtime"]["path"], context.job.assets
            )
            register_dependencies(
                context.refs, consumer=context.run_dir.name, references=runtime["bindings"]
            )
        backend = resolve_backend(context.job)
        if service["module"] == "native_stand.simulation.jump_backend":
            from native_stand.artifacts.model_refs import ModelReference
            from native_stand.simulation.driver import run_stage as drive
            from native_stand.simulation.jump_adapter import JumpAdapter

            for name, ref in context.job.assets["jump_bindings"].items():
                if file_sha256(ref["path"]) != ref["sha256"]:
                    raise ValueError("Jump依赖身份不一致")
                context.refs.register(
                    ModelReference(
                        ref["sha256"],
                        str(Path(ref["path"]).resolve()),
                        ("teacher",)
                        if name == "teacher"
                        else ("distillation",)
                        if name.startswith("data_") or name in ("dataset", "BC_state")
                        else ("source",),
                        context.run_dir.name,
                        kind=name,
                    )
                )
            return drive(context, JumpAdapter(backend), stage=stage, resume_from=payload)
        if context.job.backend == "isaac_recovery" and (
            context.job.assets.get("recovery_standalone_course", {}).get("version")
            == "recovery_Swing_priority_own_hold_v2"
            or context.job.reward.version == "recovery_leg_driven_low_wheel_v1"
        ):
            from native_stand.simulation.driver import run_stage as drive
            from native_stand.simulation.recovery_adapter import RecoveryAdapter

            return drive(context, RecoveryAdapter(backend), stage=stage, resume_from=payload)
        return run_stage(context, backend=backend, stage=stage, resume_from=payload)
    raise dispatch.DispatchError(service["reason"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--handshake-fd", type=int, default=None)
    parser.add_argument("--stage", default=None, help="只执行指定阶段；默认按预算顺序执行")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--attempt-suffix", default=None)
    parser.add_argument(
        "--max-updates", type=int, default=None, help="本次恢复最多新增更新数；不刷新预算"
    )
    args = parser.parse_args(argv)
    if args.max_updates is not None and args.max_updates <= 0:
        parser.error("--max-updates必须为正")
    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)
    run_dir = args.run_dir.resolve()
    store = RunStateStore(run_dir)
    started = time.monotonic()
    failure: str | None = None
    status = "COMPLETED"
    context: RunContext | None = None
    stages: list[str] = []
    stage_started = time.monotonic()
    settled: set[str] = set()
    try:
        if run_dir.joinpath("run.json").is_file():
            record = read_json(run_dir / "run.json") or {}
            if (record.get("config") or {}).get("algorithm") == "distillation":
                from native_stand.experiments.distillation_job import run_distillation_worker

                return run_distillation_worker(
                    run_dir,
                    job_from_record(record["config"]),
                    handshake_fd=args.handshake_fd,
                    attempt_suffix=args.attempt_suffix,
                )
        # 先按预算决定阶段，再创建attempt：不把"smoke"写死成第一选择。
        planned = _plan_stages(run_dir, only=args.stage)
        context = _new_context(run_dir, planned[0], suffix=args.attempt_suffix)
        context.startup_fd = args.handshake_fd
        bound = None if args.max_updates is None else context.updates_used + args.max_updates
        context.stop_at_update = bound
        stages = dispatch.stage_order(context.job, context.budget, only=args.stage)
        resume_payload = (
            _load_resume_source(args.resume_checkpoint) if args.resume_checkpoint else None
        )
        if resume_payload is not None:
            context.prepare_resume(resume_payload)
        service = dispatch.execution_service(context.job)
        if service["service"] == "unsupported":
            raise dispatch.DispatchError(service["reason"])
        continuation_payload: dict[str, Any] | None = None
        for index, stage in enumerate(stages):
            stage_started = time.monotonic()
            if index > 0:
                context.metrics.close()
                context = _new_context(run_dir, stage, suffix=args.attempt_suffix)
                context.startup_fd = args.handshake_fd
                context.startup_sent = True
                context.stop_at_update = bound
            context.startup_stage(stage)
            payload_for_stage = resume_payload if index == 0 else continuation_payload
            result = _execute_stage(context, stage, payload_for_stage)
            resume_payload = None
            context.mark_stage_continuation(result.last_checkpoint, stage=stage)
            # 下一阶段必须从本阶段最终完整状态接续，不重新随机初始化。
            continuation_payload = (
                _load_resume_source(result.last_checkpoint["checkpoint"])
                if result.last_checkpoint
                else None
            )
            phase_status = "STOPPED_BOUNDED" if result.stopped else "COMPLETED"
            if result.limit and result.limit != "stop_requested":
                phase_status = "STOPPED_BOUNDED"
            context.budget.finalize(
                context.attempt_id,
                context.output,
                phase_status,
                exit_code=0,
                reason=result.limit,
                wall_s=time.monotonic() - stage_started,
            )
            settled.add(context.attempt_id)
            context.metrics.close()
            if phase_status == "STOPPED_BOUNDED":
                status = "STOPPED_BOUNDED"
                break
    except BaseException as exc:
        failure = repr(exc)
        # 保存失败是真实工程故障：即使同时收到停止请求也要报FAILED。
        from native_stand.simulation.driver import SaveFailed

        if isinstance(exc, SaveFailed):
            status = "FAILED"
        else:
            status = "STOPPED_BOUNDED" if STOP_REQUESTED else "FAILED"
        if context is not None:
            if context.attempt_id not in settled:
                try:
                    context.budget.finalize(
                        context.attempt_id,
                        context.output,
                        status,
                        reason=failure,
                        wall_s=time.monotonic() - stage_started,
                    )
                    settled.add(context.attempt_id)
                except Exception as settle_exc:
                    # 结算失败不能被无声吞掉：留下记录，避免假RUNNING无迹可查。
                    write_json(
                        run_dir / "settlement_error.json",
                        {"attempt_id": context.attempt_id, "error": repr(settle_exc)},
                    )
            with contextlib.suppress(Exception):
                context.metrics.close()
        else:
            # 启动阶段就失败：没有worker接管，仍要留下真实失败记录。
            try:
                if store.read().get("status") == "SUBMITTED":
                    store.terminal("FAILED", failure=failure, allow_foreign=True)
            except Exception:
                pass
    finally:
        if context is not None:
            try:
                context.prune_checkpoints()
            except Exception as exc:
                write_json(run_dir / "cleanup_error.json", {"error": repr(exc)})
            payload = {
                "status": status,
                "failure": failure,
                "updates_used": context.updates_used,
                "samples_used": context.samples_used,
                "actor_parameters_changed": context.actor_changed,
                "wall_s": time.monotonic() - started,
                "latest_checkpoint": context.latest_reference(),
                "save_failures": context.save_failures,
            }
            try:
                write_report(
                    run_dir,
                    state_snapshot={
                        **context.store.read(),
                        **payload,
                        "capability_accepted": False,
                    },
                )
            except Exception as exc:
                write_json(run_dir / "report_error.json", {"error": repr(exc)})
            try:
                context.store.terminal(
                    status,
                    failure=failure,
                    writer_pid=os.getpid(),
                    # 没接管过state的worker只补写启动失败，不与运行中owner竞争。
                    allow_foreign=not context.claimed,
                    updates_used=context.updates_used,
                    samples_used=context.samples_used,
                    capability_accepted=False,
                    actor_parameters_changed=payload["actor_parameters_changed"],
                    wall_s=payload["wall_s"],
                    latest_model=context.latest_reference(),
                )
            except Exception as exc:
                write_json(
                    run_dir / "terminal_error.json", {"error": repr(exc), "failure": failure}
                )
            with contextlib.suppress(Exception):
                hapi.notify(
                    run_dir,
                    enabled=context.job.notifications.enabled,
                    target=context.job.notifications.target,
                )
        _send(
            args.handshake_fd,
            {
                "status": status,
                "failure": failure,
                "updates_used": context.updates_used if context else 0,
            },
        )
    return 0 if status in ("COMPLETED", "STOPPED_BOUNDED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
