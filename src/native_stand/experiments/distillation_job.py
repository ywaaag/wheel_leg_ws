"""蒸馏作业：走现有owner/worker生命周期，BC轮次独立计量。

- CLI把有效BC配置写入run.json，owner启动真正的worker子进程；
- worker claim后调用一次 `run_distillation()`（含剩余轮次），轮边界通过
  `on_round_end` 计费、写进度；`stop_requested` 提供轮边界停止；
- rounds只进 budget.distill 与 state.rounds_used，不写PPO的updates_used。
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import time
from pathlib import Path
from typing import Any

from native_stand.artifacts.checkpoint import file_sha256
from native_stand.artifacts.io import read_json, write_json
from native_stand.learning.distillation.loop import run_distillation

from .budget import Budget
from .config import JobConfig
from .owner import RUNS_ROOT
from .state import RunStateStore

REQUIRED = (
    "dataset",
    "teacher",
    "rounds",
    "batch_size",
    "learning_rate",
    "validation_fraction",
    "holdout_fraction",
    "seed",
)


def build_config(
    job: JobConfig,
    *,
    dataset: str | Path,
    teacher: str | Path,
    rounds: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    holdout_fraction: float,
    seed: int | None,
    teacher_hidden: tuple[int, ...] | None = None,
) -> tuple[JobConfig, dict[str, Any]]:
    """校验输入并生成有效BC配置；任何缺失在创建run之前拒绝。"""
    dataset = Path(dataset).resolve()
    teacher = Path(teacher).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"数据集不存在: {dataset}")
    if not teacher.is_file():
        raise FileNotFoundError(f"Teacher不存在: {teacher}")
    config = {
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "teacher": str(teacher),
        "teacher_sha256": file_sha256(teacher),
        "teacher_hidden": list(teacher_hidden) if teacher_hidden else None,
        "rounds": int(rounds),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "validation_fraction": float(validation_fraction),
        "holdout_fraction": float(holdout_fraction),
        "seed": int(job.runtime.seed if seed is None else seed),
        "action_version": job.action.version,
        "input_version": job.model.interface,
        "student_architecture": list(job.model.actor_hidden_dims),
        "labels": "teacher_online",
    }
    distill_job = dataclasses.replace(
        job,
        algorithm="distillation",
        experiment_variables={**job.experiment_variables, "distill": config},
    )
    return distill_job, config


def submit_distillation(
    job: JobConfig,
    *,
    dataset: str | Path,
    teacher: str | Path,
    rounds: int,
    runs_root: str | Path = RUNS_ROOT,
    run_id: str | None = None,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    holdout_fraction: float = 0.2,
    seed: int | None = None,
    teacher_hidden: tuple[int, ...] | None = None,
    wait_s: float = 30.0,
    stop_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """提交蒸馏run：owner启动worker，训练在worker内进行；返回提交回执。"""
    from native_stand.experiments import owner

    distill_job, _config = build_config(
        job,
        dataset=dataset,
        teacher=teacher,
        rounds=rounds,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        holdout_fraction=holdout_fraction,
        seed=seed,
        teacher_hidden=teacher_hidden,
    )
    return owner.submit(
        distill_job,
        runs_root=runs_root,
        run_id=run_id,
        purpose="distillation",
        distill_rounds=rounds,
        wait_s=wait_s,
    )


def _identity(path: Path) -> dict[str, str]:
    return {"checkpoint": str(path.resolve()), "sha256": file_sha256(path)}


def _latest_state(run_dir: Path) -> dict[str, Any] | None:
    """所有attempt产物中mtime最新的latest完整BC状态。"""
    import torch

    candidates = sorted(
        run_dir.glob("distill*/distillation_latest.pt"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        return None
    return torch.load(candidates[-1], map_location="cpu", weights_only=False)


def run_distillation_worker(
    run_dir: str | Path,
    job: JobConfig,
    *,
    handshake_fd=None,
    attempt_suffix: str | None = None,
) -> int:
    """worker内的BC执行：claim→单次训练循环（回调计费/发布）→结算。"""
    from native_stand.artifacts.runtime_paths import execution_job
    from native_stand.experiments import worker as worker_mod

    job = execution_job(job)
    directory = Path(run_dir)
    budget = Budget(directory / "budget.json")
    store = RunStateStore(directory)
    started = time.monotonic()
    # attempt身份由本函数生成并传入_new_context；目录与账本、回执保持同一标识。
    import uuid as _uuid

    attempt_id = _uuid.uuid4().hex
    suffix = attempt_suffix or f"{time.strftime('%H%M%S')}_{attempt_id[:6]}"
    output = directory / f"distill_{suffix}"
    settled = False
    try:
        context = worker_mod._new_context(
            directory, "distill", suffix=suffix, attempt_id=attempt_id
        )
        cfg = (job.experiment_variables or {}).get("distill")
        if job.supervision is not None:
            cfg = dict(
                dataset=job.supervision.dataset["path"],
                dataset_sha256=job.supervision.dataset["sha256"],
                teacher=job.supervision.teachers[0].path,
                teacher_sha256=job.supervision.teachers[0].sha256,
            )
        if not cfg:
            raise ValueError("run记录缺少distill配置；不允许回退函数默认值")
        if store.read().get("status") in ("COMPLETED", "STOPPED_BOUNDED") or (
            job.supervision is not None and store.read().get("status") == "FAILED"
        ):
            store.reopen_for_resume(
                pid=os.getpid(),
                attempt_id=attempt_id,
                stage="distill",
                allow_failed=job.supervision is not None,
            )
        store.claim(pid=os.getpid(), attempt_id=attempt_id, stage="distill")
        context.claimed = True
        # 握手必须在启动时送达owner；否则owner会等到worker退出才收到回执。
        context.startup_fd = handshake_fd if handshake_fd is not None else context.startup_fd
        context.startup_stage("distill")

        limits, used = budget.limits(), budget.used()
        remaining = int(limits["distill"]) - int(used["distill"])
        if remaining <= 0 and job.supervision is None:
            raise ValueError("蒸馏轮次额度已用尽；不刷新额度")

        # 训练前一次性核对输入身份；不一致在optimizer step前拒绝。
        if file_sha256(Path(cfg["dataset"])) != cfg["dataset_sha256"]:
            raise ValueError("数据集身份与run记录不一致")
        if file_sha256(Path(cfg["teacher"])) != cfg["teacher_sha256"]:
            raise ValueError("Teacher身份与run记录不一致")

        from native_stand.artifacts.model_refs import register_model

        register_model(
            context.refs.path, cfg["teacher"], roles=("distillation", "teacher"), run=directory.name
        )
        register_model(
            context.refs.path, cfg["dataset"], roles=("distillation",), run=directory.name
        )
        registered = (store.read().get("latest_model") or {}).get("checkpoint")
        if job.supervision is not None:
            if used["distill"] > 0:
                if not registered:
                    raise ValueError("已计费监督消耗缺少持久化恢复点；不退款或重做")
                import torch

                saved = torch.load(registered, map_location="cpu", weights_only=False)
                if int(saved["round"]) != int(used["distill"]):
                    raise ValueError("已计费监督消耗超出持久化恢复点；需要显式重算计费")
            registered = None  # phase state owns its explicit fit continuation
        if registered:
            import torch

            register_model(context.refs.path, registered, roles=("resume",), run=directory.name)
            resume_state = torch.load(registered, map_location="cpu", weights_only=False)
        elif used["distill"] > 0 and job.supervision is None:
            raise ValueError("已消耗轮次大于0但没有登记的可恢复latest；拒绝重新初始化")
        else:
            resume_state = None  # 0轮：按run内配置与seed初始化

        last_space_check = 0.0
        space_exhausted = False

        def stop_requested() -> bool:
            nonlocal last_space_check, space_exhausted
            # 唯一停止解释入口：after_round=N=累计N轮后停止；普通停止下个轮边界生效。
            wall = context.wall_budget_left()
            space = context.output_bytes_left()
            if wall is not None and wall <= 0:
                return True
            from native_stand.artifacts.retention import disk_usage

            if time.monotonic() - last_space_check >= 1:
                last_space_check = time.monotonic()
                space_exhausted = (
                    space is not None
                    and disk_usage(directory) >= space - job.storage.terminal_reserve_bytes
                )
            if space_exhausted:
                return True
            request = read_json(directory / "stop_request.json") or {}
            after = request.get("after_round")
            if after is not None:
                return budget.used()["distill"] >= int(after)
            return context.stop_requested()

        def on_round_complete(progress: dict[str, Any]) -> None:
            # 完成回调：本轮优化已完成，按实际增量计费。
            budget.charge_rounds(attempt_id, output, rounds=1)
            budget.charge_resources(
                attempt_id, output, supervised_updates=progress.get("optimizer_steps", 0)
            )
            store.update(rounds_used=int(progress["round"]), algorithm="distillation")

        def on_round_saved(progress: dict[str, Any]) -> None:
            from native_stand.artifacts.model_refs import ModelReference

            context.refs.register_output(
                ModelReference(
                    progress["latest_sha256"],
                    progress["latest_path"],
                    ("resume",),
                    run=directory.name,
                    kind="bc_full",
                    owner=directory.name,
                ),
                consumer=directory.name,
            )
            # 保存回调：latest已原子保存并核实SHA，发布身份。
            store.update(
                latest_model={
                    "checkpoint": progress["latest_path"],
                    "sha256": progress["latest_sha256"],
                },
                algorithm="distillation",
            )

        flow_complete = True
        if job.supervision is not None and not stop_requested():
            from .supervision_flow import execute

            _, flow_complete = execute(
                job,
                directory / "supervision",
                stop_requested=stop_requested,
                on_round_complete=on_round_complete,
                on_round_saved=on_round_saved,
                index=context.refs,
                logical_run_id=read_json(directory / "run.json")["logical_run_id"],
                on_physical_steps=lambda count: budget.charge_resources(
                    attempt_id, output, physical_steps=count
                ),
            )
        elif remaining > 0 and not stop_requested():
            run_distillation(
                job,
                dataset_path=cfg["dataset"],
                teacher_checkpoint=cfg["teacher"],
                output_dir=output,
                rounds=remaining,
                batch_size=int(cfg["batch_size"]),
                learning_rate=float(cfg["learning_rate"]),
                validation_fraction=float(cfg["validation_fraction"]),
                holdout_fraction=float(cfg["holdout_fraction"]),
                seed=int(cfg["seed"]),
                teacher_hidden=tuple(cfg["teacher_hidden"]) if cfg["teacher_hidden"] else None,
                resume_state=resume_state,
                stop_requested=stop_requested,
                on_round_complete=on_round_complete,
                on_round_saved=on_round_saved,
            )

        rounds_done = int(used["distill"]) + budget.used()["distill"] - int(used["distill"])
        rounds_done = budget.used()["distill"]
        status = "STOPPED_BOUNDED" if stop_requested() or not flow_complete else "COMPLETED"
        budget.finalize(
            attempt_id, output, status, exit_code=0, reason=None, wall_s=time.monotonic() - started
        )
        settled = True
        state = store.read()
        if rounds_done > 0 and state.get("latest_model"):
            best_file = _newest_best(directory)
            store.update(
                rounds_used=rounds_done,
                best_actor=_identity(best_file) if best_file else None,
                algorithm="distillation",
            )
        store.terminal(status, failure=None, writer_pid=os.getpid())
        return 0
    except BaseException as exc:
        failure = repr(exc)
        try:
            row = budget.rows().get(attempt_id) or {}
            if not settled and row.get("status") == "RUNNING":
                # 已被_new_context结算的attempt不再重复结算。
                budget.finalize(
                    attempt_id, output, "FAILED", reason=failure, wall_s=time.monotonic() - started
                )
        except Exception as settle_exc:
            # reserve未发生等情况：如实记录结算失败，不扫描账本猜归属。
            write_json(directory / "settlement_error.json", {"error": repr(settle_exc)})
        try:
            store.terminal("FAILED", failure=failure, writer_pid=os.getpid())
        except Exception as state_exc:
            write_json(directory / "state_error.json", {"error": repr(state_exc)})
        write_json(directory / "worker_failure.json", {"error": failure})
        return 1
    finally:
        with contextlib.suppress(Exception):
            context.metrics.close()


def _newest_latest(run_dir: Path) -> Path | None:
    candidates = sorted(
        run_dir.glob("distill*/distillation_latest.pt"), key=lambda p: p.stat().st_mtime
    )
    return candidates[-1] if candidates else None


def _newest_best(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("distill*/student_actor.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _precheck_resume(run_dir: Path) -> tuple[JobConfig, dict[str, Any], int, dict[str, Any]]:
    """恢复前置检查：状态、owner、RUNNING attempt、余额、latest身份。"""
    record = read_json(run_dir / "run.json")
    if record is None:
        raise FileNotFoundError(f"不是新结构run目录: {run_dir}")
    if (record["config"].get("algorithm") or "ppo") != "distillation":
        raise ValueError("该run不是蒸馏作业")
    from native_stand.artifacts.runtime_paths import execution_job
    from native_stand.experiments.config import job_from_record

    job = execution_job(job_from_record(record["config"]))
    state = read_json(run_dir / "state.json") or {}
    if state.get("status") == "RUNNING":
        raise ValueError("该run仍有RUNNING worker；不能重复启动")
    budget = Budget(run_dir / "budget.json")
    running = [row for row in budget.rows().values() if row["status"] == "RUNNING"]
    if running:
        raise ValueError("预算中仍有RUNNING attempt；不能重复启动")
    limits, used = budget.limits(), budget.used()
    remaining = int(limits["distill"]) - int(used["distill"])
    if remaining <= 0 and job.supervision is None:
        raise ValueError("蒸馏轮次额度已用尽；不刷新额度")
    if used["distill"] == 0 and not (state.get("latest_model") or {}).get("checkpoint"):
        # 0轮恢复：按run内配置与seed初始化，继续原额度；不要求checkpoint。
        return job, (job.experiment_variables or {}).get("distill") or {}, remaining, None
    if job.supervision is not None and used["distill"] > 0:
        # Recover a saved-but-not-published fit checkpoint from its explicit phase position.
        from native_stand.artifacts.checkpoint import load_actor_source

        from .supervision_flow import semantic_identity

        flow = read_json(run_dir / "supervision/supervision_state.json")
        if flow and flow.get("version") == "supervision_flow_v2":
            cursor = flow["phase_cursor"]
            phases = list(job.supervision.phases) * job.supervision.repetitions
            if cursor < len(phases) and phases[cursor].operation == "fit":
                known = (
                    run_dir / "supervision" / f"phase_{cursor:03d}_fit" / "distillation_latest.pt"
                )
                if known.is_file():
                    loaded = load_actor_source(known)
                    if (
                        loaded.payload["round"] == used["distill"]
                        and loaded.payload.get("logical_run_id") == record["logical_run_id"]
                        and loaded.payload.get("supervision_identity")
                        == semantic_identity(job.supervision)
                    ):
                        from native_stand.artifacts.model_refs import ModelIndex, ModelReference

                        index = ModelIndex(record["model_index"])
                        index.register_output(
                            ModelReference(
                                loaded.sha256,
                                str(known),
                                ("resume",),
                                run=run_dir.name,
                                kind="bc_full",
                                owner=run_dir.name,
                            ),
                            consumer=run_dir.name,
                        )
                        RunStateStore(run_dir).update(
                            allow_foreign=True,
                            latest_model=dict(checkpoint=str(known), sha256=loaded.sha256),
                        )
                        state = read_json(run_dir / "state.json")
    registered = state.get("latest_model") or {}
    if not registered.get("checkpoint"):
        raise ValueError("已消耗轮次大于0但没有登记的可恢复latest checkpoint")
    import torch

    latest_path = Path(registered["checkpoint"])
    if not latest_path.is_file():
        raise FileNotFoundError(f"登记的latest不存在: {latest_path}")
    if registered.get("sha256") != file_sha256(latest_path):
        raise ValueError("登记的latest SHA与实际文件不一致")
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    cfg = (job.experiment_variables or {}).get("distill") or {}
    if int(latest.get("round") or 0) != int(used["distill"]):
        raise ValueError("latest轮次与账本消耗不一致")
    if job.supervision is not None:
        if latest.get("logical_run_id") != record["logical_run_id"]:
            raise ValueError("监督状态不能冒用另一个逻辑run")
        from .supervision_flow import semantic_identity

        if latest.get("supervision_identity") != semantic_identity(job.supervision):
            raise ValueError("监督来源或标签合同变化")
        return job, job.supervision.record(), remaining, latest
    if latest.get("dataset_sha256") != cfg.get("dataset_sha256"):
        raise ValueError("数据集身份与run记录不一致")
    if latest.get("teacher_sha256") != cfg.get("teacher_sha256"):
        raise ValueError("Teacher身份与run记录不一致")
    return job, cfg, remaining, latest


def resume_distillation_job(run_dir: str | Path, *, wait_s: float = 30.0) -> dict[str, Any]:
    """在同一run上以剩余额度启动worker；复用owner.resume_run与握手。"""
    from native_stand.experiments import owner

    directory = Path(run_dir)
    job, _cfg, _remaining, _latest = _precheck_resume(directory)
    # 消费上一attempt的停止请求：resume是显式用户动作，新attempt不再立即停止。
    stop_file = directory / "stop_request.json"
    if stop_file.is_file():
        consumed = directory / "source" / "consumed_stop_requests.jsonl"
        consumed.parent.mkdir(parents=True, exist_ok=True)
        with consumed.open("a") as handle:
            handle.write(stop_file.read_text().strip() + "\n")
        stop_file.unlink()
    return owner.resume_run(directory, wait_s=wait_s)
