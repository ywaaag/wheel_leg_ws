"""统一命令行入口：train / evaluate / distill / diagnose / resume / export / inspect / stop。

只有参数解析、展示与服务调用；不在这里实现训练、评估或文件格式。
帮助命令不导入Isaac、不初始化GPU、不遍历历史run。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from native_stand.artifacts.checkpoint import load_full
from native_stand.artifacts.io import read_json
from native_stand.artifacts.runtime_paths import resolve_runs_root
from native_stand.experiments.config import (
    ConfigError,
    JobConfig,
    job_from_record,
    load_job_config,
    with_initialization,
)
from native_stand.experiments.owner import inspect as inspect_run
from native_stand.experiments.owner import submit

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INVALID = 2


class CommandError(RuntimeError):
    """命令无法执行；消息面向用户。"""

    def __init__(self, message: str, code: int = EXIT_FAILED):
        super().__init__(message)
        self.code = code


def require_supported(job: JobConfig, operation: str) -> None:
    """在创建run与预约之前拒绝不支持的组合。"""
    from native_stand.simulation.dispatch import DispatchError
    from native_stand.simulation.dispatch import require_supported as check

    try:
        check(job, operation)
    except (DispatchError, ValueError) as exc:
        raise CommandError(str(exc), EXIT_INVALID) from exc


def require_engine(job: JobConfig) -> None:
    require_supported(job, "train")


def _run_dir(run_id: str) -> Path:
    path = Path(run_id)
    if path.is_dir():
        return path
    try:
        candidate = resolve_runs_root() / run_id
    except ValueError as exc:
        raise CommandError(str(exc), EXIT_INVALID) from exc
    if candidate.is_dir():
        return candidate
    raise CommandError(f"找不到run: {run_id}", EXIT_INVALID)


def _emit(args, payload: dict[str, Any], *, text: str | None = None) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    elif text is not None:
        print(text)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _config(args) -> JobConfig:
    try:
        return load_job_config(args.config)
    except ConfigError as exc:
        raise CommandError(f"配置不合法: {exc}", EXIT_INVALID) from exc


def cmd_train(args) -> int:
    if getattr(args, "max_updates", None) is not None and args.max_updates <= 0:
        raise CommandError("--max-updates必须为正", EXIT_INVALID)
    job = _config(args)
    from native_stand.experiments.plan import build_plan

    plan = build_plan(job)
    from native_stand.simulation.dispatch import execution_service

    service = execution_service(job)
    plan["execution"] = service
    if service["service"] == "unsupported" and job.task == "jump":
        raise CommandError(service["reason"], EXIT_INVALID)
    if service["service"] == "production_motion" or service["module"] in (
        "native_stand.simulation.recovery_isaac.backend",
        "native_stand.simulation.jump_backend",
    ):
        try:
            module = __import__(service["module"], fromlist=["validate_job"])
            module.validate_job(job)
        except ValueError as exc:
            raise CommandError(str(exc), EXIT_INVALID) from exc
    if args.dry_run:
        _emit(
            args,
            {"action": "plan", **plan},
            text=(
                f"计划: {plan['name']} task={plan['task']} backend={plan['backend']} "
                f"updates={plan['total_updates']} 预计产物{plan['storage']['estimated_total_bytes']}字节"
                "（dry-run未创建run、未reserve、未启动worker）"
            ),
        )
        return EXIT_OK
    require_supported(job, "train")
    handshake = submit(
        job,
        purpose=args.purpose or job.name,
        run_id=args.run_id,
        wait_s=args.wait_seconds,
        runs_root=resolve_runs_root(args.runs_root),
        max_updates=getattr(args, "max_updates", None),
    )
    reported = handshake.get("worker_reported") or {}
    payload = {
        "action": "submitted",
        "dashboard": {
            "artifact_root": str(Path(handshake["run"]).parent),
            "run": handshake["run"],
            "identity_record": "startup_handshake.json",
            "discovery": "refresh dashboard authorized root; no training resubmission",
        },
        "run": handshake["run"],
        "worker_pid": handshake["worker_pid"],
        "worker_identity": handshake["worker_identity"].get("status"),
        "worker_started": handshake.get("worker_started"),
        "worker_exit_code": handshake.get("worker_exit_code"),
        "reported_status": reported.get("status"),
        "formal_started": handshake["formal_started"],
        "note": "提交成功不等于正式训练已开始；用 wheel-leg inspect 读取实际进度",
    }
    _emit(
        args,
        payload,
        text=(
            f"已提交: {payload['run']}（worker {payload['worker_pid']}，"
            f"身份{payload['worker_identity']}，启动{payload['worker_started']}，"
            f"上报{payload['reported_status']}）\n"
            f"formal_started={payload['formal_started']}；进度用 wheel-leg inspect 查询"
        ),
    )
    if reported.get("status") == "FAILED":
        return EXIT_FAILED
    if handshake.get("worker_started") is False:
        return EXIT_FAILED
    return EXIT_OK


def cmd_dashboard(args):
    """Only serving explicitly allowed artifacts; dry-run does not open files or sockets."""
    if args.dry_run:
        print(
            json.dumps(
                {
                    "operation": "dashboard",
                    "roots": args.root,
                    "host": args.host,
                    "port": args.port,
                    "read_only": True,
                },
                ensure_ascii=False,
            )
        )
        return EXIT_OK
    from native_stand.integrations.dashboard import serve

    remote = None
    if args.remote_motion:
        import getpass

        from native_stand.integrations.remote_dashboard import RemoteSource

        password = (
            getpass.getpass("远端SSH密码（留空使用key/agent；仅当前进程内存）：")
            if args.remote_password_prompt
            else None
        )
        if not all(
            (
                args.remote_host,
                args.remote_port,
                args.remote_user,
                args.remote_root,
                args.remote_fingerprint,
            )
        ):
            raise CommandError("远端读取需要显式host/port/user/root/fingerprint", EXIT_INVALID)
        remote = RemoteSource(
            host=args.remote_host,
            port=args.remote_port,
            username=args.remote_user,
            root=args.remote_root,
            fingerprint=args.remote_fingerprint,
            password=password,
        )
    serve(
        args.root,
        host=args.host,
        port=args.port,
        tensorboard_url=args.tensorboard_url,
        public_host=args.public_host,
        remote=remote,
    )
    return EXIT_OK


def cmd_inspect(args) -> int:
    payload = inspect_run(_run_dir(args.run))
    state = payload["state"]
    _emit(
        args,
        payload,
        text=(
            f"run {payload['run_id']}\n"
            f"状态 {state.get('status')} 阶段 {state.get('stage')} "
            f"更新 {state.get('updates_used')} 样本 {state.get('samples_used')}\n"
            f"最新模型 {(state.get('latest_model') or {}).get('checkpoint')}\n"
            f"选中模型 {(state.get('selected_model') or {}).get('checkpoint')}\n"
            f"能力accepted {bool(state.get('capability_accepted'))}"
        ),
    )
    return EXIT_OK


def cmd_stop(args) -> int:
    from native_stand.experiments.owner import stop as stop_run

    payload = stop_run(
        _run_dir(args.run),
        reason=args.reason,
        dry_run=args.dry_run,
        **({"request_only": True} if args.request_only else {}),
    )
    _emit(args, payload, text=f"停止请求: {payload.get('status')} stopped={payload.get('stopped')}")
    return EXIT_OK


def cmd_resume(args) -> int:
    if args.max_updates is not None and args.max_updates <= 0:
        raise CommandError("--max-updates必须为正", EXIT_INVALID)
    source = _run_dir(args.run)
    record = read_json(source / "run.json")
    if record is None:
        raise CommandError(f"不是新结构run目录: {source}", EXIT_INVALID)
    # 只认run内保存的有效配置；原始config_path仅作来源说明，文件即使删除也不影响恢复。
    job = job_from_record(record["config"])
    if job.algorithm == "distillation":
        if args.dry_run:
            _emit(
                args,
                {"action": "resume-plan", "run": str(source), "algorithm": "distillation"},
                text="蒸馏恢复计划已解析；dry-run未执行训练",
            )
            return EXIT_OK
        from native_stand.experiments.distillation_job import resume_distillation_job

        receipt = resume_distillation_job(source)
        _emit(
            args,
            {
                "action": "resumed",
                "summary": {
                    "run": receipt.get("run"),
                    "worker_pid": receipt.get("worker_pid"),
                    "worker_identity": receipt.get("worker_identity"),
                    "worker_started": receipt.get("worker_started"),
                    "worker_reported": receipt.get("worker_reported"),
                    "resumed": receipt.get("resumed"),
                    "stage": receipt.get("stage"),
                },
            },
            text=(
                f"蒸馏恢复已提交: run={receipt.get('run')} "
                f"worker_pid={receipt.get('worker_pid')}；结果用 inspect --run 查看"
            ),
        )
        return EXIT_OK
    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(source)
    if checkpoint is None:
        raise CommandError("该run没有选中或最新的完整checkpoint，无法恢复", EXIT_INVALID)
    resumed = with_initialization(job, checkpoint)
    if not resumed.initialization.sha256:
        raise CommandError("恢复checkpoint缺少SHA", EXIT_INVALID)
    load_full(resumed.initialization.checkpoint)
    if args.dry_run:
        _emit(
            args,
            {
                "action": "resume-plan",
                "from": str(source),
                "checkpoint": str(resumed.initialization.checkpoint),
                "sha256": resumed.initialization.sha256,
                "updates_used_in_source": (read_json(source / "state.json", {}) or {}).get(
                    "updates_used"
                ),
            },
            text="恢复计划已核对完整checkpoint；dry-run未创建run",
        )
        return EXIT_OK
    from native_stand.experiments.owner import resume_run

    # Resume retains the immutable run recipe/allocation. The checkpoint is
    # independently verified above and by resume_run; it is not a new init grant.
    require_supported(job, "train")
    handshake = resume_run(
        source,
        checkpoint=checkpoint,
        wait_s=args.wait_seconds,
        **({"max_updates": args.max_updates} if args.max_updates is not None else {}),
    )
    if not handshake.get("resumed"):
        _emit(
            args,
            {"action": "resume-refused", **handshake},
            text=f"无法恢复: {handshake.get('reason')}",
        )
        return EXIT_FAILED
    _emit(
        args,
        {"action": "resumed", **handshake},
        text=(
            f"已在原run上续训: {handshake['run']} 阶段{handshake['stage']} "
            f"attempt {handshake['attempt_suffix']}；预算未刷新 "
            f"used={handshake['budget_used']}"
        ),
    )
    return EXIT_OK if handshake.get("worker_started") else EXIT_FAILED


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """默认恢复点：最新写入的完整状态（按update序号），不看best/selected。"""
    from native_stand.artifacts.retention import read_inventory

    state = read_json(run_dir / "state.json", {}) or {}
    rows = [row for row in read_inventory(run_dir) if row.get("checkpoint")]
    rows = [row for row in rows if Path(row["checkpoint"]).is_file()]
    if rows:
        return Path(max(rows, key=lambda row: int(row.get("update") or 0))["checkpoint"])
    latest = (state.get("latest_model") or {}).get("checkpoint")
    return Path(latest) if latest else None


def cmd_export(args) -> int:
    run_dir = _run_dir(args.run)
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        state = read_json(run_dir / "state.json", {}) or {}
        selected = (state.get("selected_model") or {}).get("checkpoint")
        if not selected:
            raise CommandError(
                "该run没有显式选中的模型；export不会把latest当作best，请用 --checkpoint 指定",
                EXIT_INVALID,
            )
        checkpoint = Path(selected)
    if not checkpoint.is_file():
        raise CommandError(f"模型文件不存在: {checkpoint}", EXIT_INVALID)
    record = read_json(run_dir / "run.json", {}) or {}
    output = (
        Path(args.output)
        if args.output
        else (
            None
            if args.dry_run
            else resolve_runs_root() / "exports" / run_dir.name / checkpoint.stem
        )
    )
    if args.dry_run:
        _emit(
            args,
            {
                "action": "export-plan",
                "checkpoint": str(checkpoint),
                "output": str(output),
                "writes": False,
                "loads_model": False,
            },
            text="部署包计划；未加载模型或写入",
        )
        return EXIT_OK
    from native_stand.artifacts.checkpoint import load_actor_source
    from native_stand.artifacts.deployment import export_package, job_dependencies
    from native_stand.artifacts.model_refs import ModelIndex
    from native_stand.experiments.config import job_from_record
    from native_stand.learning.inference import InferenceSpec, spec_for_job

    job = job_from_record(record["config"])
    source = load_actor_source(checkpoint)
    if job.supervision is not None:
        # BC saves its explicit inference contract, including the input column
        # semantics. Reconstructing it from the PPO job would change identity.
        spec = InferenceSpec.from_record(job.supervision.student)
    else:
        contract = source.payload.get("output_contract", source.payload.get("contract"))
        if contract is None:
            raise CommandError("部署来源缺少明确输出合同", EXIT_INVALID)
        spec = spec_for_job(job, contract)
    index = ModelIndex(record.get("model_index", run_dir.parent / "models.json"))
    jump_controller = None
    if job.task == "jump":
        from native_stand.tasks.jump.outcome import controller_record, enabled

        if enabled(job):
            jump_controller = controller_record()
    result = export_package(
        source,
        spec,
        output,
        execution={
            "control": job.as_run_record()["control"],
            "observation": job.as_run_record()["observation"],
            "timing": job.assets.get("delayed_execution"),
            "hardware_output": "OFF",
            "C_core": "jump_ABI4" if job.task == "jump" else "rl_policy_core_v1",
            "jump_controller": jump_controller,
        },
        dependencies=job_dependencies(job),
        onnx=args.onnx,
        index=index,
    )
    _emit(
        args,
        {
            "action": "exported",
            "output": str(output),
            "manifest": result,
            "exports_full_recovery_state": False,
        },
        text=f"已导出部署包: {output}",
    )

    return EXIT_OK


def cmd_evaluate(args) -> int:
    job = _config(args)
    from native_stand.learning.ppo.evaluate import default_conditions

    conditions = default_conditions(job)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    run_dir = _run_dir(args.run) if args.run else None
    if checkpoint is None and run_dir is not None:
        state = read_json(run_dir / "state.json", {}) or {}
        selected = (state.get("selected_model") or {}).get("checkpoint")
        if not selected:
            raise CommandError(
                "该run没有显式选中的模型；评估不会把latest当作best，请用 --checkpoint 指定",
                EXIT_INVALID,
            )
        checkpoint = Path(selected)
    if checkpoint is None:
        raise CommandError("evaluate需要--run（使用其选中模型）或--checkpoint", EXIT_INVALID)
    if not checkpoint.is_file():
        raise CommandError(f"模型文件不存在: {checkpoint}", EXIT_INVALID)
    if args.dry_run:
        _emit(
            args,
            {
                "action": "evaluate-plan",
                "checkpoint": str(checkpoint),
                "conditions": [row.get("name") for row in conditions],
                "protocol": job.evaluation.protocol_version,
                "envs": job.evaluation.envs,
                "seconds": job.evaluation.seconds,
                "backend": job.backend,
            },
            text="评估计划已解析；dry-run未执行环境",
        )
        return EXIT_OK
    require_supported(job, "evaluate")
    from native_stand.learning.ppo.evaluate import evaluate_checkpoint

    output = (
        Path(args.output)
        if args.output
        else ((run_dir / "evaluations") if run_dir is not None else None)
    )
    result = evaluate_checkpoint(job, checkpoint, kind=args.kind, output_dir=output)
    if run_dir is not None and result.get("path"):
        from native_stand.artifacts import evaluation as evaluation_index

        evaluation_index.register(
            run_dir / "evaluations", kind=args.kind, path=result["path"], result=result
        )
    judged = bool((result.get("protocol") or {}).get("capability_judged"))
    _emit(
        args,
        {
            "action": "evaluated",
            "summary": {
                "evaluation_id": result.get("evaluation_id"),
                "kind": result["kind"],
                "conditions": len(result["conditions"]),
                "path": result.get("path"),
                "sha256": result.get("sha256"),
                "capability_judged": judged,
                "capability_accepted": result.get("capability_accepted"),
            },
        },
        text=(
            f"评估完成: id={result.get('evaluation_id')} kind={result['kind']} "
            f"条件{len(result['conditions'])} 结果{result.get('path')}；"
            + ("能力判定已执行" if judged else "该协议未实现能力判定，accepted未授予")
        ),
    )
    return EXIT_OK


def cmd_sim2sim(args):
    from native_stand.simulation.sim2sim import from_file

    if not args.dry_run and args.output is None:
        raise CommandError("sim2sim需要显式--output", EXIT_INVALID)
    result = from_file(args.config, args.output, dry_run=args.dry_run)
    _emit(args, result, text="跨引擎既有轨迹诊断；无新增积分")
    return EXIT_OK


def cmd_distill(args) -> int:
    job = _config(args)
    require_supported(job, "distill")
    if job.supervision is not None:
        if args.max_rounds is not None and not 0 < args.max_rounds <= job.supervision.rounds:
            raise CommandError("--max-rounds必须在本配方监督额度内", EXIT_INVALID)
        if args.dry_run:
            _emit(
                args,
                {"action": "distill-plan", **job.supervision.plan()},
                text="监督流程计划；未加载模型或启动引擎",
            )
            return EXIT_OK
        import dataclasses

        from native_stand.experiments.owner import submit

        result = submit(
            dataclasses.replace(job, algorithm="distillation"),
            runs_root=resolve_runs_root(args.runs_root),
            run_id=args.run_id,
            purpose="distillation",
            distill_rounds=job.supervision.rounds,
            stop_request=dict(after_round=args.max_rounds) if args.max_rounds is not None else None,
        )
        _emit(args, result, text="监督流程已提交")
        return EXIT_OK
    payload = {
        "action": "distill-plan",
        "dataset": args.dataset,
        "teacher": args.teacher,
        "rounds": args.rounds,
        "validation_fraction": args.validation_fraction,
        "holdout_fraction": args.holdout_fraction,
        "rounds_budget": args.rounds,
    }
    if args.dry_run:
        _emit(args, payload, text="蒸馏计划已解析；dry-run未读数据集、未训练")
        return EXIT_OK
    if not args.dataset or not args.teacher:
        raise CommandError("distill需要--dataset与--teacher", EXIT_INVALID)
    from native_stand.experiments.distillation_job import submit_distillation

    result = submit_distillation(
        job,
        dataset=args.dataset,
        teacher=args.teacher,
        rounds=args.rounds,
        runs_root=resolve_runs_root(getattr(args, "runs_root", None)),
        run_id=args.run_id,
        batch_size=args.batch_size,
        learning_rate=getattr(args, "learning_rate", 1e-3),
        validation_fraction=args.validation_fraction,
        holdout_fraction=args.holdout_fraction,
    )
    _emit(
        args,
        {
            "action": "distill-submitted",
            "summary": {
                "run": result.get("run"),
                "worker_pid": result.get("worker_pid"),
                "worker_identity": result.get("worker_identity"),
                "handshake": result.get("handshake"),
                "wait_s": result.get("wait_s"),
            },
        },
        text=(
            f"蒸馏已提交: run={result.get('run')} worker_pid={result.get('worker_pid')}；"
            "结果用 inspect --run 查看"
        ),
    )
    return EXIT_OK


def cmd_diagnose(args) -> int:
    if args.run:
        run_dir = _run_dir(args.run)
        payload = inspect_run(run_dir)
        state = payload["state"]
        _emit(
            args,
            {"action": "diagnose", "state": state, "metrics_tail": payload["metrics_tail"]},
            text=f"诊断: 状态 {state.get('status')} 失败 {state.get('failure')}",
        )
        return EXIT_OK
    if args.config and read_json(args.config).get("version") == "motion_support_probe_v1":
        from native_stand.simulation.motion_support_probe import from_file

        result = from_file(args.config, dry_run=args.dry_run)
        _emit(args, result, text="Motion承重/包络诊断；不是能力认证或训练")
        return EXIT_OK
    job = _config(args)
    payload = {
        "action": "diagnose-plan",
        "traces_enabled": job.storage.save_traces,
        "trace_max_bytes": job.storage.trace_max_bytes,
        "metrics_every_updates": job.storage.metrics_every_updates,
        "policy_dt_s": job.control.policy_dt_s,
        "bounded_tail_seconds": 2.0,
    }
    _emit(args, payload, text="诊断计划已解析；默认不保存完整rollout")
    if not args.dry_run:
        raise CommandError("诊断执行需要真实引擎；当前仅提供计划与已存运行状态读取", EXIT_FAILED)
    return EXIT_OK


def cmd_play(args):
    from native_stand.simulation.playback import plan, run

    prepared = plan(
        _run_dir(args.run),
        args.checkpoint,
        path_mapping=args.path_mapping,
        checkpoint_sha256=args.checkpoint_sha256,
        terrain_map=args.terrain_map,
        solver_profile=args.solver_profile,
    )
    if args.headless and (args.steps is None or args.steps < 1):
        raise CommandError("headless必须声明正整数--steps", EXIT_INVALID)
    payload = (
        prepared.record()
        if args.dry_run
        else run(
            prepared,
            headless=args.headless,
            steps=args.steps,
            panel_port=args.panel_port,
            panel_host=args.panel_host,
            no_save=args.no_save,
        )
    )
    _emit(args, payload)
    return EXIT_OK


def cmd_import_checkpoint(args):
    from native_stand.artifacts.handoff import import_checkpoint
    from native_stand.artifacts.io import read_json
    from native_stand.artifacts.model_refs import ModelIndex, ModelReference

    manifest = read_json(args.manifest)
    target_run = (
        Path(manifest["target_run"])
        if manifest.get("model_type") == "bc_full"
        else args.destination.resolve().parent.parent
    )
    if args.dry_run:
        _emit(
            args,
            dict(
                action="import-plan",
                run=str(target_run),
                logical_run_id=args.run_id,
                writes=False,
                loads_model=False,
            ),
        )
        return EXIT_OK
    record = read_json(target_run / "run.json")
    if record.get("logical_run_id") != args.run_id:
        raise ValueError("目标run逻辑身份不同；先准备明确的运行记录与预算")
    manifest = read_json(args.manifest)
    budget = read_json(target_run / "budget.json")
    if (
        sum(budget["used"].values())
        != manifest["budget"][
            "rounds_used" if manifest.get("model_type") == "bc_full" else "updates_used"
        ]
        or budget["limits"] != manifest["budget"]["limits"]
    ):
        raise ValueError("目标预算与handoff结算不同")
    mapping = read_json(args.mapping)
    index_path = Path(record["model_index"]).resolve()
    if index_path.parent != target_run.parent:
        raise ValueError("目标模型索引不在本次runs-root；必须显式映射")
    loaded = import_checkpoint(
        args.manifest, destination=args.destination, mapping=mapping, expected_run_id=args.run_id
    )
    refs = ModelIndex(index_path)
    refs.register(ModelReference(loaded.sha256, str(loaded.path), ("resume",), args.run_id))
    for ref in manifest["dependencies"]:
        refs.register(
            ModelReference(
                ref["sha256"], str(Path(mapping[ref["path"]]).resolve()), ("source",), args.run_id
            )
        )
    _emit(args, dict(checkpoint=str(loaded.path), sha256=loaded.sha256, training_started=False))
    return EXIT_OK


def cmd_cloud_plan(args):
    from native_stand.artifacts.io import read_json, write_json
    from native_stand.integrations.cloud_bundle import inventory, materialize_dependencies, package

    if args.restore_dependencies:
        if not args.dependency_archive or args.archive or args.run or args.reference:
            raise ValueError("依赖恢复需要--dependency-archive，不能混入打包或run快照")
        plan = read_json(args.manifest)
        result = dict(
            dependencies=len(plan["artifacts"]),
            destination=str(args.restore_dependencies),
            writes=False,
            starts_engine=False,
        )
        if not args.dry_run:
            result = materialize_dependencies(
                plan,
                args.dependency_archive,
                args.restore_dependencies,
                source_root=Path(__file__).resolve().parents[2],
            )
        _emit(args, result)
        return EXIT_OK
    if not args.config:
        raise ValueError("依赖计划需要至少一个--config")

    references = [
        dict(path=path, sha256=sha, role=role) for path, sha, role in (args.reference or ())
    ]
    plan = inventory(
        args.config,
        Path(__file__).resolve().parents[2],
        runs=args.run or (),
        references=references,
    )
    if not args.dry_run:
        write_json(args.manifest, plan)
        if args.archive:
            package(plan, args.archive)
    _emit(
        args,
        dict(
            manifest=str(args.manifest),
            artifacts=len(plan["artifacts"]),
            unresolved=len(plan["unresolved"]),
            remote_verified=False,
            dry_run=args.dry_run,
        ),
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wheel-leg",
        description="闭链轮腿机器人统一入口：训练、评估、蒸馏、诊断、恢复、导出、查看与停止",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    json_flag = argparse.ArgumentParser(add_help=False)
    # 子命令也接受--json；SUPPRESS保证写在子命令前时不覆盖主parser的值。
    json_flag.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="输出机器可读JSON"
    )

    train = sub.add_parser("train", help="按配置训练", parents=[json_flag])
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--dry-run", action="store_true", help="只解析与估算，不创建run")
    train.add_argument("--run-id", default=None)
    train.add_argument(
        "--runs-root", default=None, help="新run输出根；默认读取WHEEL_LEG_DATA_ROOT；不迁移现有任务"
    )
    train.add_argument("--purpose", default=None)
    train.add_argument("--wait-seconds", type=float, default=30.0)
    train.add_argument(
        "--max-updates", type=int, default=None, help="本attempt最多更新；不扩大预算，边界保存退出"
    )
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="完整评估", parents=[json_flag])
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--run", default=None, help="使用该run的选中模型并写回其evaluations")
    evaluate.add_argument("--checkpoint", default=None)
    evaluate.add_argument("--kind", choices=("best", "final", "diagnostic"), default="final")
    evaluate.add_argument("--output", default=None)
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    sim2sim = sub.add_parser(
        "sim2sim", help="显式跨引擎顺序采集或既有轨迹诊断", parents=[json_flag]
    )
    sim2sim.add_argument("--config", type=Path, required=True)
    sim2sim.add_argument("--output", type=Path)
    sim2sim.add_argument("--dry-run", action="store_true")
    sim2sim.set_defaults(func=cmd_sim2sim)

    distill = sub.add_parser("distill", help="同接口策略蒸馏", parents=[json_flag])
    distill.add_argument("--config", type=Path, required=True)
    distill.add_argument("--dataset", default=None, help="状态数据集npz：states、episodes")
    distill.add_argument("--teacher", default=None, help="固定Teacher的完整checkpoint")
    distill.add_argument("--rounds", type=int, default=200)
    distill.add_argument(
        "--max-rounds", type=int, default=None, help="本次执行在累计监督轮边界停止，不改变总额度"
    )
    distill.add_argument("--run-id", default=None)
    distill.add_argument(
        "--runs-root", default=None, help="运行输出根；默认读取WHEEL_LEG_DATA_ROOT"
    )
    distill.add_argument("--batch-size", type=int, default=256)
    distill.add_argument("--learning-rate", type=float, default=1e-3)
    distill.add_argument("--validation-fraction", type=float, default=0.2)
    distill.add_argument("--holdout-fraction", type=float, default=0.2)
    distill.add_argument(
        "--checkpoint",
        default=None,
        help="不接受：distill总是新run；从既有run继续请用 resume --run",
    )
    distill.add_argument("--dry-run", action="store_true")
    distill.set_defaults(func=cmd_distill)

    diagnose = sub.add_parser("diagnose", help="有界诊断", parents=[json_flag])
    diagnose.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Job配置；或motion_support_probe_v1固定Actor有界MuJoCo承重/包络诊断协议",
    )
    diagnose.add_argument("--run", default=None)
    diagnose.add_argument("--dry-run", action="store_true")
    diagnose.set_defaults(func=cmd_diagnose)

    resume = sub.add_parser("resume", help="从完整checkpoint恢复", parents=[json_flag])
    resume.add_argument("--run", required=True)
    resume.add_argument("--checkpoint", default=None)
    resume.add_argument("--purpose", default=None)
    resume.add_argument("--wait-seconds", type=float, default=30.0)
    resume.add_argument("--dry-run", action="store_true")
    resume.add_argument(
        "--max-updates", type=int, default=None, help="本次最多新增更新；仍消耗原预算，边界保存退出"
    )
    resume.set_defaults(func=cmd_resume)

    export = sub.add_parser("export", help="导出明确的模型", parents=[json_flag])
    export.add_argument("--run", required=True)
    export.add_argument("--checkpoint", default=None)
    export.add_argument("--output", default=None)
    export.add_argument("--dry-run", action="store_true")
    export.add_argument("--onnx", action="store_true", help="同时导出ONNX；不授予ORT或ARM验证")
    export.set_defaults(func=cmd_export)

    dashboard = sub.add_parser(
        "dashboard", help="只读训练与能力看板（不启动引擎）", parents=[json_flag]
    )
    dashboard.add_argument(
        "--root",
        action="append",
        required=True,
        help="允许读取的run目录或直接包含run的目录；可重复",
    )
    dashboard.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--dry-run", action="store_true")
    dashboard.add_argument("--tensorboard-url", help="可选本机已存在TensorBoard链接；不启动进程")
    dashboard.add_argument(
        "--public-host", help="显式允许反向代理的IPv4 Host；服务仍只绑定loopback"
    )
    dashboard.add_argument(
        "--remote-motion", action="store_true", help="启用明确Motion远端只读源；仅页面手动采集"
    )
    dashboard.add_argument("--remote-port", type=int, default=None)
    dashboard.add_argument("--remote-host", help="本次明确授权的SSH地址")
    dashboard.add_argument("--remote-user", help="SSH用户名")
    dashboard.add_argument("--remote-root", help="唯一允许读取的远端run目录")
    dashboard.add_argument("--remote-fingerprint", help="已独立核对的ED25519 SHA256主机指纹")
    dashboard.add_argument(
        "--remote-password-prompt",
        action="store_true",
        help="后端getpass私密输入；不落盘，不传浏览器",
    )
    dashboard.set_defaults(func=cmd_dashboard)

    inspect = sub.add_parser("inspect", help="只读查看run", parents=[json_flag])
    inspect.add_argument("--run", required=True)
    inspect.set_defaults(func=cmd_inspect)

    play = sub.add_parser(
        "play", help="统一MuJoCo播放（GUI/headless同一控制循环）", parents=[json_flag]
    )
    play.add_argument("--run", required=True)
    play.add_argument("--checkpoint", required=True)
    play.add_argument("--headless", action="store_true")
    play.add_argument("--steps", type=int, default=None)
    play.add_argument("--dry-run", action="store_true")
    play.add_argument("--path-mapping", help="本机RuntimePaths映射JSON；只读派生视图")
    play.add_argument("--checkpoint-sha256", help="加载时核对实际checkpoint SHA256")
    play.add_argument("--terrain-map", action="store_true", help="Motion平地＋四种H1真实碰撞地图")
    play.add_argument(
        "--panel-port", type=int, help="本机网页独占控制端口，始终不保存；断线保持最后命令"
    )
    play.add_argument(
        "--panel-host",
        default="127.0.0.1",
        help="网页监听地址；局域网需显式填写本机IP，0.0.0.0监听所有接口",
    )
    play.add_argument(
        "--no-save", action="store_true", help="不写轨迹/结果/日志；临时编译仅在内存盘并清理"
    )
    play.add_argument(
        "--solver-profile",
        default="legacy_xml_v1",
        help="显式MuJoCo求解器profile；closure_tight_v1改变闭链柔度，不是PhysX标定",
    )
    play.set_defaults(func=cmd_play)

    handoff = sub.add_parser(
        "import-checkpoint", help="按显式manifest导入完整恢复点；不启动训练", parents=[json_flag]
    )
    handoff.add_argument("--manifest", type=Path, required=True)
    handoff.add_argument("--mapping", type=Path, required=True)
    handoff.add_argument("--destination", type=Path, required=True)
    handoff.add_argument("--run-id", required=True)
    handoff.add_argument("--dry-run", action="store_true")
    handoff.set_defaults(func=cmd_import_checkpoint)

    cloud = sub.add_parser(
        "cloud-plan", help="只读核对云端依赖并生成manifest；不连接、不停止训练", parents=[json_flag]
    )
    cloud.add_argument("--config", type=Path, action="append")
    cloud.add_argument(
        "--run", type=Path, action="append", help="只读纳入当前恢复点/候选/预算快照；不授权接管"
    )
    cloud.add_argument("--manifest", type=Path, required=True)
    cloud.add_argument(
        "--reference",
        nargs=3,
        action="append",
        metavar=("PATH", "SHA256", "ROLE"),
        help="纳入明确身份的准备依赖，不修改原run或忽略其失效候选",
    )
    cloud.add_argument("--archive", type=Path)
    cloud.add_argument(
        "--restore-dependencies", type=Path, help="从已绑定包恢复依赖到本地数据根，不创建run"
    )
    cloud.add_argument("--dependency-archive", type=Path, help="现有SHA依赖包；不重新打包")
    cloud.add_argument("--dry-run", action="store_true")
    cloud.set_defaults(func=cmd_cloud_plan)

    stop = sub.add_parser("stop", help="对指定run的正常停止请求", parents=[json_flag])
    stop.add_argument("--run", required=True)
    stop.add_argument("--reason", default="user requested stop")
    stop.add_argument("--dry-run", action="store_true")
    stop.add_argument("--request-only", action="store_true", help="仅写owner停止请求，不发送OS信号")
    stop.set_defaults(func=cmd_stop)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CommandError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return exc.code
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except FileNotFoundError as exc:
        print(f"文件不存在: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except Exception as exc:
        import traceback

        print(f"执行失败: {exc!r}", file=sys.stderr)
        print(traceback.format_exc()[-900:], file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
