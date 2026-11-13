"""一个run只由一个owner管理生命周期。

职责划分：
- owner：建run目录、写run.json/初始state、一次有界启动握手、停止请求、退出确认与终态。
- worker：构建环境、执行训练、回报实际update与checkpoint引用、退出。
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from native_stand.artifacts import evaluation as evaluation_index
from native_stand.artifacts import source_identity
from native_stand.artifacts.io import (
    file_sha256,
    new_dir,
    read_json,
    read_jsonl,
    write_json,
    write_json_new,
)
from native_stand.artifacts.retention import read_inventory, write_inventory

from .budget import Budget
from .config import JobConfig
from .plan import build_plan
from .state import RunStateStore, write_stop_request

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = PROJECT_ROOT / "runs"
WORKER_MODULE = "native_stand.experiments.worker"


def experiment_run_id(purpose: str, *, now: datetime | None = None) -> str:
    """按Asia/Shanghai时间生成run目录名。"""
    moment = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    stamp = f"{moment.year}_{moment.month}_{moment.day}-{moment.hour:02d}_{moment.minute:02d}"
    cleaned = purpose.strip().replace(" ", "_")
    if not cleaned:
        raise ValueError("purpose不能为空")
    return f"{stamp}__{cleaned}"


def process_state(pid: int) -> dict[str, Any]:
    """读取进程cmdline与启动tick，用于识别PID复用。"""
    base = Path("/proc") / str(pid)
    try:
        cmdline = (base / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        stat = (base / "stat").read_text()
    except FileNotFoundError:
        return {"alive": False}
    fields = stat.rsplit(")", 1)[-1].split()
    return {"alive": True, "cmdline": cmdline.strip(), "start_ticks": fields[19]}


def process_identity(pid: int, token: str, *, start_ticks: str | None = None) -> dict[str, Any]:
    """核实该PID确实是本run的worker：命令行匹配，且启动tick未必被PID复用。"""
    state = process_state(pid)
    if not state.get("alive"):
        return {"status": "DEAD", "pid": pid}
    if token not in state["cmdline"]:
        return {"status": "MISMATCH", "pid": pid, "cmdline": state["cmdline"]}
    if start_ticks is not None and str(state.get("start_ticks")) != str(start_ticks):
        return {
            "status": "MISMATCH",
            "pid": pid,
            "start_ticks": state.get("start_ticks"),
            "expected_start_ticks": start_ticks,
            "reason": "启动tick不同，PID已被复用",
        }
    return {"status": "OK", "pid": pid, "start_ticks": state["start_ticks"]}


def run_paths(run_dir: str | Path) -> dict[str, Path]:
    directory = Path(run_dir)
    return {
        "run": directory,
        "checkpoints": directory / "checkpoints",
        "evaluations": directory / "evaluations",
        "traces": directory / "traces",
        "source": directory / "source",
    }


def create_run(
    job: JobConfig,
    *,
    runs_root: str | Path = RUNS_ROOT,
    purpose: str | None = None,
    run_id: str | None = None,
) -> Path:
    """创建新run目录与run.json；不创建预算、不启动worker。"""
    identifier = run_id or experiment_run_id(purpose or job.name)
    directory = Path(runs_root) / identifier
    if Path(runs_root).resolve() == RUNS_ROOT.resolve() or job.assets.get("path_mapping"):
        from native_stand.artifacts.model_refs import ModelIndex, initialize_fixed
        from native_stand.artifacts.runtime_paths import RuntimePaths

        initialize_fixed(
            ModelIndex(Path(runs_root) / "models.json"),
            PROJECT_ROOT / "configs/model_refs.json",
            paths=RuntimePaths.from_assets(job.assets),
        )
    paths = run_paths(directory)
    new_dir(directory)
    for key in ("checkpoints", "evaluations", "traces", "source"):
        new_dir(paths[key])
    identity = source_identity.record(job.source_root)
    write_json(paths["source"] / "identity.json", identity)
    record = {
        "schema_version": 1,
        "run_id": identifier,
        "logical_run_id": __import__("uuid").uuid4().hex,
        "model_index": str((Path(runs_root) / "models.json").resolve()),
        "created_unix_s": time.time(),
        "config": job.as_run_record(),
        "plan": build_plan(job),
        "source": identity,
        "limits": {
            "smoke_updates": job.limits.smoke_updates,
            "formal_updates": job.limits.formal_updates,
            "sample_limit": job.limits.sample_limit,
            "evaluation_wall_s": job.limits.evaluation_wall_s,
            "chain_wall_s": job.limits.chain_wall_s,
            "max_output_bytes": job.storage.max_output_bytes,
        },
        "save_policy": {
            "checkpoint_every_updates": job.storage.checkpoint_every_updates,
            "keep_recent_checkpoints": job.storage.keep_recent_checkpoints,
            "metrics_every_updates": job.storage.metrics_every_updates,
            "milestones": list(job.storage.milestones),
            "tensorboard": job.storage.tensorboard,
            "save_traces": job.storage.save_traces,
        },
        "protocol": {
            "version": job.evaluation.protocol_version,
            "envs": job.evaluation.envs,
            "scored_envs": job.evaluation.scored_envs,
            "seconds": job.evaluation.seconds,
        },
    }
    write_json(directory / "run.json", record)
    return directory


def submit(
    job: JobConfig,
    *,
    runs_root: str | Path = RUNS_ROOT,
    run_id: str | None = None,
    purpose: str | None = None,
    wait_s: float = 30.0,
    device: str | None = None,
    distill_rounds: int | None = None,
    stop_request: dict[str, Any] | None = None,
    max_updates: int | None = None,
) -> dict[str, Any]:
    """创建run并启动owner管理的worker，完成一次有界握手后返回。

    返回SUBMITTED回执；formal是否已经开始由worker事件决定，不在这里假定。
    """
    if not 0 <= wait_s <= 600:
        raise ValueError("握手等待必须在0..600秒")
    if max_updates is not None and (type(max_updates) is not int or max_updates <= 0):
        raise ValueError("max_updates必须为正整数")
    # preflight与实际执行使用同一个解析结果；只在配置边界判断一次。
    from native_stand.simulation import dispatch

    service = dispatch.execution_service(job)
    if job.algorithm == "distillation" and job.supervision is not None:
        from .supervision import preflight

        preflight(job)
    if (
        job.algorithm != "distillation"
        and service["service"] != "unsupported"
        and job.backend != "cpu"
    ):
        module = __import__(service["module"], fromlist=["preflight"])
        prepare = getattr(module, "prepare_submission", None)
        if prepare is not None:
            job = prepare(job)
        module.preflight(job)
    directory = create_run(job, runs_root=runs_root, purpose=purpose, run_id=run_id)
    run_paths(directory)
    if distill_rounds is not None:
        # BC不继承PPO可执行额度：smoke/formal为0，总额度只在distill。
        Budget.create(
            directory / "budget.json",
            smoke_updates=0,
            formal_updates=0,
            distill_rounds=distill_rounds,
        )
    else:
        Budget.create(
            directory / "budget.json",
            smoke_updates=job.limits.smoke_updates,
            formal_updates=job.limits.formal_updates,
            sample_limit=job.limits.sample_limit,
            evaluation_wall_s=job.limits.evaluation_wall_s,
        )
    store = RunStateStore(directory)
    first_stage = (
        "distill"
        if distill_rounds is not None
        else ("smoke" if job.limits.smoke_updates else "formal")
    )
    store.create(
        run_id=directory.name,
        owner={"pid": os.getpid(), "module": Path(__file__).name},
        stage=first_stage,
    )
    if stop_request:
        # 预注册停止请求：在worker启动前落盘，保证轮边界语义确定。
        from native_stand.artifacts.io import write_json as _wj

        _wj(
            directory / "stop_request.json",
            {
                "requested_unix_s": time.time(),
                "applies_to_run": str(directory),
                **stop_request,
            },
        )
    reader, writer = os.pipe()
    # 不传--stage：由worker按预算依次执行smoke与formal，避免只跑第一阶段。
    command = [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--run-dir",
        str(directory),
        "--handshake-fd",
        str(writer),
    ]
    if max_updates is not None:
        command.extend(["--max-updates", str(max_updates)])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(job.source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    device_name = device or job.runtime.device
    # CPU作业不得把可见性改回某块GPU。
    environment["CUDA_VISIBLE_DEVICES"] = (
        "" if device_name.startswith("cpu") else device_name.split(":")[-1]
    )
    log = (directory / "worker.log").open("x")
    try:
        child = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(writer,),
            start_new_session=True,
        )
    except BaseException as exc:
        os.close(reader)
        os.close(writer)
        log.close()
        store.terminal("FAILED", failure=f"worker spawn failed: {exc!r}")
        raise
    os.close(writer)
    log.close()
    identity = process_state(child.pid)
    handshake_message: dict[str, Any] = {"status": "UNKNOWN", "reason": "handshake not delivered"}
    deadline = time.monotonic() + wait_s
    messages: list[dict[str, Any]] = []
    try:
        ready, _, _ = select.select([reader], [], [], wait_s)
        if ready:
            chunk = os.read(reader, 65536)
            for line in chunk.split(b"\n"):
                if not line.strip():
                    continue
                try:
                    messages.append(json.loads(line))
                except (ValueError, UnicodeError):
                    messages.append({"status": "UNKNOWN", "reason": "malformed handshake"})
            if messages:
                handshake_message = messages[-1]
    finally:
        os.close(reader)
    exit_code = child.poll()
    if not messages and ready and not chunk and exit_code is None:
        # EOF can precede waitpid visibility by a few instructions. Keep the
        # original bounded handshake deadline; never signal the child.
        from contextlib import suppress

        with suppress(subprocess.TimeoutExpired):
            exit_code = child.wait(timeout=max(0.0, deadline - time.monotonic()))
    if not messages and exit_code is not None:
        handshake_message = {
            "status": "FAILED",
            "reason": "worker exited before any handshake",
            "exit_code": exit_code,
        }
    started_events = [row for row in messages if row.get("first_update_verified")]
    handshake = {
        "run": str(directory.resolve()),
        "worker_pid": child.pid,
        "worker_command": command,
        "worker_start_ticks": identity.get("start_ticks"),
        "wait_s": wait_s,
        "remaining_wait_s": max(0.0, deadline - time.monotonic()),
        "worker_identity": process_identity(
            child.pid, str(directory), start_ticks=identity.get("start_ticks")
        ),
        "worker_reported": handshake_message,
        "worker_messages": messages,
        "worker_exit_code": exit_code,
        # 启动成功 = 收到worker自己的消息；formal开始只由真实update事件确认。
        "worker_started": bool(messages) and handshake_message.get("status") != "FAILED",
        "submitted": True,
        "formal_started": bool(started_events),
        "handshake_unix_s": time.time(),
    }
    write_json_new(directory / "startup_handshake.json", handshake)
    # worker没留下任何消息就退出：此刻没有其他写入者，launcher必须记录真实失败。
    if not messages and exit_code is not None:
        try:
            if store.read().get("status") == "SUBMITTED":
                store.terminal(
                    "FAILED",
                    failure=f"worker exited before handshake (exit={exit_code})",
                    allow_foreign=True,
                )
        except Exception:
            pass
    # 其余情况launcher不写运行状态：worker用claim()接管，避免两个写入者。
    return handshake


def inspect(run_dir: str | Path, *, metrics_tail: int = 5) -> dict[str, Any]:
    """只读查看运行状态、模型引用与评估索引。"""
    directory = Path(run_dir)
    run = read_json(directory / "run.json")
    if run is None:
        raise FileNotFoundError(f"不是新结构run目录: {directory}")
    store = RunStateStore(directory)
    state = store.read()
    budget = read_json(directory / "budget.json", {})
    return {
        "run": str(directory.resolve()),
        "run_id": run.get("run_id"),
        "task": run.get("task"),
        "algorithm": run.get("algorithm"),
        "backend": run.get("backend"),
        "config": (run.get("config") or {}).get("name"),
        "limits": run.get("limits"),
        "used": budget.get("used"),
        "state": state,
        "metrics_tail": read_jsonl(directory / "metrics.jsonl")[-metrics_tail:],
        "evaluations": evaluation_index.read_index(directory / "evaluations"),
        "checkpoints": (read_json(directory / "checkpoints" / "inventory.json", {}) or {}).get(
            "checkpoints", []
        ),
    }


def _candidate_pids(directory: Path, state: dict[str, Any]) -> list[int]:
    """候选进程：state.owner与启动握手记录，去重后按顺序核验。"""
    candidates: list[int] = []
    owner_pid = (state.get("owner") or {}).get("pid")
    if owner_pid:
        candidates.append(int(owner_pid))
    handshake = read_json(directory / "startup_handshake.json", {}) or {}
    worker_pid = handshake.get("worker_pid")
    if worker_pid and int(worker_pid) not in candidates:
        candidates.append(int(worker_pid))
    return candidates


def stop(
    run_dir: str | Path,
    *,
    reason: str = "user requested stop",
    timeout_s: float = 120.0,
    dry_run: bool = False,
    request_only: bool = False,
) -> dict[str, Any]:
    """对指定run已核验的owner提出正常停止请求。

    只发送SIGTERM；终态由worker自己保存并写入。
    """
    directory = Path(run_dir)
    state = RunStateStore(directory).read()
    if not state:
        raise FileNotFoundError(f"state.json不存在: {directory}")
    status = state.get("status")
    if status in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
        return {
            "run": str(directory.resolve()),
            "status": status,
            "stopped": False,
            "reason": "run already terminal",
        }
    identity: dict[str, Any] = {"status": "UNKNOWN"}
    pid = 0
    for candidate in _candidate_pids(directory, state):
        checked = process_identity(candidate, str(directory))
        if checked.get("status") == "OK":
            pid, identity = candidate, checked
            break
        identity = checked
    if dry_run:
        return {
            "run": str(directory.resolve()),
            "status": status,
            "would_signal": pid,
            "request_recorded": False,
            "signal_sent": False,
            "exited": False,
            "stopped": False,
            "identity": identity,
            "dry_run": True,
        }
    if identity.get("status") != "OK":
        write_stop_request(directory, reason=reason, requester="wheel-leg stop")
        return {
            "run": str(directory.resolve()),
            "status": status,
            "stopped": False,
            "request_recorded": True,
            "signal_sent": False,
            "exited": False,
            "identity": identity,
            "reason": "owner identity not verified; request recorded only",
        }
    write_stop_request(directory, reason=reason, requester="wheel-leg stop")
    if request_only:
        return dict(
            run=str(directory.resolve()),
            status=status,
            stopped=False,
            request_recorded=True,
            signal_sent=False,
            exited=False,
            identity=identity,
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = RunStateStore(directory).read()
        if current.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            return {
                "run": str(directory.resolve()),
                "status": current["status"],
                "request_recorded": True,
                "signal_sent": True,
                "exited": True,
                "stopped": True,
                "identity": identity,
            }
        if not process_state(pid).get("alive"):
            break
        time.sleep(2.0)
    current = RunStateStore(directory).read()
    return {
        "run": str(directory.resolve()),
        "status": current.get("status"),
        "request_recorded": True,
        "signal_sent": True,
        "exited": False,
        "stopped": False,
        "identity": identity,
        "note": "signal sent; process exit not observed yet",
    }


def reconcile(run_dir: str | Path) -> dict[str, Any]:
    """worker已退出但未写终态时，由owner补写一次终态并返回。"""
    directory = Path(run_dir)
    store = RunStateStore(directory)
    state = store.read()
    if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
        return state
    pid = int((state.get("owner") or {}).get("pid") or 0)
    if pid and process_state(pid).get("alive"):
        return state
    book = read_json(directory / "budget.json", {})
    running = [
        (key, row)
        for key, row in (book.get("attempts") or {}).items()
        if row.get("status") == "RUNNING"
    ]
    budget = Budget(directory / "budget.json")
    for attempt, row in running:
        budget.finalize(
            attempt,
            row["output_dir"],
            "TERMINATED",
            reason="worker exited without terminal report",
        )
    return store.terminal(
        "FAILED", failure="worker exited without terminal state", allow_foreign=True
    )


def write_run_artifacts(run_dir: str | Path, *, checkpoints: list[dict[str, Any]]) -> None:
    """写入checkpoint清单；由worker在每次保存后调用。"""
    write_inventory(run_dir, checkpoints)


def _next_attempt_suffix(directory: Path, stage: str) -> str:
    # 唯一标识：不按目录数量推算，避免重复恢复撞日志文件。
    return f"{stage}_r{time.time_ns() % 10_000_000:07d}_{uuid.uuid4().hex[:6]}"


def resume_run(
    run_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    wait_s: float = 30.0,
    device: str | None = None,
    max_updates: int | None = None,
) -> dict[str, Any]:
    """在同一个逻辑run上续训：同一份预算、新attempt、旧记录不动。"""
    if max_updates is not None and max_updates <= 0:
        raise ValueError("有界resume更新数必须为正")
    directory = Path(run_dir).resolve()
    record = read_json(directory / "run.json")
    if record is None:
        raise FileNotFoundError(f"不是新结构run目录: {directory}")
    state = RunStateStore(directory).read()
    if state.get("status") in ("RUNNING", "SUBMITTED"):
        raise ValueError("该run仍有未结束的attempt；先确认原进程已退出")
    budget = Budget(directory / "budget.json")
    limits = budget.limits()
    used = budget.used()
    algorithm = record["config"].get("algorithm") or "ppo"
    candidates = ("distill",) if algorithm == "distillation" else ("smoke", "formal")
    stage = next(
        (name for name in candidates if int(used.get(name, 0)) < int(limits.get(name, 0))),
        None,
    )
    if stage is None and algorithm == "distillation":
        flow = read_json(directory / "supervision" / "supervision_state.json")
        recipe = record["config"].get("supervision")
        if (
            recipe
            and flow
            and flow["phase_cursor"] < len(recipe["phases"]) * recipe.get("repetitions", 1)
        ):
            stage = "distill"  # Finish committed, already charged fit/export; no fresh rounds.
    if stage is None:
        return {
            "run": str(directory),
            "resumed": False,
            "status": "NOT_RESUMABLE",
            "reason": "预算已耗尽；resume不刷新额度，新额度请创建新任务配置",
            "limits": limits,
            "used": used,
        }
    from .config import job_from_record

    job = job_from_record(record["config"])
    from native_stand.simulation.dispatch import execution_service

    service = execution_service(job)
    if (
        job.algorithm != "distillation"
        and job.backend != "cpu"
        and service["service"] != "unsupported"
    ):
        module = __import__(service["module"], fromlist=["preflight"])
        module.preflight(job)
    selected = Path(checkpoint).resolve() if checkpoint else None
    if selected is None and stage != "distill":
        rows = [row for row in read_inventory(directory) if Path(row["checkpoint"]).is_file()]
        if not rows:
            return {
                "run": str(directory),
                "resumed": False,
                "status": "NOT_RESUMABLE",
                "reason": "没有可用的完整恢复状态；best与selected不作为默认续训回退点",
            }
        selected = Path(max(rows, key=lambda row: int(row.get("update") or -1))["checkpoint"])
    if stage != "distill":
        # BC的恢复点由worker从state登记的latest加载，不走inventory校验。
        from native_stand.artifacts.checkpoint import load_full

        load_full(selected)  # 只接受完整恢复状态
    # distill阶段：恢复点由worker从state登记的latest加载，这里没有selected。
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(job.source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    device_name = device or job.runtime.device
    # CPU作业不得把可见性改回某块GPU。
    environment["CUDA_VISIBLE_DEVICES"] = (
        "" if device_name.startswith("cpu") else device_name.split(":")[-1]
    )
    suffix = _next_attempt_suffix(directory, stage)
    previous_stop = directory / "stop_request.json"
    if previous_stop.exists():
        archived = directory / f"stop_request_{suffix}.json"
        if archived.exists():
            raise ValueError("停止请求历史目标已存在")
        previous_stop.rename(archived)
    reader, writer = os.pipe()
    command = [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--run-dir",
        str(directory),
        "--handshake-fd",
        str(writer),
        "--attempt-suffix",
        suffix,
    ]
    if stage != "distill":
        command += ["--stage", stage, "--resume-checkpoint", str(selected)]
    if max_updates is not None:
        command += ["--max-updates", str(max_updates)]
    log = (directory / f"worker_{suffix}.log").open("x")
    try:
        child = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(writer,),
            start_new_session=True,
        )
    except BaseException as exc:
        os.close(reader)
        os.close(writer)
        log.close()
        raise RuntimeError(f"worker spawn failed: {exc!r}") from exc
    os.close(writer)
    log.close()
    identity = process_state(child.pid)
    messages: list[dict[str, Any]] = []
    try:
        ready, _, _ = select.select([reader], [], [], wait_s)
        if ready:
            chunk = os.read(reader, 65536)
            for line in chunk.split(b"\n"):
                if line.strip():
                    try:
                        messages.append(json.loads(line))
                    except (ValueError, UnicodeError):
                        messages.append({"status": "UNKNOWN", "reason": "malformed handshake"})
    finally:
        os.close(reader)
    exit_code = child.poll()
    reported = (
        messages[-1] if messages else {"status": "UNKNOWN", "reason": "handshake not delivered"}
    )
    if not messages and exit_code is not None:
        reported = {
            "status": "FAILED",
            "reason": "worker exited before any handshake",
            "exit_code": exit_code,
        }
    handshake = {
        "run": str(directory),
        "resumed": True,
        "stage": stage,
        "attempt_suffix": suffix,
        "checkpoint": str(selected) if selected else None,
        "checkpoint_sha256": file_sha256(selected) if selected else None,
        "worker_pid": child.pid,
        "worker_start_ticks": identity.get("start_ticks"),
        "worker_identity": process_identity(
            child.pid, str(directory), start_ticks=identity.get("start_ticks")
        ),
        "worker_reported": reported,
        "worker_messages": messages,
        "worker_exit_code": exit_code,
        "worker_started": bool(messages) and reported.get("status") != "FAILED",
        "budget_limits": limits,
        "budget_used": used,
        "budget_refreshed": False,
        "handshake_unix_s": time.time(),
    }
    write_json_new(directory / f"resume_{suffix}.json", handshake)
    return handshake
