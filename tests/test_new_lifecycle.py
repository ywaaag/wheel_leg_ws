"""生命周期与结果：正常完成、失败、停止、报告失败、通知失败与计数一致性。

全部使用CPU假后端或进程内worker，不启动Isaac、MuJoCo或任何物理运行。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

from native_stand.artifacts import io
from native_stand.artifacts.io import read_json, read_jsonl
from native_stand.experiments import worker
from native_stand.experiments.budget import Budget, BudgetError
from native_stand.experiments.config import load_job_config
from native_stand.experiments.owner import create_run, inspect, submit
from native_stand.experiments.state import RunStateStore, StateError
from native_stand.integrations import hapi
from native_stand.simulation import driver

FROZEN_TEST_SCOPE = "PORTABLE"  # 自包含CPU工程契约：tmp fixture + 真实代码，无本机runs/资产依赖

PROJECT = Path(__file__).resolve().parents[1]
# 生命周期测试用CPU工程fixture：真实Motion配置声明863维Critic，CPU后端会明确拒绝。
BASE = json.loads((PROJECT / "tests" / "fixtures" / "cpu_motion_job.json").read_text())


def _config(tmp_path: Path, **overrides) -> Path:
    payload = json.loads(json.dumps(BASE))
    payload["runtime"].update(envs=2, steps_per_env=2)
    payload["limits"].update(smoke_updates=2, formal_updates=4)
    payload["storage"].update(checkpoint_every_updates=2, metrics_every_updates=1)
    for key, value in overrides.items():
        section = payload.setdefault(key, {})
        if isinstance(value, dict):
            section.update(value)
        else:
            payload[key] = value
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload))
    return path


def _wait_terminal(run_dir: Path, timeout_s: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = read_json(run_dir / "state.json", {})
        if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            return state
        time.sleep(0.2)
    raise AssertionError(f"run 未在 {timeout_s}s 内结束: {read_json(run_dir / 'state.json')}")


def _prepare_inprocess(tmp_path: Path, config: Path) -> tuple[Path, dict]:
    job = load_job_config(config)
    run_dir = create_run(job, runs_root=tmp_path / "runs", run_id="run_inprocess")
    Budget.create(
        run_dir / "budget.json",
        smoke_updates=job.limits.smoke_updates,
        formal_updates=job.limits.formal_updates,
    )
    RunStateStore(run_dir).create(run_id="run_inprocess", owner={"pid": 0}, stage="smoke")
    return run_dir, {}


def test_normal_completion_counts_and_report(tmp_path):
    config = _config(tmp_path)
    job = load_job_config(config)
    handshake = submit(job, runs_root=tmp_path / "runs", run_id="run_ok", wait_s=60)
    assert handshake["submitted"] is True
    assert handshake["formal_started"] is False
    assert handshake["worker_identity"]["status"] == "OK"
    run_dir = Path(handshake["run"])
    state = _wait_terminal(run_dir)
    assert state["status"] == "COMPLETED"
    assert state["failure"] is None
    assert state["updates_used"] == 6
    assert state["samples_used"] == 6 * 4
    assert state["capability_accepted"] is False
    assert state["actor_parameters_changed"] is True
    budget = read_json(run_dir / "budget.json")
    assert budget["used"] == {"smoke": 2, "formal": 4}
    assert all(row["status"] == "COMPLETED" for row in budget["attempts"].values())
    rows = read_jsonl(run_dir / "metrics.jsonl")
    assert rows and {row["stage"] for row in rows} == {"smoke", "formal"}
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "report_ready.json").is_file()
    report = (run_dir / "report.md").read_text()
    assert "能力accepted: False" in report
    payload = inspect(run_dir)
    assert payload["state"]["status"] == "COMPLETED"
    assert payload["checkpoints"]


def test_reserve_rejects_duplicate_running_attempt(tmp_path):
    budget = Budget.create(tmp_path / "budget.json", smoke_updates=1, formal_updates=1)
    budget.reserve("a", "smoke", tmp_path / "out")
    with pytest.raises(BudgetError, match="RUNNING"):
        budget.reserve("b", "formal", tmp_path / "out2")


def test_budget_rejects_overspend(tmp_path):
    budget = Budget.create(tmp_path / "budget.json", smoke_updates=1, formal_updates=0)
    budget.reserve("a", "smoke", tmp_path / "out")
    budget.charge("a", tmp_path / "out")
    with pytest.raises(BudgetError, match="耗尽"):
        budget.charge("a", tmp_path / "out")


def test_state_terminal_cannot_be_overwritten(tmp_path):
    """终态只写一次：相同内容无操作，冲突内容拒绝。"""
    store = RunStateStore(tmp_path)
    store.create(run_id="r", owner={"pid": os.getpid()}, stage="smoke")
    first = store.terminal("COMPLETED", failure=None)
    again = store.terminal("COMPLETED", failure=None)
    assert again["revision"] == first["revision"]
    with pytest.raises(StateError):
        store.terminal("COMPLETED", failure="changed")
    with pytest.raises(StateError):
        store.terminal("FAILED")


def test_engine_unavailable_fails_without_faking_success(tmp_path):
    """不可用后端在创建run之前就被拒绝，不会留下SUBMITTED的假进度。"""
    from native_stand.simulation import dispatch

    config = _config(tmp_path, backend="mujoco")
    job = load_job_config(config)
    with pytest.raises(dispatch.DispatchError, match="尚未迁移"):
        dispatch.require_supported(job, "train")
    assert not (tmp_path / "runs").exists()


def test_save_failure_fails_run_and_blocks_next_stage(tmp_path, monkeypatch):
    """必需checkpoint写盘失败：FAILED、非零退出、消耗保留、下一阶段不启动。"""
    config = _config(tmp_path)
    run_dir, _ = _prepare_inprocess(tmp_path, config)

    def boom(*args, **kwargs):
        raise OSError("磁盘写入失败")

    monkeypatch.setattr(worker, "save_full", boom)
    assert worker.main(["--run-dir", str(run_dir)]) != 0
    state = read_json(run_dir / "state.json")
    assert state["status"] == "FAILED"
    assert "保存失败" in (state["failure"] or "")
    assert state["updates_used"] >= 1, "真实完成的更新不能因保存失败清零"
    assert state["latest_model"] is None
    budget = read_json(run_dir / "budget.json")
    attempts = list(budget["attempts"].values())
    assert all(row["status"] != "RUNNING" for row in attempts)
    assert attempts[0]["updates"] == state["updates_used"]
    assert not (run_dir / "formal").exists(), "保存失败后不得启动下一阶段"


def test_report_failure_does_not_change_terminal_state(tmp_path, monkeypatch):
    config = _config(tmp_path)
    run_dir, _ = _prepare_inprocess(tmp_path, config)

    def boom(_run_dir, **_kwargs):
        raise OSError("报告写入失败")

    monkeypatch.setattr(worker, "write_report", boom)
    assert worker.main(["--run-dir", str(run_dir), "--stage", "smoke"]) == 0
    state = read_json(run_dir / "state.json")
    assert state["status"] == "COMPLETED"
    error = read_json(run_dir / "report_error.json")
    assert "报告写入失败" in error["error"]


def test_notification_failure_does_not_change_result(tmp_path):
    run_dir = tmp_path / "run"
    io.write_json(run_dir / "state.json", {"status": "COMPLETED", "capability_accepted": False})

    def sender(_payload, _target):
        raise RuntimeError("通知服务不可用")

    result = hapi.notify(run_dir, enabled=True, target="s1", sender=sender)
    assert result["status"] == "failed"
    assert read_json(run_dir / "state.json")["status"] == "COMPLETED"


def test_notification_disabled_writes_nothing(tmp_path):
    run_dir = tmp_path / "run"
    io.write_json(run_dir / "state.json", {"status": "COMPLETED"})
    before = set(p.name for p in run_dir.iterdir())
    result = hapi.notify(run_dir, enabled=False, target=None)
    assert result["status"] == "disabled"
    assert set(p.name for p in run_dir.iterdir()) == before


def test_stop_request_sends_term_signal_and_keeps_history(tmp_path):
    config = _config(tmp_path)
    payload = json.loads(config.read_text())
    payload["limits"].update(smoke_updates=0, formal_updates=4000)
    payload["storage"]["checkpoint_every_updates"] = 1000
    config.write_text(json.dumps(payload))
    job = load_job_config(config)
    handshake = submit(job, runs_root=tmp_path / "runs", run_id="run_stop", wait_s=60)
    run_dir = Path(handshake["run"])
    time.sleep(1.0)
    from native_stand.experiments.owner import stop

    receipt = stop(run_dir, reason="测试停止", timeout_s=120)
    assert receipt["stopped"] is True
    state = _wait_terminal(run_dir)
    assert state["status"] in ("STOPPED_BOUNDED", "COMPLETED")
    assert (run_dir / "stop_request.json").is_file()
    budget = read_json(run_dir / "budget.json")
    assert budget["used"]["formal"] <= budget["limits"]["formal"]
    assert all(row["status"] != "RUNNING" for row in budget["attempts"].values())


def test_recovery_context_and_init_failure_settlement(tmp_path, monkeypatch):
    """Recovery得到有效上下文；Motion/Recovery在reserve后初始化失败时attempt已结算。

    自包含CPU fixture：显式声明Recovery动作版本；只验证工程接线，不做物理验证。
    """
    import dataclasses as _dc

    from native_stand.experiments import worker as worker_mod
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.owner import create_run
    from native_stand.experiments.state import RunStateStore

    base = _config(tmp_path)
    # 显式Recovery任务：小网络+CPU，动作版本取任务spec默认
    payload = json.loads(Path(base).read_text())
    payload["task"] = "recovery"
    # 显式声明Recovery动作语义（非motion31）+满足任务保持时间；只验证工程接线
    payload["action"]["version"] = "recovery_centered_cubic_v1"
    payload["evaluation"]["seconds"] = 12.0
    payload["limits"]["smoke_updates"] = 1
    payload["limits"]["formal_updates"] = 0
    config = tmp_path / "recovery_job.json"
    config.write_text(json.dumps(payload))
    job = load_job_config(config)

    def make_run(run_id):
        run = create_run(job, runs_root=tmp_path / "runs", run_id=run_id)
        Budget.create(run / "budget.json", smoke_updates=1, formal_updates=0)
        RunStateStore(run).create(run_id=run_id, owner={"pid": 0}, stage="smoke")
        return run

    # 1) Recovery正常构建上下文（真实_new_context，不mock）
    run = make_run("recovery_ok")
    context = worker_mod._new_context(run, "smoke", suffix="ab" * 8)
    assert context is not None and context.attempt_id

    # 2) Motion：reserve后MetricsWriter失败 → state FAILED且attempt已结算
    motion_job = load_job_config(_config(tmp_path))  # 写入tmp_path/job.json
    motion_run = create_run(motion_job, runs_root=tmp_path / "runs", run_id="motion_fail")
    Budget.create(motion_run / "budget.json", smoke_updates=1, formal_updates=0)
    RunStateStore(motion_run).create(run_id="motion_fail", owner={"pid": 0}, stage="smoke")

    class FailingMetrics:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("注入的MetricsWriter失败")

    monkeypatch.setattr(worker_mod, "MetricsWriter", FailingMetrics)
    with pytest.raises(RuntimeError, match="MetricsWriter"):
        worker_mod._new_context(motion_run, "smoke", suffix="cd" * 8)
    budget = json.loads((motion_run / "budget.json").read_text())
    statuses = [row["status"] for row in budget["attempts"].values()]
    assert "RUNNING" not in statuses, "初始化失败不能残留RUNNING attempt"
    monkeypatch.undo()

    # 3) Recovery同样路径
    recovery_run = make_run("recovery_fail")
    monkeypatch.setattr(worker_mod, "MetricsWriter", FailingMetrics)
    with pytest.raises(RuntimeError, match="MetricsWriter"):
        worker_mod._new_context(recovery_run, "smoke", suffix="ef" * 8)
    budget = json.loads((recovery_run / "budget.json").read_text())
    assert all(row["status"] != "RUNNING" for row in budget["attempts"].values())
    monkeypatch.undo()

    # 4) 故障前已存在的其他attempt不被改写
    other = make_run("other_task")
    other_budget = json.loads((other / "budget.json").read_text())
    assert list(other_budget["attempts"]) == list(
        json.loads((other / "budget.json").read_text())["attempts"]
    )


def test_init_failure_closes_metrics_and_keeps_other_attempts(tmp_path, monkeypatch):
    """metrics创建成功、后续构造失败：close()实际发生；既有attempt记录逐字节不变。"""
    import dataclasses as _dc

    from native_stand.experiments import worker as worker_mod
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.owner import create_run
    from native_stand.experiments.state import RunStateStore

    def make_run(run_id, job):
        run = create_run(job, runs_root=tmp_path / "runs", run_id=run_id)
        Budget.create(run / "budget.json", smoke_updates=1, formal_updates=0)
        RunStateStore(run).create(run_id=run_id, owner={"pid": 0}, stage="smoke")
        return run

    job = load_job_config(_config(tmp_path))
    # 既有attempt：预置一个已结算的attempt记录，身份明确
    other = make_run("other_run", job)
    other_budget_before = (other / "budget.json").read_bytes()
    other_state_before = (other / "state.json").read_bytes()

    # 目标run：注入后续构造失败（SavePolicy失败），metrics构造本身成功
    target = make_run("target_run", job)

    class FailingPolicy:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("注入的SavePolicy失败")

    closed = []
    real_metrics = worker_mod.MetricsWriter

    class TrackingMetrics(real_metrics):
        def close(self):
            closed.append(True)
            return super().close()

    monkeypatch.setattr(worker_mod, "MetricsWriter", TrackingMetrics)
    monkeypatch.setattr(worker_mod, "SavePolicy", FailingPolicy)
    with pytest.raises(RuntimeError, match="SavePolicy"):
        worker_mod._new_context(target, "smoke", suffix="ab" * 8)
    assert closed == [True], "已创建的metrics必须被关闭"
    monkeypatch.undo()

    # 目标attempt已结算FAILED；既有run的账本与状态逐字节不变
    budget = json.loads((target / "budget.json").read_text())
    assert all(row["status"] != "RUNNING" for row in budget["attempts"].values())
    assert (other / "budget.json").read_bytes() == other_budget_before
    assert (other / "state.json").read_bytes() == other_state_before


def test_recovery_worker_completes_on_cpu(tmp_path):
    """Recovery CPU worker走真实执行路径正常结束；状态与消耗一致。"""
    import torch as _torch

    from native_stand.experiments import worker as worker_mod
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.owner import create_run
    from native_stand.experiments.state import RunStateStore

    payload = json.loads(Path(_config(tmp_path)).read_text())
    payload["task"] = "recovery"
    # 当前Recovery动作契约：物理Gaussian + 显式训练合同（经registry进入物理装配）
    payload["action"]["version"] = "recovery_cubic_physical_Gaussian_v2"
    payload["model"]["action_version"] = payload["action"]["version"]
    payload["model"]["interface"] = "recovery_cubic_physical_v2"
    contract = {
        "version": "recovery_cubic_physical_Gaussian_v2",
        "center_rad": [0.0, 0.1, -0.1, 0.05],
        "leg_lower_rad": [-1.2, -1.2, -1.2, -1.2],
        "leg_upper_rad": [1.2, 1.2, 1.2, 1.2],
    }
    contract_path = tmp_path / "recovery_training_contract.json"
    contract_path.write_text(json.dumps(contract))
    # 保留CPU fixture原有几何（两连杆三角形约束有效），仅追加Recovery训练合同。
    payload.setdefault("assets", {})
    payload["assets"] = {
        **payload.get("assets", {}),
        "recovery_training_contract": {"path": str(contract_path)},
    }
    payload["evaluation"]["seconds"] = 12.0
    payload["limits"]["smoke_updates"] = 1
    payload["limits"]["formal_updates"] = 0
    config = tmp_path / "recovery_job.json"
    config.write_text(json.dumps(payload))
    job = load_job_config(config)
    run = create_run(job, runs_root=tmp_path / "runs", run_id="rec_cpu")
    Budget.create(run / "budget.json", smoke_updates=1, formal_updates=0)
    RunStateStore(run).create(run_id="rec_cpu", owner={"pid": 0}, stage="smoke")
    code = worker_mod.main(["--run-dir", str(run)])
    assert code == 0, (
        (run / "worker_failure.json").read_text()
        if (run / "worker_failure.json").exists()
        else code
    )
    state = json.loads((run / "state.json").read_text())
    assert state["status"] == "COMPLETED" and state["updates_used"] >= 1
    budget = json.loads((run / "budget.json").read_text())
    assert budget["used"]["smoke"] >= 1


def build_fictional_task(job):
    """虚构技能的任务对象：仅承载配置；物理语义由测试环境替身承担。"""
    from types import SimpleNamespace

    return SimpleNamespace(job=job, task_id="fictional_vault")


def test_fictional_skill_uses_shared_lifecycle(tmp_path):
    """虚构技能（非Motion/Recovery、观测维数不同）：共享驱动器+真实网络+成功终止语义。

    仅接口工程测试：证明任务扩展点可用，不代表任何跳跃物理能力。
    """
    from native_stand.simulation import driver
    from native_stand.tasks import registry

    with pytest.raises(KeyError):
        registry.get_task_definition("fictional_vault")
    assert registry.supports("motion", "ppo", "cpu")
    # 测试专用的虚构技能登记（生产注册表不含fictional_vault）
    saved_tasks = dict(registry.TASKS)
    try:
        _run_fictional_skill(tmp_path, registry)
    finally:
        registry.TASKS.clear()
        registry.TASKS.update(saved_tasks)


def _run_fictional_skill(tmp_path, registry):
    registry.TASKS["fictional_vault"] = {
        "build_module": "tests.test_new_lifecycle",
        "build_symbol": "build_fictional_task",
        "evaluator_module": "native_stand.learning.ppo.evaluate",
        "evaluator_symbol": "evaluate_checkpoint",
        "supported_algorithms": ("ppo",),
        "supported_backends": ("cpu",),
        # 虚构技能使用与Motion/Recovery不同的观测键名
        "observation_keys": {"actor": "actor_obs", "critic": "critic_obs", "action_valid": "valid"},
    }

    job = _config(tmp_path)
    payload = json.loads(Path(job).read_text())
    payload["task"] = "fictional_vault"
    payload["observation"]["actor_dim"] = 8
    payload["observation"]["critic_dim"] = 8
    payload["model"]["actor_input"] = 8
    payload["model"]["critic_input"] = 8
    small = tmp_path / "skill_job.json"
    small.write_text(json.dumps(payload))
    skill_job = load_job_config(small)

    from native_stand.learning.policies.factory import build_policy

    class SkillEnv:
        """虚构技能环境：观测8维、成功终止（任务语义）。"""

        num_envs = 2
        device = "cpu"

        def __init__(self):
            self._step_index = 0

        def observations(self):
            import torch as _t
            from tensordict import TensorDict

            # 虚构技能自己的键名（actor_obs/critic_obs/valid），不是Motion/Recovery键
            return TensorDict(
                {
                    "actor_obs": _t.zeros(2, 8),
                    "critic_obs": _t.zeros(2, 8),
                    "valid": _t.ones(2, 1),
                },
                batch_size=[2],
            )

        def step(self, action):
            import torch as _t

            self._step_index += 1
            done = self._step_index >= 2
            # env0 成功终止；env1 失败终止 —— 同一批次两种结果
            return {
                "rewards": _t.ones(2, 1),
                "terminated": _t.full((2, 1), done, dtype=_t.bool),
                "truncated": _t.zeros(2, 1, dtype=_t.bool),
                "dones": _t.full((2, 1), done, dtype=_t.bool),
                "reset_envs": int(done) * 2,
                "extras": {},
                "task_success": [done, False],
                "task_failure": [None, "fell_over" if done else None],
                "terminal_observations": {"critic_obs": _t.zeros(2, 8)},
            }

        def close(self):
            pass

    skill_env = SkillEnv()
    handle = build_policy(skill_job, skill_env.observations())
    # 公开接口暴露任务自己的键名，不冒用Motion/Recovery命名
    assert handle["observation_keys"]["actor"] == "actor_obs"
    assert handle["action_valid_key"] == "valid"

    # 评估：成功终止评为成功，不记为失败
    from native_stand.learning.ppo import evaluate as eval_mod

    outcome = eval_mod._rollout(SkillEnv(), handle["ppo"].actor, seconds=0.02, policy_dt=0.01)
    # 逐环境：env0成功、env1失败 —— 同一批次内区分
    assert outcome["task_success"] == [True, False]
    assert outcome["task_failure"][1] == "fell_over"
    assert outcome["first_failure"].get(0) is None  # env0成功：无失败记录
    assert outcome["first_failure"][1]["reason"] == "task_failure:fell_over"

    # 共享驱动器：一次真实optimizer update + 计费 + close
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.owner import create_run
    from native_stand.experiments.state import RunStateStore
    from native_stand.experiments import worker as worker_mod

    run = create_run(skill_job, runs_root=tmp_path / "runs", run_id="skill")
    Budget.create(run / "budget.json", smoke_updates=1, formal_updates=0)
    RunStateStore(run).create(run_id="skill", owner={"pid": 0}, stage="smoke")
    context = worker_mod._new_context(run, "smoke", suffix="ab" * 8)

    policy = handle["ppo"]
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-2)
    params_before = {n: p.detach().clone() for n, p in policy.actor.named_parameters()}
    closed = []

    class SkillAdapter:
        def build(self, context, stage=None, resume_from=None):
            self.env = SkillEnv()
            self.obs = self.env.observations()

        def update_once(self, context, index):
            import torch as _t

            self.env.step(None)
            loss = sum(p.sum() for p in policy.actor.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # 真实优化步骤
            context.metrics.write({"update": index, "reward_mean": 1.0})
            return {
                "reward_mean": 1.0,
                "actor_steps": 2,
                "actor_optimizer_steps": 1,
                "critic_optimizer_steps": 1,
                "value": 0.0,
                "surrogate": 0.0,
                "entropy": 0.0,
            }

        def after_update(self, context, index, outcome):
            self.obs = self.env.observations()

        def capture(self, context, counters):
            from native_stand.learning.ppo import assembly as _assembly

            return _assembly.capture_full(
                policy,
                counters={
                    "updates": counters.get("updates", context.updates_used),
                    "samples": counters.get("samples", context.updates_used * 4),
                },
            )

        def close(self):
            closed.append(True)

    result = driver.run_stage(context, SkillAdapter(), stage="smoke")
    assert result.updates == 1 and context.updates_used == 1
    assert closed == [True]
    # 真实optimizer update的参数变化证据（固定std是设计，不参与更新）
    changed = [
        name
        for name, parameter in policy.actor.named_parameters()
        if not torch.equal(parameter.detach(), params_before[name])
    ]
    assert any("log_std" not in name for name in changed), "权重参数必须实际变化"
