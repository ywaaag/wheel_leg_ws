"""第三阶段修复：统一执行分派、完整初始化、恢复不依赖原配置、探索单位、inventory限定。"""

from __future__ import annotations

import json
import time
import types
from pathlib import Path

import pytest

from native_stand.artifacts import model_refs
from native_stand.artifacts.checkpoint import save_full
from native_stand.artifacts.retention import SavePolicy, plan_prune, prune, write_inventory
from native_stand.experiments import config as job_config
from native_stand.experiments.owner import create_run, submit
from native_stand.experiments.state import RunStateStore
from native_stand.simulation import dispatch

FROZEN_TEST_SCOPE = "PORTABLE"  # 自包含CPU工程契约：tmp fixture + 真实代码，无本机runs/资产依赖

PROJECT = Path(__file__).resolve().parents[1]
ACTIVE_CONFIG = (
    PROJECT.parent
    / "wheel_leg_native_train"
    / "configs"
    / "jobs"
    / "motion442_scale_corrected_resume.json"
)
CPU_FIXTURE = PROJECT / "tests" / "fixtures" / "cpu_motion_job.json"

torch = pytest.importorskip("torch")


def _cpu_job(tmp_path: Path, **overrides):
    payload = json.loads(CPU_FIXTURE.read_text())
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key].update(value)
        else:
            payload[key] = value
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload))
    return job_config.load_job_config(path)


def _full_state(update: int = 5, sentinel: float = 0.0):
    return {
        "actor": {"w": torch.full((4, 31), sentinel)},
        "critic": {"w": torch.full((4, 64), sentinel)},
        "optimizer": {"state": {}, "param_groups": [{"lr": 1e-5, "params": []}]},
        "std": torch.full((6,), 0.05),
        "rng": torch.get_rng_state(),
        "counters": {"updates": update, "samples": update * 4},
    }


# ---------------------------------------------------------------- 分派
@pytest.mark.skipif(not ACTIVE_CONFIG.is_file(), reason="活动配置不在本机")
def test_active_config_is_accepted_by_single_dispatch():
    job = job_config.load_job_config(ACTIVE_CONFIG)
    service = dispatch.execution_service(job)
    assert service["service"] == "production_motion"
    from native_stand.simulation.isaac import engine

    ok, reason = engine.available(job)
    assert ok, reason


def test_execution_service_is_pure_and_distinguishes_paths(tmp_path):
    runtime_job = _cpu_job(
        tmp_path, backend="isaac", assets={"runtime_template": str(tmp_path / "t.json")}
    )
    legacy_job = _cpu_job(
        tmp_path, backend="isaac", assets={"legacy_spec_path": str(tmp_path / "spec.json")}
    )
    assert dispatch.execution_service(runtime_job)["service"] == "production_motion"
    assert dispatch.execution_service(legacy_job)["service"] == "generic"
    assert dispatch.execution_service(_cpu_job(tmp_path))["service"] == "generic"
    # 判定只看配置，不创建环境、不查GPU。
    assert not (tmp_path / "t.json").exists()


# ---------------------------------------------------------------- 完整初始化
def test_full_initialization_payload_is_loaded_from_config(tmp_path):
    job = _cpu_job(tmp_path)
    checkpoint = save_full(tmp_path / "source.pt", _full_state(sentinel=0.375))
    job = job_config.with_initialization(job, checkpoint)
    from native_stand.simulation.engine import _initialization_payload
    from native_stand.artifacts.io import file_sha256

    payload = _initialization_payload(job)
    assert payload is not None, "配置声明full就必须真的读来源"
    assert torch.allclose(payload.payload["actor"]["w"], torch.full((4, 31), 0.375))
    assert payload.payload["counters"]["updates"] == 5
    assert payload.path == checkpoint.resolve()
    assert payload.sha256 == file_sha256(checkpoint)


def _sentinel_full_state(job, sentinel: float = 0.375) -> dict:
    """真实结构的状态字典：所有参数填同一个哨兵值，便于证明来源被加载。"""
    from native_stand.learning.ppo import assembly
    from native_stand.simulation.cpu import CpuEnvFixture

    env = CpuEnvFixture(job)
    ppo = assembly.assemble_ppo(
        assembly.build_actor(job), assembly.build_critic(job), env.observations(), job, device="cpu"
    )
    with torch.no_grad():
        for parameter in list(ppo.actor.parameters()) + list(ppo.critic.parameters()):
            parameter.fill_(sentinel)
    return assembly.capture_full(ppo, counters={"updates": 5, "samples": 20})


def test_full_initialization_reaches_the_optimizer(tmp_path, monkeypatch):
    """穿过 engine.run_stage：restore_full 收到配置声明的来源，而不是空。"""
    from native_stand.simulation import engine as engine_module

    job = _cpu_job(tmp_path, limits={"smoke_updates": 1, "formal_updates": 0})
    checkpoint = save_full(tmp_path / "source.pt", _sentinel_full_state(job))
    job = job_config.with_initialization(job, checkpoint)
    run_dir = create_run(job, runs_root=tmp_path / "runs", run_id="init")
    from native_stand.experiments.budget import Budget

    Budget.create(run_dir / "budget.json", smoke_updates=1, formal_updates=0)
    RunStateStore(run_dir).create(run_id="init", owner={"pid": 0}, stage="smoke")
    from native_stand.experiments import worker

    context = worker._new_context(run_dir, "smoke", suffix="ab" * 8)
    seen = {}
    original = engine_module.assembly.restore_full

    def spy(ppo, payload):
        seen["actor"] = {k: v.clone() for k, v in payload["actor"].items()}
        return original(ppo, payload)

    monkeypatch.setattr(engine_module.assembly, "restore_full", spy)
    backend = dispatch.backend_for(job)
    engine_module.run_stage(context, backend=backend, stage="smoke")
    assert "actor" in seen, "完整初始化没有传到恢复函数"
    for name, value in seen["actor"].items():
        assert torch.allclose(value, torch.full_like(value, 0.375)), f"{name} 不是来源权重"


def test_missing_initialization_source_is_a_clear_error(tmp_path):
    """入口就拒绝不存在的来源；不会留下"声明了full但没加载"的运行。"""
    job = _cpu_job(tmp_path)
    with pytest.raises(FileNotFoundError):
        job_config.with_initialization(job, tmp_path / "source.pt")


# ---------------------------------------------------------------- 恢复不依赖原配置
def test_cli_resume_works_after_original_config_deleted(tmp_path):
    job = _cpu_job(tmp_path, limits={"smoke_updates": 1, "formal_updates": 2})
    run_dir = create_run(job, runs_root=tmp_path / "runs", run_id="resume_cli")
    from native_stand.experiments.budget import Budget

    Budget.create(run_dir / "budget.json", smoke_updates=1, formal_updates=2)
    store = RunStateStore(run_dir)
    store.create(run_id="resume_cli", owner={"pid": 0}, stage="smoke")
    checkpoint = save_full(run_dir / "checkpoints" / "model_update_1.pt", _full_state(update=1))
    write_inventory(
        run_dir,
        [
            {
                "checkpoint": str(checkpoint),
                "update": 1,
                "sha256": model_refs.file_sha256(checkpoint),
                "bytes": 1,
            }
        ],
    )
    job.config_path.unlink()

    from native_stand.cli import main

    code = main(["resume", "--run", str(run_dir), "--dry-run"])
    assert code == 0
    assert (
        job_config.job_from_record(json.loads((run_dir / "run.json").read_text())["config"]).limits
        == job.limits
    )


# ---------------------------------------------------------------- 探索单位
def test_exploration_std_uses_output_envelope(tmp_path):
    from native_stand.learning.ppo.networks import exploration_std, initial_std

    job = _cpu_job(
        tmp_path,
        action={"leg_scale_rad": 0.15, "wheel_scale_rad_s": 40.0},
        ppo={"exploration_leg_rad": 0.3, "exploration_wheel_rad_s": 8.0},
    )
    std = initial_std(job)
    for name, value in zip(job.action.order, std.tolist(), strict=True):
        scale = job.action.leg_scale_rad if "wheel" not in name else job.action.wheel_scale_rad_s
        target = (
            job.ppo.exploration_leg_rad if "wheel" not in name else job.ppo.exploration_wheel_rad_s
        )
        assert value * scale == pytest.approx(target), f"{name}物理尺度不符"
    raw = exploration_std(
        ["left_hip", "left_wheel"],
        leg_rad=0.3,
        wheel_rad_s=1.0,
        leg_output_rad=0.15,
        wheel_output_rad_s=40.0,
    )
    assert float(raw[0]) == pytest.approx(2.0)
    assert float(raw[1]) == pytest.approx(0.025)


# ---------------------------------------------------------------- inventory限定
@pytest.mark.parametrize("unregistered", [10])
def test_prune_ignores_files_not_in_inventory(tmp_path, unregistered):
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    paths = {}
    for update in (10, 20, 30, 40):
        path = checkpoints / f"model_update_{update}.pt"
        save_full(path, _full_state(update=update))
        paths[update] = path
    # 只登记20/30/40；10未登记
    write_inventory(
        run,
        [
            {
                "checkpoint": str(paths[u]),
                "update": u,
                "sha256": model_refs.file_sha256(paths[u]),
                "bytes": 1,
            }
            for u in (20, 30, 40)
        ],
    )
    report = prune(run, SavePolicy(every_updates=10, keep_recent=2))
    assert paths[unregistered].is_file(), "未登记文件不能被删除"
    assert str(paths[unregistered]) not in report["removed"]


def test_prune_skips_replaced_and_external_and_symlink(tmp_path):
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    inside = checkpoints / "model_update_10.pt"
    save_full(inside, _full_state(update=10))
    outside = tmp_path / "external.pt"
    save_full(outside, _full_state(update=99))
    link = checkpoints / "model_update_11.pt"
    link.symlink_to(outside)
    replaced = checkpoints / "model_update_12.pt"
    save_full(replaced, _full_state(update=12))
    write_inventory(
        run,
        [
            {"checkpoint": str(inside), "update": 10, "sha256": "0" * 64, "bytes": 1},
            {
                "checkpoint": str(link),
                "update": 11,
                "sha256": model_refs.file_sha256(outside),
                "bytes": 1,
            },
            {
                "checkpoint": str(outside),
                "update": 99,
                "sha256": model_refs.file_sha256(outside),
                "bytes": 1,
            },
            {
                "checkpoint": str(replaced),
                "update": 12,
                "sha256": model_refs.file_sha256(replaced),
                "bytes": 1,
            },
        ],
    )
    prune(run, SavePolicy(every_updates=1, keep_recent=1))
    assert inside.is_file(), "身份变化的同名文件不能删"
    assert outside.is_file(), "外部路径不能删"
    assert link.is_symlink()
    assert replaced.is_file()


def test_plan_and_execution_match_after_inventory_scope(tmp_path):
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    rows = []
    for update in (2, 4, 6):
        path = checkpoints / f"model_update_{update}.pt"
        save_full(path, _full_state(update=update))
        rows.append(
            {
                "checkpoint": str(path),
                "update": update,
                "sha256": model_refs.file_sha256(path),
                "bytes": 1,
            }
        )
    write_inventory(run, rows)
    policy = SavePolicy(every_updates=2, keep_recent=1)
    plan = plan_prune(run, policy)
    executed = prune(run, policy)
    assert plan["removed"] == executed["removed"]


# ---------------------------------------------------------------- 执行接线
class _FakeProduction:
    """Isaac生产适配器的替身：只验证接线，不验证物理。"""

    def __init__(self):
        self.calls = []

    def validate_job(self, job):
        self.calls.append(("validate", job.backend))

    def run(self, context, *, stage=None, resume_from=None):
        self.calls.append(("run", stage, resume_from))
        return "production-result"


def test_worker_executes_production_service_for_runtime_template(tmp_path, monkeypatch):
    import importlib
    import types

    from native_stand.experiments import worker

    template = tmp_path / "template.json"
    template.write_text("{}")
    job = _cpu_job(tmp_path, backend="isaac", assets={"runtime_template": str(template)})
    fake = _FakeProduction()
    module = types.SimpleNamespace(validate_job=fake.validate_job, run=fake.run)

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == dispatch.PRODUCTION_MOTION:
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(worker.importlib, "import_module", fake_import)
    context = types.SimpleNamespace(job=job)
    result = worker._execute_stage(context, "formal", {"sentinel": 1})
    assert result == "production-result"
    assert ("run", "formal", {"sentinel": 1}) in fake.calls
    assert ("validate", "isaac") in fake.calls


def test_worker_uses_generic_engine_for_cpu_jobs(tmp_path, monkeypatch):
    from native_stand.experiments import worker

    job = _cpu_job(tmp_path)
    monkeypatch.setattr(dispatch, "execution_service", dispatch.execution_service)
    captured = {}

    def fake_run_stage(context, *, backend, stage, resume_from=None):
        captured.update(stage=stage, backend=backend.name, payload=resume_from)
        return "generic-result"

    monkeypatch.setattr(worker, "run_stage", fake_run_stage)
    context = types.SimpleNamespace(job=job)
    assert worker._execute_stage(context, "smoke", None) == "generic-result"
    assert captured["backend"] == "cpu"


def test_unsupported_service_is_refused_before_execution(tmp_path):
    from native_stand.experiments import worker

    job = _cpu_job(tmp_path, backend="isaac", assets={})
    context = types.SimpleNamespace(job=job)
    with pytest.raises(dispatch.DispatchError, match="runtime_template"):
        worker._execute_stage(context, "smoke", None)


# ---------------------------------------------------------------- 环境结束语义
class _FakeEnv:
    """声明式假环境：返回真实的 terminated/truncated/reset。"""

    def __init__(self, *, extras=True):
        self.extras = extras
        self.num_envs = 2
        self.device = "cpu"

    def step(self, action, **kwargs):
        import torch

        terminated = torch.tensor([True, False])
        truncated = torch.tensor([False, True])
        reward = torch.tensor([1.0, -1.0])
        next_obs = {"student_actor": torch.zeros(2, 31), "student_critic": torch.zeros(2, 64)}
        if self.extras:
            return next_obs, reward, terminated, truncated, {"time_outs": truncated}
        return next_obs, reward, terminated | truncated


def test_isaac_step_preserves_termination_and_reset():
    from native_stand.simulation.isaac import engine as isaac_engine

    class Probe(isaac_engine.IsaacMotionEnv):
        def __init__(self, env):  # 跳过Isaac启动
            self.env = env
            self.num_envs = 2
            self.device = "cpu"
            self.job = types.SimpleNamespace()
            self._obs = None
            import inspect

            self._return_extras = "return_extras" in inspect.signature(env.step).parameters

    import types

    outcome = Probe(_FakeEnv(extras=True)).step({"student_actor": None})
    assert outcome["terminated"].tolist() == [[True], [False]]
    assert outcome["truncated"].tolist() == [[False], [True]]
    assert outcome["dones"].tolist() == [[True], [True]]
    assert outcome["reset_envs"] == 2
    assert outcome["time_outs"].tolist() == [0.0, 1.0]


def test_isaac_step_handles_dones_only_environments():
    import inspect
    import types

    from native_stand.simulation.isaac import engine as isaac_engine

    class Probe(isaac_engine.IsaacMotionEnv):
        def __init__(self, env):
            self.env = env
            self.num_envs = 2
            self.device = "cpu"
            self.job = types.SimpleNamespace()
            self._obs = None
            self._return_extras = "return_extras" in inspect.signature(env.step).parameters

    outcome = Probe(_FakeEnv(extras=False)).step(None)
    assert outcome["dones"].tolist() == [[True], [True]]
    assert outcome["truncated"].tolist() == [[True], [True]]
    assert outcome["extras"] == {}


# ---------------------------------------------------------------- 独立评估
class _CountingEnv:
    constructions = 0
    closes = 0

    def __init__(self, job, condition_log):
        type(self).constructions += 1
        self.num_envs = job.runtime.envs
        self.device = "cpu"
        self.command = None
        self._log = condition_log
        self.step_index = 0

    def set_command(self, command):
        self.command = list(command)
        self._log.append(("command", tuple(command)))

    def observations(self):
        import torch

        return {
            "student_actor": torch.zeros(self.num_envs, 31),
            "student_critic": torch.zeros(self.num_envs, 64),
        }

    def step(self, action):
        import torch

        self.step_index += 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.command and self.command[0] > 0:
            terminated[0] = True
        return {
            "rewards": torch.ones(self.num_envs),
            "terminated": terminated.unsqueeze(-1),
            "dones": terminated.unsqueeze(-1),
        }

    def close(self):
        type(self).closes += 1


def test_evaluation_isolates_conditions_and_uses_declared_envs(tmp_path, monkeypatch):
    from native_stand.learning.ppo import evaluate as evaluate_module
    from native_stand.simulation.driver import Backend

    log = []
    _CountingEnv.constructions = 0
    _CountingEnv.closes = 0
    job = _cpu_job(
        tmp_path,
        runtime={"envs": 4},
        evaluation={
            "envs": 2,
            "scored_envs": 1,
            "seconds": 0.03,
            "seed": 77,
            "conditions": [
                {"name": "forward", "command": [1.0, 0.0, 0.0]},
                {"name": "stand", "command": [0.0, 0.0, 0.0]},
            ],
        },
    )
    checkpoint = save_full(tmp_path / "model.pt", _sentinel_full_state(job))
    monkeypatch.setattr(
        evaluate_module.dispatch,
        "backend_for_evaluation",
        lambda _job: Backend(
            name="fake", make_env=lambda j, resume_from=None: _CountingEnv(j, log)
        ),
    )
    result = evaluate_module.evaluate_checkpoint(
        job, checkpoint, kind="final", output_dir=tmp_path / "evaluations"
    )
    assert _CountingEnv.constructions == 2, "每个条件必须新建环境"
    assert _CountingEnv.closes == 2
    assert [row["condition"] for row in result["conditions"]] == ["forward", "stand"]
    assert all(len(row["per_env"]) == 2 for row in result["conditions"]), "必须用evaluation.envs"
    assert [entry for entry in log if entry[0] == "command"] == [
        ("command", (1.0, 0.0, 0.0)),
        ("command", (0.0, 0.0, 0.0)),
    ]
    # 第一个条件的提前终止不被第二个条件继承
    assert result["conditions"][0]["failures"] == 1
    assert result["conditions"][1]["failures"] == 0
    assert result["protocol"]["seed"] == 77
    assert result["protocol"]["capability_judged"] is False
    assert result["capability_accepted"] is False


def test_two_evaluations_of_same_kind_are_both_kept(tmp_path, monkeypatch):
    import json as _json

    from native_stand.artifacts import evaluation as evaluation_format
    from native_stand.learning.ppo import evaluate as evaluate_module
    from native_stand.simulation.driver import Backend

    job = _cpu_job(
        tmp_path,
        evaluation={
            "envs": 1,
            "scored_envs": 1,
            "seconds": 0.01,
            "conditions": [{"name": "nominal"}],
        },
    )
    checkpoint = save_full(tmp_path / "model.pt", _sentinel_full_state(job))
    monkeypatch.setattr(
        evaluate_module.dispatch,
        "backend_for_evaluation",
        lambda _job: Backend(name="fake", make_env=lambda j, resume_from=None: _CountingEnv(j, [])),
    )
    first = evaluate_module.evaluate_checkpoint(
        job, checkpoint, kind="best", output_dir=tmp_path / "evaluations"
    )
    second = evaluate_module.evaluate_checkpoint(
        job, checkpoint, kind="best", output_dir=tmp_path / "evaluations"
    )
    evaluation_format.register(
        tmp_path / "evaluations", kind="best", path=first["path"], result=first
    )
    evaluation_format.register(
        tmp_path / "evaluations", kind="best", path=second["path"], result=second
    )
    assert Path(first["path"]) != Path(second["path"])
    assert Path(first["path"]).is_file() and Path(second["path"]).is_file()
    index = _json.loads((tmp_path / "evaluations" / "index.json").read_text())
    ids = {row["evaluation_id"] for row in index["evaluations"]}
    assert {first["evaluation_id"], second["evaluation_id"]} <= ids
    assert index["latest"]["best"] == second["evaluation_id"]


# ---------------------------------------------------------------- inventory 一致性
def test_skipped_records_stay_in_inventory(tmp_path):
    from native_stand.artifacts.retention import read_inventory

    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    good = checkpoints / "model_update_10.pt"
    save_full(good, _full_state(update=10))
    missing = checkpoints / "model_update_20.pt"
    save_full(missing, _full_state(update=20))
    missing.unlink()
    write_inventory(
        run,
        [
            {"checkpoint": str(good), "update": 10, "sha256": "0" * 64, "bytes": 1},
            {"checkpoint": str(missing), "update": 20, "sha256": "1" * 64, "bytes": 1},
        ],
    )
    report = prune(run, SavePolicy(every_updates=10, keep_recent=5))
    rows = {row["checkpoint"]: row for row in read_inventory(run)}
    assert str(missing) in rows, "缺失文件的记录不能被删除"
    assert str(good) in rows
    assert report["skipped"], "跳过原因必须记录"


# ---------------------------------------------------------------- 生产适配器（真实方法，替身环境）
def _sentinel_full_state(job, sentinel: float = 0.375) -> dict:
    from native_stand.learning.ppo import assembly
    from native_stand.simulation.cpu import CpuEnvFixture

    env = CpuEnvFixture(job)
    ppo = assembly.assemble_ppo(
        assembly.build_actor(job), assembly.build_critic(job), env.observations(), job, device="cpu"
    )
    with torch.no_grad():
        for parameter in list(ppo.actor.parameters()) + list(ppo.critic.parameters()):
            parameter.fill_(sentinel)
    return assembly.capture_full(ppo, counters={"updates": 1, "samples": 4})


def _production_context(tmp_path, job, *, smoke=2, formal=0):
    from native_stand.experiments import worker
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.owner import create_run

    run_dir = create_run(job, runs_root=tmp_path / "runs", run_id="prod")
    Budget.create(run_dir / "budget.json", smoke_updates=smoke, formal_updates=formal)
    RunStateStore(run_dir).create(run_id="prod", owner={"pid": 0}, stage="smoke")
    return worker._new_context(run_dir, "smoke", suffix="cd" * 8)


class _FakeStream:
    def __init__(self):
        self.calls = 0

    def export(self, update):
        self.calls += 1
        self.last_update = update
        return {"update": update}


class _FakeDR:
    def __init__(self):
        self.episodes = torch.zeros(2, dtype=torch.long)


class _FakeV4Env:
    """只提供 after_update/capture/close 需要的冻结环境表面。"""

    def __init__(self, *, revision=False):
        self.device = "cpu"
        self.onpolicy_episodes = _FakeStream()
        self.identified_dr = _FakeDR()
        self.closed = 0
        self.stage_spec = {"commands": [[0.0, 0.0]]}
        self.transient_events = types.SimpleNamespace(summary=lambda: {})
        self.motion_timing = types.SimpleNamespace(
            tick=11, dr=types.SimpleNamespace(caps=torch.ones(6, 1))
        )
        self.supported_requested_target_post = torch.zeros(2, 6)
        self.supported_adopted_target_post = torch.zeros(2, 6)
        self.supported_saturation_post = {
            "driver_at_cap": torch.zeros(2, 6),
            "motor_at_cap": torch.zeros(2, 6),
        }

    def close(self):
        self.closed += 1


def _course_adapter(tmp_path, *, revision=False):
    from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter

    adapter = MotionProductionAdapter()
    adapter.context = types.SimpleNamespace(run_dir=tmp_path)
    adapter.job = types.SimpleNamespace(
        assets={"historical_online_updates": 454},
        runtime=types.SimpleNamespace(envs=2),
        ppo=types.SimpleNamespace(critic_warmup_updates=0),
        diagnostics={},
        curriculum=types.SimpleNamespace(version="motion31_supported2048_dqgain_ppo_v1"),
        action=types.SimpleNamespace(version="motion31_v1", dim=6),
    )
    adapter.env = _FakeV4Env()
    adapter.state = {"stage": 0}
    adapter.source = {
        "update": 100,
        "recipe_updates": 544,
        "reward_refinement_state": [],
        "symmetry_mix_state": [],
        "interface_version": "multiskill_mcu_v2",
        "role": "MotionZeroStudent",
        "dataset_sha256": "d" * 64,
        "manifest_sha256": "m" * 64,
        "asset_sha256": "s" * 64,
        "network_architecture": {},
        "critic_privileged_version": "v1",
        "output_contract": {},
        "action_version": "motion31_v1",
    }
    adapter.job.as_run_record = lambda: {"ppo": {}}
    adapter.source_path = tmp_path / "resume.pt"
    adapter.source_sha = "a" * 64
    adapter.course_update_base = 100
    adapter.restore_point_updates = 100
    adapter.config = {}
    adapter.spec = {"student_training": {}, "curriculum": {"stages": [adapter.env.stage_spec]}}
    adapter.fixed_std = torch.zeros(6)
    adapter.algorithm = types.SimpleNamespace(
        actor=types.SimpleNamespace(state_dict=lambda: {}, log_std=torch.zeros(6)),
        critic=types.SimpleNamespace(state_dict=lambda: {}),
        optimizer=types.SimpleNamespace(state_dict=lambda: {}),
    )
    adapter.modules = {
        "rng_state": __import__("native_stand.students.resume", fromlist=["rng_state"]).rng_state,
        "restore_rng": __import__(
            "native_stand.students.resume", fromlist=["restore_rng"]
        ).restore_rng,
        "build_environment_spec": lambda *a, **k: {
            "student_training": {}, "curriculum": {"stages": [adapter.env.stage_spec]}
        },
        "course": types.SimpleNamespace(stage_config=lambda *a, **k: {}),
    }
    from native_stand.tasks.motion.course_strategy import CourseStrategy

    adapter.course_strategy = CourseStrategy(adapter.modules["course"], (), legacy=True)
    adapter._update_scope = {
        "stats": types.SimpleNamespace(export=lambda: {}),
        "diagnostics": {},
        "detail": {"actor_optimizer_steps": 1},
        "channel_clips": torch.zeros(6),
        "valid_samples": torch.tensor(1.0),
        "nonzero": 2,
    }
    return adapter


def test_after_update_advances_course_exactly_at_100_and_200(tmp_path, monkeypatch):
    import sys as _sys

    calls = []

    def fake_advance(state, review, update):
        calls.append(update)
        return state, {"state_revision_changed": False}

    monkeypatch.setitem(_sys.modules, "onpolicy", types.SimpleNamespace(advance=fake_advance))
    adapter = _course_adapter(tmp_path)
    adapter.restore_point_updates = 0  # 首次初始化：来源不含本run进度
    context = types.SimpleNamespace(run_dir=tmp_path, updates_used=0)

    for completed in (99, 100, 101, 199, 200, 201):
        context.updates_used = completed
        adapter.after_update(context, completed, {})

    # advance 用全局模型进度（来源update 100 + run内完成数），文件与记录用run内完成数。
    assert calls == [200, 300], f"课程推进次数不正确: {calls}"
    assert adapter.env.onpolicy_episodes.calls == 2
    assert (tmp_path / "diagnostics" / "update_00100.json").is_file()
    assert (tmp_path / "diagnostics" / "update_00200.json").is_file()
    payload = json.loads((tmp_path / "diagnostics" / "update_00100.json").read_text())
    assert payload["update"] == 100


def test_course_switch_closes_old_environment_once(tmp_path, monkeypatch):
    import sys as _sys

    monkeypatch.setitem(
        _sys.modules,
        "onpolicy",
        types.SimpleNamespace(
            advance=lambda state, review, update: (state, {"state_revision_changed": True})
        ),
    )
    adapter = _course_adapter(tmp_path)
    old_env = adapter.env
    new_env = _FakeV4Env()
    monkeypatch.setattr(adapter, "_make_env", lambda: (new_env, {"student_actor": None}))
    context = types.SimpleNamespace(run_dir=tmp_path, updates_used=100)
    adapter.after_update(context, 100, {})
    assert old_env.closed == 0
    adapter.after_save(context, {"update": 100})

    assert old_env.closed == 1, "旧环境必须只关闭一次"
    assert adapter.env is new_env, "adapter 必须持有新环境"
    assert new_env.closed == 0, "切换时不能关闭新环境"
    adapter.close()
    assert new_env.closed == 1
    assert adapter.app is None


def test_capture_does_not_double_count_recipe_updates(tmp_path):
    adapter = _course_adapter(tmp_path)
    context = types.SimpleNamespace(updates_used=101)
    payload = adapter.capture(context, {"updates": 101})
    assert payload["recipe_updates"] == 545, "来源已含的100次更新不能再加一次"
    assert payload["online_updates_including_smoke"] == 454 + 101
    assert payload["counters"]["resume_point_new_updates"] == 100
    assert payload["source_sha256"] == "a" * 64, "来源身份必须是实际恢复点"


def test_production_close_is_idempotent(tmp_path):
    adapter = _course_adapter(tmp_path)
    adapter.app = types.SimpleNamespace(closed=0)
    adapter.app.close = lambda: setattr(adapter.app, "closed", adapter.app.closed + 1)
    env = adapter.env
    adapter.close()
    adapter.close()
    assert env.closed == 1
    assert adapter._closed is True


def test_production_resume_source_priority(tmp_path):
    from native_stand.simulation.isaac import motion_backend

    job = _cpu_job(tmp_path)
    config_source = save_full(tmp_path / "config_init.pt", _full_state(update=1))
    job = job_config.with_initialization(job, config_source)
    from native_stand.artifacts.checkpoint import load_source

    resume_file = save_full(tmp_path / "resume.pt", _full_state(update=9))
    source = load_source(resume_file)
    resolved = motion_backend._resolve_source(job, source)
    assert resolved is source and resolved.path == resume_file.resolve()
    assert len(resolved.sha256) == 64
    loaded = motion_backend._resolve_source(job, None)
    assert loaded.payload["counters"]["updates"] == 1
    assert loaded.path.is_file() and len(loaded.sha256) == 64


def test_production_missing_initialization_is_rejected(tmp_path):
    from native_stand.simulation.isaac import motion_backend

    job = _cpu_job(tmp_path)
    job = job_config.with_initialization(job, save_full(tmp_path / "s.pt", _full_state()))
    Path(job.initialization.checkpoint).unlink()
    with pytest.raises(FileNotFoundError, match="恢复来源不存在"):
        motion_backend._resolve_source(job, None)


def test_source_progress_accepts_continuation_and_rejects_inconsistent(tmp_path):
    from native_stand.simulation.isaac.motion_backend import _check_source_progress

    job = types.SimpleNamespace(
        assets={"historical_online_updates": 454, "original_online_limit": 10000}
    )
    fresh = {"online_updates_including_smoke": 454}
    assert _check_source_progress(job, fresh, None) == 0
    continuation = {"online_updates_including_smoke": 555, "counters": {"new_run_updates": 101}}
    assert _check_source_progress(job, continuation, continuation) == 101
    with pytest.raises(ValueError, match="不一致"):
        _check_source_progress(job, continuation, {"counters": {"new_run_updates": 7}})
    with pytest.raises(ValueError, match="首次初始化"):
        _check_source_progress(job, continuation, None)


def test_driver_calls_after_update_after_charge(tmp_path):
    """驱动器顺序：update → 计费 → after_update → 保存。"""
    from native_stand.simulation import driver

    job = _cpu_job(tmp_path, limits={"smoke_updates": 2, "formal_updates": 0})
    context = _production_context(tmp_path, job)
    events = []

    class Adapter:
        def build(self, context, stage, resume_from):
            events.append(("build", context.updates_used))

        def update_once(self, context, index):
            events.append(("update", context.updates_used))
            return {"reward_mean": 0.0, "actor_steps": 1}

        def after_update(self, context, index, metrics):
            events.append(("after", context.updates_used))

        def capture(self, context, counters):
            events.append(("capture", context.updates_used))
            return _sentinel_full_state(job, sentinel=0.1)

        def close(self):
            events.append(("close", None))

    driver.run_stage(context, Adapter(), stage="smoke")
    assert events[0] == ("build", 0)
    assert events[1] == ("update", 0)
    assert events[2] == ("after", 1), "after_update 必须在计费之后"
    assert events[-1] == ("close", None)


def test_production_incomplete_rollout_is_not_charged(tmp_path):
    from native_stand.simulation import driver

    job = _cpu_job(tmp_path, limits={"smoke_updates": 5, "formal_updates": 0})
    context = _production_context(tmp_path, job, smoke=5)
    calls = {"n": 0}

    class Adapter:
        def build(self, context, stage, resume_from):
            return None

        def update_once(self, context, index):
            calls["n"] += 1
            return {"incomplete": True} if index == 2 else {"actor_steps": 1}

        def capture(self, context, counters):
            return _sentinel_full_state(job, sentinel=0.1)

        def close(self):
            return None

    result = driver.run_stage(context, Adapter(), stage="smoke")
    assert calls["n"] == 2
    assert context.updates_used == 1, "未完成的rollout不能计费"
    assert result.limit == "rollout_incomplete"


# ---------------------------------------------------------------- 恢复身份与课程时间轴
def test_resume_source_identity_comes_from_loader(tmp_path):
    """worker的加载入口给出B的身份；生产加载器不再回退到初始A。"""
    from native_stand.artifacts.checkpoint import load_source
    from native_stand.experiments.worker import _load_resume_source
    from native_stand.simulation.isaac import motion_backend

    job = _cpu_job(tmp_path)
    initial = save_full(tmp_path / "A.pt", _full_state(update=442))
    job = job_config.with_initialization(job, initial)
    checkpoint_b = save_full(tmp_path / "B.pt", _full_state(update=542))

    loaded = _load_resume_source(checkpoint_b)
    assert callable(load_source)
    assert loaded.path == checkpoint_b.resolve()
    resolved = motion_backend._resolve_source(job, loaded)
    assert resolved.path == checkpoint_b.resolve(), "不能回退到初始A"
    assert resolved.sha256 == loaded.sha256
    with pytest.raises(TypeError, match="LoadedSource"):
        motion_backend._resolve_source(job, {"actor": {}})


def test_course_timeline_same_for_continuous_and_resumed(tmp_path, monkeypatch):
    """连续到第200次 与 第100次保存→恢复→第200次 传给课程的编号一致。"""
    import sys as _sys

    seen = []

    def fake_advance(state, review, update):
        seen.append(update)
        return state, {"state_revision_changed": False}

    monkeypatch.setitem(_sys.modules, "onpolicy", types.SimpleNamespace(advance=fake_advance))

    # 连续：来源是初始模型(update=442)，恢复点不含本run进度
    continuous = _course_adapter(tmp_path)
    continuous.source["update"] = 442
    continuous.course_update_base = 442
    continuous.restore_point_updates = 0
    continuous.context.updates_used = 200
    continuous.after_update(continuous.context, 200, {})

    # 分段：来源是第100次的checkpoint(update=542)，恢复点含100次本run进度
    resumed = _course_adapter(tmp_path)
    resumed.source["update"] = 542
    resumed.course_update_base = 542
    resumed.restore_point_updates = 100
    resumed.context.updates_used = 200
    resumed.after_update(resumed.context, 200, {})

    assert seen == [642, 642], f"课程编号不一致: {seen}"


# ---------------------------------------------------------------- attempt 结算
def test_normal_run_settles_wall_time(tmp_path):
    job = _cpu_job(
        tmp_path,
        limits={"smoke_updates": 1, "formal_updates": 1},
        runtime={"envs": 2, "steps_per_env": 1},
        storage={"checkpoint_every_updates": 5},
    )
    handshake = submit(job, runs_root=tmp_path / "runs", run_id="settle", wait_s=180)
    run = Path(handshake["run"])
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = json.loads((run / "state.json").read_text())
        if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            break
        time.sleep(0.2)
    budget = json.loads((run / "budget.json").read_text())
    assert budget["attempts"], budget
    for row in budget["attempts"].values():
        assert row["status"] in ("COMPLETED", "STOPPED_BOUNDED"), row
        assert row.get("wall_s") is not None and row["wall_s"] > 0, row


def test_missing_resume_file_leaves_no_running_attempt(tmp_path):
    """恢复文件缺失：worker非零退出、state FAILED、attempt FAILED且无残留RUNNING。"""
    from native_stand.experiments import worker
    from native_stand.experiments.budget import Budget

    job = _cpu_job(
        tmp_path,
        limits={"smoke_updates": 1, "formal_updates": 0},
        runtime={"envs": 2, "steps_per_env": 1},
    )
    run = create_run(job, runs_root=tmp_path / "runs", run_id="settle_fail")
    Budget.create(run / "budget.json", smoke_updates=1, formal_updates=0)
    RunStateStore(run).create(run_id="settle_fail", owner={"pid": 0}, stage="smoke")

    ghost = tmp_path / "ghost.pt"
    code = worker.main(["--run-dir", str(run), "--resume-checkpoint", str(ghost)])
    assert code != 0
    state = json.loads((run / "state.json").read_text())
    assert state["status"] == "FAILED"
    budget = json.loads((run / "budget.json").read_text())
    assert all(
        row["status"] != "RUNNING" for row in budget["attempts"].values()
    ), "启动失败不能留下RUNNING attempt"


# ---------------------------------------------------------------- 评估窗口与全部环境
def test_first_step_failure_counts_one_interval():
    from native_stand.learning.ppo import evaluate as evaluate_module

    class Env:
        num_envs = 1

        def __init__(self):
            self.step_index = 0

        def observations(self):
            return {"student_actor": torch.zeros(1, 31)}

        def step(self, action):
            self.step_index += 1
            terminated = torch.tensor([[self.step_index == 1]])
            return {
                "rewards": torch.tensor([[1.0]]),
                "terminated": terminated,
                "truncated": torch.zeros(1, 1, dtype=torch.bool),
            }

    class Actor:
        def __call__(self, obs, stochastic_output=False):
            return torch.zeros(1, 6)

    outcome = evaluate_module._rollout(Env(), Actor(), seconds=0.03, policy_dt=0.01)
    failure = outcome["first_failure"][0]
    assert failure["seconds"] == pytest.approx(0.01), "首步失败要记一个policy interval"
    assert outcome["returns"][0].item() == pytest.approx(1.0), "失败后奖励不再累计"


def test_full_window_timeout_is_not_early_failure():
    from native_stand.learning.ppo import evaluate as evaluate_module

    class Env:
        num_envs = 1

        def observations(self):
            return {"student_actor": torch.zeros(1, 31)}

        def __init__(self):
            self.step_index = 0

        def step(self, action):
            self.step_index += 1
            last = self.step_index == 3
            return {
                "rewards": torch.ones(1, 1),
                "terminated": torch.zeros(1, 1, dtype=torch.bool),
                "truncated": torch.full((1, 1), last, dtype=torch.bool),
            }

    class Actor:
        def __call__(self, obs, stochastic_output=False):
            return torch.zeros(1, 6)

    outcome = evaluate_module._rollout(Env(), Actor(), seconds=0.03, policy_dt=0.01)
    assert outcome["first_failure"] == {}, "走到窗口终点的截断不是提前失败"
    assert outcome["returns"][0].item() == pytest.approx(3.0)


def test_command_check_covers_all_scored_envs():
    from native_stand.simulation.isaac.motion_backend import ProductionEvalEnv

    class Raw:
        def __init__(self):
            self.motion_action_command = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        def set_evaluation_command(self, command):
            pass  # 只记住期望，不写入：模拟pre-step覆盖

        def step(self, action):
            return (
                {},
                torch.ones(2, 1),
                torch.zeros(2, 1, dtype=torch.bool),
                torch.zeros(2, 1, dtype=torch.bool),
                {},
            )

        def close(self):
            return None

    job = _cpu_job(Path(__import__("tempfile").mkdtemp()), evaluation={"envs": 2, "scored_envs": 2})
    wrapper = ProductionEvalEnv(Raw(), None, job, lambda e, c: None)
    wrapper.set_command([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="偏差环境"):
        wrapper.step(torch.zeros(2, 6))


def test_reward_mean_is_per_update_sum_over_samples(tmp_path):
    """reward_mean = 本轮奖励总和 / 实际样本数；多环境、逐轮不同、累计器每轮重置。"""
    from native_stand.simulation import engine as engine_module
    from native_stand.simulation.driver import Backend
    from native_stand.simulation.cpu import CpuEnvFixture

    job = _cpu_job(tmp_path, runtime={"envs": 4, "steps_per_env": 2})
    env = CpuEnvFixture(job)

    captured = {}
    rewards_by_update = {1: 1.0, 2: 3.0}

    class ProbePPO:
        gamma = 0.99
        mask_summary = {"actor_optimizer_steps": 1, "critic_optimizer_steps": 1}
        actor_update_enabled = True

        class _Critic:
            @staticmethod
            def __call__(tensor):
                rows = (
                    tensor["student_critic"].shape[0]
                    if isinstance(tensor, dict)
                    else tensor.shape[0]
                )
                return torch.zeros(rows, 1)

        critic = _Critic()

        def act(self, obs):
            return torch.zeros(env.num_envs, 6)

        def process_env_step(self, obs, rewards, dones, extras):
            captured.setdefault("rewards", []).append(rewards.mean().item())

        def compute_returns(self, obs):
            return None

        def update(self):
            return {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}

    adapter = engine_module.GenericPPOAdapter(Backend(name="cpu", make_env=lambda j: env))
    adapter.job = job
    adapter.ppo = ProbePPO()
    adapter.env = env
    adapter._warmup = 0
    adapter._steps_per_env = job.runtime.steps_per_env
    adapter._reward_total = 0.0
    adapter._reward_total_device = None
    adapter._reward_samples_device = None

    # 让环境在两次update里给出不同奖励：第一次全1，第二次全3
    original_step = env.step
    state = {"update": 1}

    def scripted_step(action):
        outcome = original_step(action)
        outcome = dict(outcome)
        outcome["rewards"] = torch.full_like(outcome["rewards"], rewards_by_update[state["update"]])
        return outcome

    env.step = scripted_step
    context = types.SimpleNamespace(updates_used=0)
    first = adapter.update_once(context, 1)
    assert first["reward_mean"] == pytest.approx(1.0), first
    state["update"] = 2
    context.updates_used = 1
    second = adapter.update_once(context, 2)
    # 第二轮全3：本轮均值必须是3，而不是被累计均值污染
    assert second["reward_mean"] == pytest.approx(3.0), second
    # 分母是实际样本数：4环境×2步 = 8样本，总和24 → 均值3
    assert captured["rewards"]  # PPO输入未变
