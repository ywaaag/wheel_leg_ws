"""第二阶段修复的行为测试：保留、引用、恢复、最终保存、PPO装配、生命周期。

这些测试穿过实际调用边界（文件、worker配置、真实网络与优化器），
不使用"actor_changed=True"之类的硬编码证明。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from native_stand.artifacts import model_refs
from native_stand.artifacts.checkpoint import load_full, save_full
from native_stand.artifacts.retention import (
    SavePolicy,
    plan_prune,
    prune,
    read_inventory,
    write_inventory,
)
from native_stand.experiments import config as job_config
from native_stand.experiments.owner import stop, submit
from native_stand.experiments.state import RunStateStore, StateError

FROZEN_TEST_SCOPE = "PORTABLE"  # 自包含CPU工程契约：tmp fixture + 真实代码，无本机runs/资产依赖

PROJECT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT / "tests" / "fixtures" / "cpu_motion_job.json"

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------- 工具
def make_job(tmp_path: Path, **overrides):
    payload = json.loads(FIXTURE.read_text())
    limits = overrides.pop("limits", {})
    runtime = overrides.pop("runtime", {})
    payload["limits"].update(limits)
    payload["runtime"].update(runtime)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key].update(value)
        else:
            payload[key] = value
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload))
    return job_config.load_job_config(path)


def fake_checkpoint(path: Path, update: int) -> Path:
    return save_full(
        path,
        {
            "actor": {"w": torch.full((2, 2), float(update))},
            "critic": {"w": torch.full((2, 2), float(update))},
            "optimizer": {"state": {}, "param_groups": [{"lr": 1e-5, "params": []}]},
            "std": torch.full((6,), 0.1),
            "rng": torch.get_rng_state(),
            "counters": {"updates": update},
        },
    )


def make_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    return run


# ---------------------------------------------------------------- 保留
def test_earlier_best_outside_recent_window_is_kept(tmp_path):
    run = make_run_dir(tmp_path)
    rows = []
    for update in (10, 20, 30, 40):
        path = fake_checkpoint(run / "checkpoints" / f"model_update_{update}.pt", update)
        rows.append(
            {
                "checkpoint": str(path),
                "update": update,
                "sha256": model_refs.file_sha256(path),
                "bytes": 1,
            }
        )
    write_inventory(run, rows)
    best = (run / "checkpoints" / "model_update_10.pt").resolve()
    report = prune(run, SavePolicy(every_updates=10, keep_recent=2), protected=[best])
    kept = {Path(item).name for item in report["kept"]}
    assert {"model_update_10.pt", "model_update_30.pt", "model_update_40.pt"} <= kept
    assert "model_update_20.pt" in {Path(item).name for item in report["removed"]}
    assert best.is_file()


def test_earlier_selected_and_resume_source_survive_rotation(tmp_path):
    run = make_run_dir(tmp_path)
    for update in (5, 15, 25, 35):
        fake_checkpoint(run / "checkpoints" / f"model_update_{update}.pt", update)
    rows = []
    for update in (5, 15, 25, 35):
        path = (run / "checkpoints" / f"model_update_{update}.pt").resolve()
        rows.append(
            {
                "checkpoint": str(path),
                "update": update,
                "sha256": model_refs.file_sha256(path),
                "bytes": 1,
            }
        )
    write_inventory(run, rows)
    selected = (run / "checkpoints" / "model_update_5.pt").resolve()
    continuation = (run / "checkpoints" / "model_update_15.pt").resolve()
    report = prune(
        run, SavePolicy(every_updates=10, keep_recent=2), protected=[selected, continuation]
    )
    kept = {Path(item) for item in report["kept"]}
    assert selected in kept and continuation in kept
    assert selected.is_file() and continuation.is_file()


def test_model_registered_by_reference_is_protected(tmp_path):
    run = make_run_dir(tmp_path)
    rows = []
    for update in (1, 2, 3, 4):
        path = fake_checkpoint(run / "checkpoints" / f"model_update_{update}.pt", update)
        rows.append(
            {
                "checkpoint": str(path),
                "update": update,
                "sha256": model_refs.file_sha256(path),
                "bytes": 1,
            }
        )
    write_inventory(run, rows)
    index = run / "models.json"
    teacher = (run / "checkpoints" / "model_update_1.pt").resolve()
    model_refs.register_model(index, teacher, roles=("teacher",), run=run.name)
    report = prune(
        run,
        SavePolicy(every_updates=1, keep_recent=2),
        registered=model_refs.ModelIndex(index).consumer_paths(),
    )
    assert str(teacher) in report["kept"]
    assert teacher.is_file()


def test_prune_plan_matches_execution_and_inventory(tmp_path):
    run = make_run_dir(tmp_path)
    rows = []
    for update in (2, 4, 6, 8):
        path = fake_checkpoint(run / "checkpoints" / f"model_update_{update}.pt", update)
        rows.append(
            {"checkpoint": str(path.resolve()), "update": update, "sha256": model_refs.file_sha256(path), "bytes": 1}
        )
    write_inventory(run, rows)
    milestone = (run / "checkpoints" / "model_update_2.pt").resolve()
    plan = plan_prune(
        run, SavePolicy(every_updates=2, keep_recent=2, milestones=(2,)), protected=[milestone]
    )
    executed = prune(
        run, SavePolicy(every_updates=2, keep_recent=2, milestones=(2,)), protected=[milestone]
    )
    assert plan["removed"] == executed["removed"]
    for item in executed["removed"]:
        assert not Path(item).exists()
    inventory = {row["checkpoint"] for row in read_inventory(run)}
    assert inventory == {str(Path(item)) for item in executed["kept"]}


def test_symlink_candidate_is_not_treated_as_owned(tmp_path):
    run = make_run_dir(tmp_path)
    outside = tmp_path / "external.pt"
    fake_checkpoint(outside, 99)
    link = run / "checkpoints" / "model_update_999.pt"
    link.symlink_to(outside)
    report = prune(run, SavePolicy(every_updates=1, keep_recent=1))
    assert str(link.resolve()) not in report["kept"]
    assert outside.is_file(), "外部模型不能被删"
    assert link.is_symlink()


# ---------------------------------------------------------------- 引用
def test_same_sha_two_paths_both_protected(tmp_path):
    run = make_run_dir(tmp_path)
    first = run / "checkpoints" / "model_update_3.pt"
    fake_checkpoint(first, 3)
    second = run / "exports" / "copy.pt"
    second.parent.mkdir()
    second.write_bytes(first.read_bytes())
    index = run / "models.json"
    model_refs.register_model(index, first, roles=("selected",))
    model_refs.register_model(index, second, roles=("deployment",))
    paths = model_refs.ModelIndex(index).consumer_paths()
    assert str(first.resolve()) in paths
    assert str(second.resolve()) in paths


def test_ancestors_follow_parent_chain(tmp_path):
    index = tmp_path / "models.json"
    a = fake_checkpoint(tmp_path / "a.pt", 1)
    b = fake_checkpoint(tmp_path / "b.pt", 2)
    c = fake_checkpoint(tmp_path / "c.pt", 3)
    model_refs.register_model(index, a, roles=("source",))
    model_refs.register_model(index, b, roles=("initialization",), parent=a)
    model_refs.register_model(index, c, roles=("resume",), parent=b)
    chain = model_refs.ModelIndex(index).ancestors(model_refs.file_sha256(c), str(c))
    assert model_refs.file_sha256(b) in chain
    assert model_refs.file_sha256(a) in chain


# ---------------------------------------------------------------- 配置与worker
def test_run_record_rebuilds_worker_config_without_original_file(tmp_path):
    job = make_job(tmp_path)
    record = {"config": job.as_run_record()}
    original = job.config_path
    rebuilt = job_config.job_from_record(record["config"])
    assert rebuilt.action.order == job.action.order
    assert rebuilt.ppo == job.ppo
    original.unlink()
    again = job_config.job_from_record(record["config"])
    assert again.limits == job.limits and again.backend == job.backend


def test_resume_continuation_and_budget_not_refreshed(tmp_path):
    job = make_job(
        tmp_path,
        limits={"smoke_updates": 2, "formal_updates": 6},
        runtime={"envs": 2, "steps_per_env": 1},
        storage={"checkpoint_every_updates": 2, "metrics_every_updates": 1},
    )
    handshake = submit(job, runs_root=tmp_path / "runs", run_id="resume_case", wait_s=120)
    run = Path(handshake["run"])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = json.loads((run / "state.json").read_text())
        if state.get("updates_used", 0) >= 2:
            break
        if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            break
        time.sleep(0.2)
    stop(run, reason="test stop", timeout_s=120)
    budget_before = json.loads((run / "budget.json").read_text())
    from native_stand.experiments.owner import resume_run

    resumed = resume_run(run, wait_s=120)
    assert resumed["resumed"] is True
    assert resumed["budget_refreshed"] is False
    assert resumed["budget_limits"] == budget_before["limits"]
    budget_after = json.loads((run / "budget.json").read_text())
    assert budget_after["limits"] == budget_before["limits"]
    assert budget_after["used"]["smoke"] >= budget_before["used"]["smoke"]
    assert (run / f"resume_{resumed['attempt_suffix']}.json").is_file()


def test_resume_refuses_when_budget_exhausted(tmp_path):
    job = make_job(
        tmp_path,
        limits={"smoke_updates": 1, "formal_updates": 0},
        runtime={"envs": 2, "steps_per_env": 1},
    )
    handshake = submit(job, runs_root=tmp_path / "runs", run_id="exhausted", wait_s=120)
    run = Path(handshake["run"])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = json.loads((run / "state.json").read_text())
        if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            break
        time.sleep(0.2)
    from native_stand.experiments.owner import resume_run

    result = resume_run(run, wait_s=30)
    assert result["resumed"] is False
    assert result["status"] == "NOT_RESUMABLE"
    assert "不刷新额度" in result["reason"]


def test_worker_uses_run_record_not_original_config(tmp_path):
    job = make_job(
        tmp_path,
        limits={"smoke_updates": 1, "formal_updates": 0},
        runtime={"envs": 2, "steps_per_env": 1},
    )
    handshake = submit(job, runs_root=tmp_path / "runs", run_id="record_only", wait_s=120)
    run = Path(handshake["run"])
    # 原始配置文件在启动后立刻消失：worker只认run.json。
    job.config_path.unlink()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = json.loads((run / "state.json").read_text())
        if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            break
        time.sleep(0.2)
    state = json.loads((run / "state.json").read_text())
    assert state["status"] == "COMPLETED", state.get("failure")
    assert state["updates_used"] == 1


# ---------------------------------------------------------------- 最终保存
def test_final_save_covers_non_periodic_end(tmp_path):
    job = make_job(
        tmp_path,
        limits={"smoke_updates": 2, "formal_updates": 3},
        runtime={"envs": 2, "steps_per_env": 1},
        storage={"checkpoint_every_updates": 2, "metrics_every_updates": 1},
    )
    handshake = submit(job, runs_root=tmp_path / "runs", run_id="final_save", wait_s=180)
    run = Path(handshake["run"])
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = json.loads((run / "state.json").read_text())
        if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            break
        time.sleep(0.2)
    state = json.loads((run / "state.json").read_text())
    assert state["status"] == "COMPLETED"
    assert state["updates_used"] == 5
    latest = load_full(state["latest_model"]["checkpoint"])
    assert latest["update"] == 5
    assert latest["counters"]["updates"] == 5


# ---------------------------------------------------------------- PPO装配
def test_critic_uses_declared_privileged_columns(tmp_path):
    from native_stand.learning.ppo import assembly

    job = make_job(tmp_path)
    actor = assembly.build_actor(job)
    critic = assembly.build_critic(job)
    assert actor.mlp[0].in_features == job.model.actor_input == 31
    assert critic.mlp[0].in_features == job.model.critic_input == 64
    observation = {
        "student_actor": torch.zeros(3, 31),
        "student_critic": torch.zeros(3, 64),
    }
    first = critic(observation).detach().clone()
    observation["student_critic"][:, 40] = 5.0
    second = critic(observation).detach()
    assert not torch.allclose(first, second), "Critic没有消费声明的特权列"
    observation["student_actor"][:, 3] = 9.0
    third = critic(observation).detach()
    assert torch.allclose(second, third), "Critic不应随Actor列变化"


def test_optimizer_splits_actor_and_critic_learning_rates(tmp_path):
    from native_stand.learning.ppo import assembly
    from native_stand.simulation.cpu import CpuEnvFixture

    job = make_job(tmp_path)
    env = CpuEnvFixture(job)
    ppo = assembly.assemble_ppo(
        assembly.build_actor(job), assembly.build_critic(job), env.observations(), job, device="cpu"
    )
    groups = ppo.optimizer.param_groups
    assert [group["lr"] for group in groups] == [
        job.ppo.learning_rate,
        job.ppo.critic_learning_rate,
    ]
    actor_ids = {id(p) for p in ppo.actor.parameters()}
    critic_ids = {id(p) for p in ppo.critic.parameters()}
    assert {id(p) for p in groups[0]["params"]} == actor_ids
    assert {id(p) for p in groups[1]["params"]} == critic_ids
    assert not (actor_ids & critic_ids)


def _one_update(ppo, env):
    observations = env.observations()
    actions = ppo.act(observations)
    outcome = env.step(actions)
    ppo.process_env_step(observations, outcome["rewards"], outcome["dones"], {})
    ppo.compute_returns(env.observations())
    return ppo.update()


def test_warmup_updates_only_critic_then_actor(tmp_path):
    from native_stand.learning.ppo import assembly
    from native_stand.simulation.cpu import CpuEnvFixture

    job = make_job(tmp_path, ppo={"critic_warmup_updates": 1, "epochs": 1, "minibatches": 1})
    env = CpuEnvFixture(job)
    ppo = assembly.assemble_ppo(
        assembly.build_actor(job), assembly.build_critic(job), env.observations(), job, device="cpu"
    )
    actor_before = {k: v.detach().clone() for k, v in ppo.actor.state_dict().items()}
    critic_before = {k: v.detach().clone() for k, v in ppo.critic.state_dict().items()}

    ppo.actor_update_enabled = False
    _one_update(ppo, env)
    assert all(torch.equal(ppo.actor.state_dict()[k], v) for k, v in actor_before.items())
    assert not all(torch.equal(ppo.critic.state_dict()[k], v) for k, v in critic_before.items())

    ppo.actor_update_enabled = True
    _one_update(ppo, env)
    assert not all(torch.equal(ppo.actor.state_dict()[k], v) for k, v in actor_before.items())


def test_full_state_restore_reproduces_next_update(tmp_path):
    from native_stand.learning.ppo import assembly
    from native_stand.simulation.cpu import CpuEnvFixture

    job = make_job(tmp_path, ppo={"epochs": 1, "minibatches": 1})
    torch.manual_seed(11)
    env = CpuEnvFixture(job)
    ppo = assembly.assemble_ppo(
        assembly.build_actor(job), assembly.build_critic(job), env.observations(), job, device="cpu"
    )
    _one_update(ppo, env)
    state = assembly.capture_full(ppo, counters={"updates": 1, "samples": 2})

    torch.manual_seed(11)
    env_resumed = CpuEnvFixture(job)
    ppo_resumed = assembly.assemble_ppo(
        assembly.build_actor(job),
        assembly.build_critic(job),
        env_resumed.observations(),
        job,
        device="cpu",
    )
    assembly.restore_full(ppo_resumed, state)
    env_resumed.load_runtime_state(env.runtime_state())

    probe = {"student_actor": torch.linspace(-1, 1, 31).repeat(3, 1)}
    with torch.no_grad():
        reference = ppo.actor(probe)
        restored = ppo_resumed.actor(probe)
    assert torch.allclose(reference, restored), "恢复后的Actor必须是同一函数"
    # 下一个update在两侧都从头采样，随机序列一致时结果相同。
    torch.manual_seed(4242)
    left = _one_update(ppo, env)
    torch.manual_seed(4242)
    right = _one_update(ppo_resumed, env_resumed)
    assert left.keys() == right.keys()
    for key in left:
        assert left[key] == pytest.approx(right[key], rel=1e-5, abs=1e-7)


# ---------------------------------------------------------------- 分派
def test_world_model_training_is_refused(tmp_path):
    from native_stand.simulation import dispatch

    job = make_job(tmp_path, algorithm="world_model")
    with pytest.raises(dispatch.DispatchError, match="不支持"):
        dispatch.require_supported(job, "train")


def test_evaluate_requires_ppo(tmp_path):
    from native_stand.simulation import dispatch

    job = make_job(tmp_path, algorithm="world_model")
    with pytest.raises(dispatch.DispatchError):
        dispatch.require_supported(job, "evaluate")


def test_stage_order_respects_remaining_budget(tmp_path):
    from native_stand.experiments.budget import Budget
    from native_stand.simulation import dispatch

    job = make_job(tmp_path)
    budget = Budget.create(tmp_path / "budget.json", smoke_updates=2, formal_updates=3)
    assert dispatch.stage_order(job, budget) == ["smoke", "formal"]
    budget.reserve("a", "smoke", tmp_path / "out")
    budget.charge("a", tmp_path / "out", updates=2)
    budget.finalize("a", tmp_path / "out", "COMPLETED")
    assert dispatch.stage_order(job, budget) == ["formal"]


# ---------------------------------------------------------------- 状态写入
def test_terminal_is_idempotent_and_rejects_conflicts(tmp_path):
    store = RunStateStore(tmp_path)
    store.create(run_id="r", owner={"pid": os.getpid()}, stage="smoke")
    first = store.terminal("COMPLETED", failure=None, updates_used=5)
    again = store.terminal("COMPLETED", failure=None, updates_used=5)
    assert again["revision"] == first["revision"], "相同终态重复提交必须是无操作"
    with pytest.raises(StateError):
        store.terminal("COMPLETED", failure="changed after terminal", updates_used=5)
    with pytest.raises(StateError):
        store.terminal("FAILED", failure=None, updates_used=5)


def test_launcher_cannot_overwrite_after_claim(tmp_path):
    store = RunStateStore(tmp_path)
    store.create(run_id="r", owner={"pid": 1111, "role": "launcher"}, stage="smoke")
    store.claim(pid=2222)
    with pytest.raises(StateError, match="只允许owner写入"):
        store.update(writer_pid=1111, updates_used=3)
    assert store.update(writer_pid=2222, updates_used=3)["updates_used"] == 3


# ---------------------------------------------------------------- 握手与停止
def test_worker_crash_reports_failure(tmp_path, monkeypatch):
    job = make_job(tmp_path, limits={"smoke_updates": 1, "formal_updates": 0})
    from native_stand.experiments import owner as owner_module

    monkeypatch.setattr(owner_module, "WORKER_MODULE", "native_stand.experiments.missing_module")
    handshake = owner_module.submit(job, runs_root=tmp_path / "runs", run_id="crash", wait_s=30)
    assert handshake["worker_started"] is False
    assert handshake["worker_reported"]["status"] == "FAILED"
    assert handshake["worker_exit_code"] not in (None, 0)


def test_stop_distinguishes_request_signal_and_exit(tmp_path):
    job = make_job(
        tmp_path,
        limits={"smoke_updates": 50, "formal_updates": 0},
        runtime={"envs": 2, "steps_per_env": 1},
        ppo={"epochs": 1, "minibatches": 1},
    )
    handshake = submit(job, runs_root=tmp_path / "runs", run_id="stop_case", wait_s=120)
    run = Path(handshake["run"])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = json.loads((run / "state.json").read_text())
        if state.get("status") == "RUNNING" and state.get("updates_used"):
            break
        time.sleep(0.2)
    dry = stop(run, dry_run=True)
    assert dry["signal_sent"] is False and dry["stopped"] is False
    result = stop(run, reason="test", timeout_s=120)
    assert result["request_recorded"] is True
    assert result["signal_sent"] is True
    assert result["exited"] is True
    assert result["status"] == "STOPPED_BOUNDED"


def test_process_identity_rejects_reused_pid(tmp_path):
    from native_stand.experiments.owner import process_identity, process_state

    state = process_state(os.getpid())
    token = "python" if "python" in state["cmdline"] else state["cmdline"].split(" ")[0]
    assert process_identity(os.getpid(), token)["status"] == "OK"
    mismatch = process_identity(os.getpid(), token, start_ticks="1")
    assert mismatch["status"] == "MISMATCH"


# ---------------------------------------------------------------- 评估与蒸馏
def test_evaluation_writes_per_condition_per_env(tmp_path):
    from native_stand.learning.ppo.evaluate import evaluate_checkpoint

    job = make_job(
        tmp_path,
        runtime={"envs": 3, "steps_per_env": 1},
        evaluation={
            "envs": 2,
            "scored_envs": 1,
            "seconds": 0.01,
            "seed": 5,
            "conditions": [{"name": "nominal", "command": [0.0, 0.0, 0.0]}],
        },
    )
    from native_stand.learning.ppo import assembly as ppo_assembly

    actor = ppo_assembly.build_actor(job)
    critic = ppo_assembly.build_critic(job)
    checkpoint = save_full(
        tmp_path / "model.pt",
        {
            "actor": {k: v.detach().clone() for k, v in actor.state_dict().items()},
            "critic": {k: v.detach().clone() for k, v in critic.state_dict().items()},
            "optimizer": {"state": {}, "param_groups": [{"lr": 1e-5, "params": []}]},
            "std": actor.log_std.detach().clone(),
            "rng": torch.get_rng_state(),
            "counters": {"updates": 4, "samples": 8},
        },
    )
    result = evaluate_checkpoint(job, checkpoint, kind="final", output_dir=tmp_path / "evaluations")
    assert result["kind"] == "final"
    assert result["capability_accepted"] is False
    rows = result["conditions"]
    assert len(rows) == 1
    # 独立评估按 evaluation.envs 构造环境，不使用训练环境数。
    assert len(rows[0]["per_env"]) == 2
    assert {"env_index", "status", "return"} <= set(rows[0]["per_env"][0])
    assert Path(result["path"]).is_file()


def test_distillation_holdout_not_used_for_selection(tmp_path):
    import numpy as np

    from native_stand.learning.distillation.loop import holdout_loss, run_distillation
    from native_stand.learning.ppo import assembly

    job = make_job(tmp_path)
    torch.manual_seed(3)
    actor = assembly.build_actor(job)
    critic = assembly.build_critic(job)
    teacher = save_full(
        tmp_path / "teacher.pt",
        {
            "actor": {k: v.detach().clone() for k, v in actor.state_dict().items()},
            "critic": {k: v.detach().clone() for k, v in critic.state_dict().items()},
            "optimizer": {"state": {}, "param_groups": [{"lr": 1e-5, "params": []}]},
            "std": actor.log_std.detach().clone(),
            "rng": torch.get_rng_state(),
            "counters": {"updates": 0, "samples": 0},
            "network_architecture": {"actor_hidden_sizes": list(job.model.actor_hidden_dims)},
        },
    )
    states = np.random.RandomState(1).randn(120, 31).astype("float32")
    episodes = np.array([index // 20 for index in range(120)])
    dataset = tmp_path / "states.npz"
    np.savez(dataset, states=states, episodes=episodes)
    record = run_distillation(
        job,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "distill",
        rounds=6,
        batch_size=32,
    )
    assert record["holdout_used_for_selection"] is False
    assert record["best_round"] <= record["completed_rounds"]
    split = record["split"]
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["holdout"])
    assert Path(record["candidate"]["path"]).is_file()
    holdout = holdout_loss(job, tmp_path / "distill")
    assert holdout["holdout_episodes"] == len(split["holdout"])


def test_training_run_does_not_invent_best_or_accepted(tmp_path):
    job = make_job(
        tmp_path,
        limits={"smoke_updates": 1, "formal_updates": 0},
        runtime={"envs": 2, "steps_per_env": 1},
    )
    handshake = submit(job, runs_root=tmp_path / "runs", run_id="no_best", wait_s=120)
    run = Path(handshake["run"])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = json.loads((run / "state.json").read_text())
        if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            break
        time.sleep(0.2)
    state = json.loads((run / "state.json").read_text())
    assert state["best_model"] is None and state["selected_model"] is None
    assert state["capability_accepted"] is False
