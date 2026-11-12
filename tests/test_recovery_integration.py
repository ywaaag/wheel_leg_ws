"""Recovery公共执行路径CPU工程闭环：原始配置→公共worker装配→更新→保存恢复。

只验证工程接线与保存/恢复语义；不代表真实Isaac训练结果。
"""

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from native_stand.artifacts.checkpoint import save_full, load_source
from native_stand.experiments.config import load_job_config
from native_stand.learning.ppo.recovery_cubic import PhysicalActor
from native_stand.simulation.engine import GenericPPOAdapter

FROZEN_TEST_SCOPE = "PORTABLE"

STD = [0.02, 0.02, 2.0, 0.02, 0.02, 2.0]


def _contract(tmp_path: Path) -> Path:
    contract = {
        "version": "recovery_cubic_physical_Gaussian_v2",
        "center_rad": [0.0, 0.1, -0.1, 0.05],
        "leg_lower_rad": [-1.2, -1.2, -1.2, -1.2],
        "leg_upper_rad": [1.2, 1.2, 1.2, 1.2],
    }
    path = tmp_path / "recovery_training_contract.json"
    path.write_text(json.dumps(contract))
    return path


def _recovery_job(tmp_path: Path, *, overrides: dict | None = None):
    """自包含最小Recovery配置：preset=_recovery_cubic_v2 + 显式actor_only来源。"""
    payload = {
        "name": f"recovery_probe_{abs(hash(str(tmp_path))) % 100000}",
        "preset": "recovery_cubic_v2",
    }
    contract_path = _contract(tmp_path)
    payload["assets"] = {"recovery_training_contract": {"path": str(contract_path)}}
    payload["runtime"] = {
        "envs": 2,
        "steps_per_env": 2,
        "seed": 7,
        "device": "cpu",
        "headless": True,
    }
    payload["ppo"] = {
        "learning_rate": 7.3e-5,
        "critic_learning_rate": 8.6e-5,
        "epochs": 3,
        "minibatches": 4,
        "critic_warmup_updates": 1,
    }
    payload["limits"] = {"smoke_updates": 1, "formal_updates": 0}
    payload["initialization"] = {"mode": "scratch"}
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(payload.get(key), dict):
                payload[key].update(value)
            else:
                payload[key] = value
    config = tmp_path / "recovery_job.json"
    config.write_text(json.dumps(payload))
    return load_job_config(config)


def _actor_source(tmp_path: Path, job, *, sentinel: float | None = None):
    """生成Actor来源：Recovery格式（带output_contract）；sentinel标记参数身份。"""
    contract = json.loads(_contract(tmp_path).read_text())
    actor = PhysicalActor(contract, hidden_dims=job.model.actor_hidden_dims)
    if sentinel is not None:
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.fill_(sentinel)
    path = tmp_path / "actor_source.pt"
    torch.save({"actor": actor.state_dict(), "output_contract": contract}, path)
    return path, actor


def _observation_sample(job):
    return TensorDict(
        {
            "student_actor": torch.zeros(2, job.observation.actor_dim),
            "student_critic": torch.zeros(2, job.observation.critic_dim),
            "recovery_actor_valid": torch.ones(2, 1, dtype=torch.bool),
        },
        batch_size=[2],
    )


class _ScriptedEnv:
    """受控CPU物理I/O替身：奖励恒定、无终止；只验证工程路径。"""

    num_envs = 2
    device = "cpu"

    def __init__(self, job):
        self._job = job
        self._tick = 0

    def observations(self):
        return _observation_sample(self._job)

    def step(self, actions):
        self._tick += 1
        return {
            "rewards": torch.ones(2, 1),
            "terminated": torch.zeros(2, 1, dtype=torch.bool),
            "truncated": torch.zeros(2, 1, dtype=torch.bool),
            "dones": torch.zeros(2, 1, dtype=torch.bool),
            "reset_envs": 0,
            "extras": {},
            "terminal_observations": {
                "student_critic": torch.zeros(2, self._job.observation.critic_dim)
            },
        }

    def runtime_state(self):
        return {"tick": self._tick}

    def load_runtime_state(self, state):
        self._tick = state.get("tick", 0)


class _Backend:
    def __init__(self, env):
        self._env = env

    def make_env(self, job):
        return self._env


class _Context:
    def __init__(self, job, completed=0):
        self.job = job
        self.updates_used = completed

    def total_updates_before(self, stage):
        return self.updates_used

    def curriculum_state(self):
        return {}

    def restore_runtime_state(self, state):
        return None


def _adapter(job, tmp_path):
    env = _ScriptedEnv(job)
    adapter = GenericPPOAdapter(_Backend(env))
    adapter.build(_Context(job), "smoke", None)
    return adapter


def test_public_adapter_assembles_physical_actor_with_job_params(tmp_path):
    """公共worker路径（GenericPPOAdapter.build）实际创建PhysicalActor；参数来自job。"""
    job = _recovery_job(tmp_path)
    adapter = _adapter(job, tmp_path)
    ppo = adapter.ppo
    # 实际对象类型与物理动作接口
    assert isinstance(ppo.actor, PhysicalActor)
    assert hasattr(ppo.actor, "physical_mean") and hasattr(ppo.actor, "physical_std")
    # 固定物理std（float32表示；用allclose而非精确浮点字面量比较）
    assert torch.allclose(ppo.actor.physical_std, torch.tensor(STD), atol=0)
    assert ppo.actor.log_std is None if hasattr(ppo.actor, "log_std") else True
    # PPO参数逐项透传：区别于assemble默认值的值必须出现在实际对象
    assert ppo.learning_rate == job.ppo.learning_rate == 7.3e-5
    assert ppo.num_learning_epochs == job.ppo.epochs == 3
    assert ppo.num_mini_batches == job.ppo.minibatches == 4
    # 优化器参数归属：actor/critic两组、学习率一致
    assert ppo.optimizer.actor.param_groups[0]["lr"] == 7.3e-5
    assert ppo.optimizer.critic.param_groups[0]["lr"] == 8.6e-5
    ppo.optimizer.binding()  # 参数归属核对（重复/缺失即抛错）


def test_actor_only_source_loads_once_and_rebuilds_critic(tmp_path):
    """actor_only来源经正式路径加载；Critic按本次配置新建；来源只加载一次。"""
    job = _recovery_job(tmp_path)
    source_path, source_actor = _actor_source(tmp_path, job, sentinel=0.125)
    import hashlib

    job = dataclasses.replace(
        job,
        initialization=dataclasses.replace(
            job.initialization,
            mode="actor_only",
            checkpoint=str(source_path),
            sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        ),
    )
    adapter = _adapter(job, tmp_path)
    ppo = adapter.ppo
    # Actor参数确实来自来源（哨兵值）
    for parameter in ppo.actor.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 0.125))
    # 物理std仍是本次配置的固定值，不被来源覆盖
    assert torch.allclose(ppo.actor.physical_std, torch.tensor(STD), atol=0)
    # Critic不是来源对象（新建）：Critic参数未全部为哨兵值
    assert any(not torch.allclose(p, torch.full_like(p, 0.125)) for p in ppo.critic.parameters())


def test_actor_source_sha_mismatch_rejected(tmp_path):
    """来源SHA与配置声明不符在准备边界拒绝。"""
    job = _recovery_job(tmp_path)
    source_path, _ = _actor_source(tmp_path, job, sentinel=0.5)
    job = dataclasses.replace(
        job,
        initialization=dataclasses.replace(
            job.initialization,
            mode="actor_only",
            checkpoint=str(source_path),
            sha256="0" * 64,
        ),
    )
    env = _ScriptedEnv(job)
    adapter = GenericPPOAdapter(_Backend(env))
    with pytest.raises(ValueError, match="SHA"):
        adapter.build(_Context(job), "smoke", None)


def test_cpu_update_warmup_then_actor_and_fixed_std(tmp_path):
    """真实CPU优化：critic-only期Actor不变、Critic更新；warmup后Actor更新；std固定。"""
    job = _recovery_job(tmp_path)
    adapter = _adapter(job, tmp_path)
    ppo = adapter.ppo
    before_actor = {n: v.detach().clone() for n, v in ppo.actor.named_parameters()}
    before_critic = {n: v.detach().clone() for n, v in ppo.critic.named_parameters()}
    context = _Context(job)
    first = adapter.update_once(context, 1)
    assert first["critic_only"] is True
    assert all(torch.equal(v, before_actor[n]) for n, v in ppo.actor.named_parameters())
    assert any(not torch.equal(v, before_critic[n]) for n, v in ppo.critic.named_parameters())
    context.updates_used = adapter._warmup
    second = adapter.update_once(context, 2)
    assert second["critic_only"] is False
    assert second["actor_optimizer_steps"] > 0
    assert any(not torch.equal(v, before_actor[n]) for n, v in ppo.actor.named_parameters())
    # 梯度与优化器状态非空
    assert ppo.optimizer.actor.state, "Adam必须有actor优化状态"
    # 固定物理std在优化后不变
    assert torch.allclose(ppo.actor.physical_std, torch.tensor(STD), atol=0)


def test_save_restore_matches_continuous_training(tmp_path):
    """连续2轮 与 1轮→完整保存→新对象恢复→继续1轮 结果一致。

    确定性前提：CPU、固定种子、受控环境替身；不代表PhysX物理轨迹逐位一致。
    """
    job = _recovery_job(tmp_path)
    torch.manual_seed(1234)
    continuous = _adapter(job, tmp_path)
    context_c = _Context(job)
    torch.manual_seed(101)
    continuous.update_once(context_c, 1)
    context_c.updates_used = continuous._warmup
    torch.manual_seed(202)
    continuous.update_once(context_c, 2)
    final_c = {n: v.detach().clone() for n, v in continuous.ppo.actor.named_parameters()}
    opt_c = continuous.ppo.optimizer.actor.state
    critic_c = {n: v.detach().clone() for n, v in continuous.ppo.critic.named_parameters()}

    # 分段：1轮 → capture（含RNG）→ 新对象恢复 → 继续1轮
    torch.manual_seed(1234)
    segmented = _adapter(job, tmp_path)
    context_s = _Context(job)
    torch.manual_seed(101)
    segmented.update_once(context_s, 1)
    state = segmented.capture(context_s, {"updates": 1, "samples": 8})
    assert state["rng"] is not None

    torch.manual_seed(1234)
    resumed = _adapter(job, tmp_path)
    resumed.ppo = None
    # 恢复上下文携带已完成轮次：warmup进度与RNG状态来自恢复点。
    state["run_identity"] = str(tmp_path.resolve())
    resume_path = save_full(tmp_path / "recovery_resume.pt", state)
    restored_context = _Context(job, completed=1)
    restored_context.run_dir = tmp_path
    resumed.build(restored_context, "smoke", load_source(resume_path))
    torch.manual_seed(202)
    resumed.update_once(_Context(job, completed=1), 2)
    final_s = {n: v.detach().clone() for n, v in resumed.ppo.actor.named_parameters()}
    assert set(final_c) == set(final_s)
    for name in final_c:
        assert torch.equal(final_c[name], final_s[name]), name
    critic_s = {n: v.detach().clone() for n, v in resumed.ppo.critic.named_parameters()}
    for name in critic_c:
        assert torch.equal(critic_c[name], critic_s[name]), name
    # 优化器状态：Adam state以Parameter对象为键；按参数名对齐逐项比较。
    opt_s = resumed.ppo.optimizer.actor.state
    params_c = dict(continuous.ppo.actor.named_parameters())
    params_s = dict(resumed.ppo.actor.named_parameters())
    assert set(params_c) == set(params_s)
    compared = 0
    for name, p_c in params_c.items():
        slots_c = opt_c.get(p_c)
        slots_s = opt_s.get(params_s[name])
        assert (slots_c is None) == (slots_s is None), name
        if slots_c is None:
            continue
        assert set(slots_c) == set(slots_s), name
        for slot, value in slots_c.items():
            assert torch.equal(value, slots_s[slot]), (name, slot)
            compared += 1
    assert compared > 0, "优化器状态必须有可比较的张量槽"
    # 恢复后物理std一致
    assert torch.allclose(resumed.ppo.actor.physical_std, torch.tensor(STD), atol=0)


def test_raw_config_and_record_rebuild_agree(tmp_path):
    """原始配置与run记录重建形成同一装配：同一factory、同一合同、同一Actor类。"""
    job = _recovery_job(tmp_path)
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.config import job_from_record
    from native_stand.experiments.owner import create_run
    from native_stand.experiments.state import RunStateStore

    run = create_run(job, runs_root=tmp_path / "runs", run_id="probe")
    Budget.create(run / "budget.json", smoke_updates=1, formal_updates=0)
    RunStateStore(run).create(run_id="probe", owner={"pid": 0}, stage="smoke")
    rebuilt = job_from_record(json.loads((run / "run.json").read_text())["config"])
    a1 = _adapter(job, tmp_path)
    a2 = _adapter(rebuilt, tmp_path)
    assert type(a1.ppo.actor) is type(a2.ppo.actor) is PhysicalActor
    assert a1.ppo.actor.contract == a2.ppo.actor.contract
    assert a1.ppo.learning_rate == a2.ppo.learning_rate


def test_invalid_combination_rejected(tmp_path):
    """未登记组合在配置入口拒绝；不回退CPU。"""
    payload = {"name": "recovery_bad_combination", "preset": "recovery_cubic_v2"}
    payload["assets"] = {"recovery_training_contract": {"path": str(_contract(tmp_path))}}
    payload["backend"] = "isaac_recovery_v2"  # 未登记后端
    config = tmp_path / "bad_job.json"
    config.write_text(json.dumps(payload))
    with pytest.raises(Exception, match="backend"):
        load_job_config(config)


def test_height909_current_contract_policy_and_schedule(tmp_path):
    from native_stand.tasks.recovery.height_context import command, append, VERSION

    job = _recovery_job(
        tmp_path,
        overrides={
            "model": {"critic_input": 909, "critic_privileged_version": VERSION},
            "observation": {"critic_dim": 909},
            "ppo": {
                "gamma": 0.91,
                "lam": 0.90,
                "clip": 0.15,
                "value_coef": 0.7,
                "max_grad_norm": 0.6,
            },
        },
    )
    adapter = _adapter(job, tmp_path)
    assert adapter.ppo.critic.input_dim == 909
    assert isinstance(adapter.ppo.actor, PhysicalActor)
    assert adapter.ppo.gamma == 0.91 and adapter.ppo.lam == 0.90
    assert adapter.ppo.clip_param == 0.15 and adapter.ppo.value_loss_coef == 0.7
    time = torch.tensor([0.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
    h = command(torch.full_like(time, 0.34), torch.ones_like(time, dtype=torch.bool), time)
    torch.testing.assert_close(h, torch.tensor([0.34, 0.33, 0.32, 0.34, 0.36, 0.35, 0.34]))
    encoded = append(
        torch.zeros(7, 904), torch.full_like(time, 0.34), h, torch.ones_like(time, dtype=torch.bool)
    )
    assert encoded.shape == (7, 909)
    torch.testing.assert_close(encoded[:, 905], h / 0.5)
    assert encoded[:, 908].eq(1).all()


def test_actual_actor_only_recipe_full_resume_does_not_convert_as_C0_again(tmp_path):
    from dataclasses import replace
    from native_stand.experiments.config import InitializationConfig
    from native_stand.artifacts.io import file_sha256

    job = _recovery_job(tmp_path)
    source, actor = _actor_source(tmp_path, job, sentinel=0.03)
    job = replace(
        job, initialization=InitializationConfig("actor_only", source, file_sha256(source))
    )
    first = _adapter(job, tmp_path)
    context = _Context(job)
    first.update_once(context, 1)
    saved = first.capture(context, {"updates": 1, "samples": 4})
    second = _adapter(job, tmp_path)
    saved["run_identity"] = str(tmp_path.resolve())
    resume_path = save_full(tmp_path / "recovery_resume.pt", saved)
    restored_context = _Context(job, completed=1)
    restored_context.run_dir = tmp_path
    second.build(restored_context, "smoke", load_source(resume_path))
    for name, value in first.ppo.actor.state_dict().items():
        assert torch.equal(value, second.ppo.actor.state_dict()[name]), name
