"""算法边界：warmup语义、完整恢复状态、蒸馏按episode划分。"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from pathlib import Path

import pytest

from native_stand.artifacts.checkpoint import CheckpointError, export_actor, load_full, save_full
from native_stand.artifacts.io import read_json
from native_stand.experiments.config import load_job_config
from native_stand.experiments.distillation_job import build_config
from native_stand.experiments.state import RunStateStore
from native_stand.learning.distillation.splits import SplitError, split_by_episode
from native_stand.learning.ppo.assembly import actor_update_allowed, warmup_updates

FROZEN_TEST_SCOPE = "PORTABLE"  # 自包含CPU工程契约：tmp fixture + 真实代码，无本机runs/资产依赖

PROJECT = Path(__file__).resolve().parents[1]
BASE = json.loads((PROJECT / "configs" / "jobs" / "motion_supported015_dqgain.json").read_text())

torch = pytest.importorskip("torch")


def _job(tmp_path: Path, **overrides):
    payload = json.loads(json.dumps(BASE))
    payload["limits"].update(smoke_updates=12, formal_updates=no_smoke_overlap(overrides))
    for key, value in overrides.items():
        if key in ("limits", "ppo"):
            payload.setdefault(key, {}).update(value)
        else:
            payload[key] = value
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload))
    return load_job_config(path)


def no_smoke_overlap(overrides) -> int:
    limits = overrides.get("limits") or {}
    return int(limits.get("formal_updates", 9988))


def test_smoke12_first_five_are_critic_only(tmp_path):
    job = _job(tmp_path)
    assert job.ppo.critic_warmup_updates == 5
    allowed = [actor_update_allowed(job, "smoke", update) for update in range(1, 13)]
    assert allowed[:5] == [False] * 5
    assert allowed[5:] == [True] * 7
    assert sum(allowed) == 7


def test_formal_does_not_repeat_warmup(tmp_path):
    job = _job(tmp_path, ppo={"critic_warmup_updates": 5})
    assert warmup_updates(job, "formal", completed_updates=12) == 0
    assert all(
        actor_update_allowed(job, "formal", update, completed_updates=12) for update in range(1, 6)
    )
    assert warmup_updates(job, "smoke", completed_updates=0) == 5


def test_budget_keeps_smoke_and_formal_separate(tmp_path):
    job = _job(tmp_path, limits={"smoke_updates": 12, "formal_updates": 9988})
    assert job.limits.smoke_updates + job.limits.formal_updates == 10000


def _full_state() -> dict:
    generator = torch.Generator().manual_seed(3)
    return {
        "actor": {"w": torch.rand((3, 3), generator=generator)},
        "critic": {"w": torch.rand((2, 2), generator=generator)},
        "optimizer": {"state": {0: {"exp_avg": torch.ones(3)}}, "param_groups": [{"lr": 1e-5}]},
        "std": torch.full((6,), 0.02),
        "rng": torch.get_rng_state(),
        "counters": {"updates": 12, "samples": 768},
        "motion_course_state": {"stage": 0},
        "symmetry_mix_state": {"mix": [0.5, 0.5]},
        "source_identity": {"modules_sha256": "d" * 64},
    }


def test_full_state_roundtrip_preserves_all_keys(tmp_path):
    state = _full_state()
    path = save_full(tmp_path / "full.pt", state)
    restored = load_full(path)
    assert torch.equal(restored["actor"]["w"], state["actor"]["w"])
    assert torch.equal(restored["optimizer"]["state"][0]["exp_avg"], torch.ones(3))
    assert torch.equal(restored["std"], state["std"])
    assert restored["counters"]["updates"] == 12
    assert restored["motion_course_state"] == {"stage": 0}
    assert torch.equal(restored["rng"], state["rng"])


def test_rng_restore_reproduces_next_draw(tmp_path):
    torch.manual_seed(123)
    state = _full_state()
    state["rng"] = torch.get_rng_state()
    path = save_full(tmp_path / "full.pt", state)
    expected = torch.rand(3)
    torch.manual_seed(999)  # 打乱当前序列
    restored = load_full(path)
    torch.set_rng_state(restored["rng"])
    assert torch.allclose(torch.rand(3), expected)


def test_actor_only_export_is_not_a_recovery_state(tmp_path):
    state = _full_state()
    path = save_full(tmp_path / "full.pt", state)
    exported = export_actor(
        tmp_path / "actor.pt", path, metadata={"interface_version": "multiskill_mcu_v2"}
    )
    with pytest.raises(CheckpointError):
        load_full(exported)
    import torch as _torch

    payload = _torch.load(exported, map_location="cpu", weights_only=False)
    assert payload["actor_only"] is True
    assert payload["exports_full_recovery_state"] is False
    assert payload["source_checkpoint_sha256"]


def test_resume_config_records_full_mode_and_sha(tmp_path):
    from native_stand.experiments.config import with_initialization

    job = _job(tmp_path)
    checkpoint = save_full(tmp_path / "full.pt", _full_state())
    resumed = with_initialization(job, checkpoint)
    assert resumed.initialization.mode == "full"
    assert resumed.initialization.checkpoint == checkpoint
    assert len(resumed.initialization.sha256) == 64
    assert resumed.limits == job.limits  # 不从checkpoint推断预算


def test_episode_split_has_no_leakage():
    split = split_by_episode(range(20), validation_fraction=0.2, holdout_fraction=0.2, seed=5)
    assert len(split.train) + len(split.validation) + len(split.holdout) == 20
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.holdout)
    assert set(split.validation).isdisjoint(split.holdout)
    assert split.as_record()["holdout"]


def test_episode_split_is_deterministic():
    first = split_by_episode(range(30), seed=11)
    second = split_by_episode(range(30), seed=11)
    assert first == second
    third = split_by_episode(range(30), seed=12)
    assert first != third


def test_episode_split_rejects_bad_ratios():
    with pytest.raises(SplitError):
        split_by_episode(range(5), validation_fraction=0.6, holdout_fraction=0.5)
    with pytest.raises(SplitError):
        split_by_episode([])


def test_timing_ratios_are_exact():
    from native_stand.simulation.timing import Timing

    timing = Timing(0.001, 0.005, 0.01)
    assert timing.physics_steps_per_policy == 10
    assert timing.pd_steps_per_policy == 2
    assert timing.physics_steps_per_pd == 5
    assert timing.seconds_for_updates(10, 64) == pytest.approx(6.4)


def test_timing_rejects_non_integer_ratio():
    from native_stand.simulation.timing import Timing

    with pytest.raises(ValueError, match="整数分频"):
        Timing(0.003, 0.005, 0.01)


def test_run_json_keeps_save_policy(tmp_path):
    from native_stand.experiments.owner import create_run

    job = _job(tmp_path)
    run_dir = create_run(job, runs_root=tmp_path / "runs", run_id="r")
    record = read_json(run_dir / "run.json")
    assert record["save_policy"]["checkpoint_every_updates"] == job.storage.checkpoint_every_updates
    assert record["plan"]["total_updates"] == job.limits.smoke_updates + job.limits.formal_updates
    assert record["source"]["modules_sha256"]


# ---------------------------------------------------------------- 蒸馏边界
def _distill_dataset(tmp_path, episodes=10, samples=120):
    import numpy as np

    rng = np.random.RandomState(5)
    states = rng.randn(samples, 31).astype("float32")
    ids = np.array([index % episodes for index in range(samples)])
    path = tmp_path / "states.npz"
    np.savez(path, states=states, episodes=ids)
    return path


def _teacher_checkpoint(tmp_path, job, hidden):
    import dataclasses

    from native_stand.learning.ppo import assembly

    teacher_job = dataclasses.replace(
        job, model=dataclasses.replace(job.model, actor_hidden_dims=tuple(hidden))
    )
    actor = assembly.build_actor(teacher_job)
    critic = assembly.build_critic(job)
    return save_full(
        tmp_path / "teacher.pt",
        {
            "actor": {k: v.detach().clone() for k, v in actor.state_dict().items()},
            "critic": {k: v.detach().clone() for k, v in critic.state_dict().items()},
            "optimizer": {"state": {}, "param_groups": [{"lr": 1e-5, "params": []}]},
            "std": actor.log_std.detach().clone(),
            "rng": torch.get_rng_state(),
            "counters": {"updates": 0, "samples": 0},
            "network_architecture": {"actor_hidden_sizes": list(hidden)},
        },
    )


def test_distillation_supports_large_teacher_and_small_student(tmp_path):
    from native_stand.learning.distillation.loop import run_distillation

    job = _job(tmp_path)
    teacher = _teacher_checkpoint(tmp_path, job, hidden=(256, 256))
    dataset = _distill_dataset(tmp_path)
    record = run_distillation(
        job,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "distill",
        rounds=3,
        batch_size=32,
    )
    import torch as _torch

    teacher_state = _torch.load(teacher, map_location="cpu", weights_only=False)["actor"]
    candidate = _torch.load(record["candidate"]["path"], map_location="cpu", weights_only=False)
    assert teacher_state["mlp.0.weight"].shape != candidate["actor"]["mlp.0.weight"].shape
    assert record["labels"]["source"] == "teacher_online"
    assert Path(record["latest_state"]["path"]).is_file()
    assert Path(record["best_state"]["path"]).is_file()


def test_distillation_latest_state_matches_last_round_not_best(tmp_path):
    """best早于最后一轮时，latest的模型、Adam和轮次仍然一致。"""
    import torch as _torch

    from native_stand.learning.distillation.loop import run_distillation

    job = _job(tmp_path)
    teacher = _teacher_checkpoint(tmp_path, job, hidden=(64, 64))
    dataset = _distill_dataset(tmp_path)
    record = run_distillation(
        job,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "distill",
        rounds=4,
        batch_size=16,
    )
    latest = _torch.load(record["latest_state"]["path"], map_location="cpu", weights_only=False)
    best = _torch.load(record["best_state"]["path"], map_location="cpu", weights_only=False)
    if record["best_round"] != record["completed_rounds"]:
        assert latest["round"] == record["completed_rounds"]
        assert best["round"] == record["best_round"]
        assert any(not _torch.equal(latest["student"][k], v) for k, v in best["student"].items())
    # 两个状态都带同轮optimizer与RNG
    assert latest["optimizer"]["param_groups"] and best["optimizer"]["param_groups"]
    assert set(latest["rng"]) == {"python", "numpy", "torch"}
    assert latest["split"] == record["split"] == best["split"]


def test_distillation_rejects_changed_dataset_identity(tmp_path):
    import numpy as np

    from native_stand.learning.distillation.loop import (
        DistillationError,
        holdout_loss,
        run_distillation,
    )

    job = _job(tmp_path)
    teacher = _teacher_checkpoint(tmp_path, job, hidden=(32, 32))
    dataset = _distill_dataset(tmp_path)
    run_distillation(
        job,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "distill",
        rounds=1,
        batch_size=16,
    )
    data = dict(np.load(dataset))
    data["states"] = data["states"][:10]
    data["episodes"] = data["episodes"][:10]
    np.savez(dataset, **data)
    with pytest.raises(DistillationError, match="身份"):
        holdout_loss(job, tmp_path / "distill")


def test_holdout_reads_saved_split_instead_of_resplitting(tmp_path):
    from native_stand.learning.distillation.loop import holdout_loss, run_distillation

    job = _job(tmp_path)
    teacher = _teacher_checkpoint(tmp_path, job, hidden=(64, 64))
    dataset = _distill_dataset(tmp_path)
    record = run_distillation(
        job,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "distill",
        rounds=2,
        batch_size=16,
        validation_fraction=0.3,
        holdout_fraction=0.4,
        seed=3,
    )
    holdout = holdout_loss(job, tmp_path / "distill")
    assert holdout["holdout_episodes"] == len(record["split"]["holdout"])
    assert record["split"]["holdout"] == holdout["episodes"]
    assert record["fractions"] == {"validation": 0.3, "holdout": 0.4}
    assert record["seed"] == 3


def test_precomputed_labels_without_provenance_are_rejected(tmp_path):
    import numpy as np

    from native_stand.learning.distillation.loop import DistillationError, run_distillation

    job = _job(tmp_path)
    teacher = _teacher_checkpoint(tmp_path, job, hidden=(32, 32))
    data = np.load(_distill_dataset(tmp_path))
    dataset = tmp_path / "labelled.npz"
    np.savez(
        dataset,
        states=data["states"],
        episodes=data["episodes"],
        teacher_actions=np.zeros((len(data["states"]), 6), dtype="float32"),
    )
    with pytest.raises(DistillationError, match="label_provenance"):
        run_distillation(
            job,
            dataset_path=dataset,
            teacher_checkpoint=teacher,
            output_dir=tmp_path / "distill",
            rounds=1,
            batch_size=8,
        )


def test_precomputed_labels_with_provenance_are_accepted(tmp_path):
    import json as _json

    import numpy as np

    from native_stand.artifacts.io import file_sha256
    from native_stand.learning.distillation.loop import run_distillation

    job = _job(tmp_path)
    teacher = _teacher_checkpoint(tmp_path, job, hidden=(32, 32))
    data = np.load(_distill_dataset(tmp_path))
    dataset = tmp_path / "labelled.npz"
    provenance = {
        "teacher_sha256": file_sha256(teacher),
        "input_version": job.model.interface,
        "action_version": job.action.version,
        "action_units": "policy_output",
        "action_dim": job.action.dim,
    }
    np.savez(
        dataset,
        states=data["states"],
        episodes=data["episodes"],
        teacher_actions=np.zeros((len(data["states"]), job.action.dim), dtype="float32"),
        label_provenance=np.array(_json.dumps(provenance)),
    )
    record = run_distillation(
        job,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "distill",
        rounds=1,
        batch_size=8,
    )
    assert record["labels"]["source"] == "dataset"
    assert record["labels"]["teacher_sha256"] == provenance["teacher_sha256"]
    import torch as _torch

    latest = _torch.load(record["latest_state"]["path"], map_location="cpu", weights_only=False)
    assert "shuffle_rng" in latest


def test_bad_label_provenance_fails_before_any_optimizer_step(tmp_path, monkeypatch):
    import json as _json

    import numpy as np

    from native_stand.artifacts.io import file_sha256
    from native_stand.learning.distillation.loop import DistillationError, run_distillation

    job = _job(tmp_path)
    teacher = _teacher_checkpoint(tmp_path, job, hidden=(32, 32))
    data = np.load(_distill_dataset(tmp_path))
    dataset = tmp_path / "bad.npz"
    provenance = {
        "teacher_sha256": file_sha256(teacher),
        "input_version": "wrong_input",
        "action_version": job.action.version,
        "action_units": "Nm",
        "action_dim": job.action.dim,
    }
    np.savez(
        dataset,
        states=data["states"],
        episodes=data["episodes"],
        teacher_actions=np.zeros((len(data["states"]), job.action.dim), dtype="float32"),
        label_provenance=np.array(_json.dumps(provenance)),
    )

    steps = {"n": 0}
    original = torch.optim.Adam.step

    def counting_step(self, *args, **kwargs):
        steps["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counting_step)
    with pytest.raises(DistillationError):
        run_distillation(
            job,
            dataset_path=dataset,
            teacher_checkpoint=teacher,
            output_dir=tmp_path / "distill",
            rounds=3,
            batch_size=8,
        )
    assert steps["n"] == 0, "标签不合法时不得发生任何optimizer step"


def test_segmented_distillation_matches_continuous(tmp_path):
    """连续4轮 与 2轮+恢复2轮：模型、Adam、shuffle、split、best、history 全部一致。"""
    from native_stand.learning.distillation.loop import resume_distillation, run_distillation

    job = _job(tmp_path)
    teacher = _teacher_checkpoint(tmp_path, job, hidden=(32, 32))
    dataset = _distill_dataset(tmp_path)

    continuous = run_distillation(
        job,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "continuous",
        rounds=4,
        batch_size=16,
    )
    first = run_distillation(
        job,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "segmented",
        rounds=2,
        batch_size=16,
    )
    second = resume_distillation(job, run_dir=tmp_path / "segmented", rounds=2)

    assert second["split"] == first["split"] == continuous["split"]
    assert second["completed_rounds"] == continuous["completed_rounds"] == 4
    assert [row["round"] for row in second["history"]] == [1, 2, 3, 4]
    assert second["best_round"] == continuous["best_round"]
    assert second["best_validation_loss"] == continuous["best_validation_loss"]

    a = torch.load(
        tmp_path / "continuous" / "distillation_latest.pt", map_location="cpu", weights_only=False
    )
    b = torch.load(
        tmp_path / "segmented" / "distillation_latest.pt", map_location="cpu", weights_only=False
    )
    assert b["round"] == a["round"] == 4
    for key in a["student"]:
        assert torch.allclose(a["student"][key], b["student"][key], atol=1e-8), key
    assert torch.equal(a["shuffle_rng"], b["shuffle_rng"])
    for key in a["optimizer"]["state"]:
        ka, kb = a["optimizer"]["state"][key], b["optimizer"]["state"][key]
        assert int(ka["step"]) == int(kb["step"]), "Adam step 必须一致"
        for field in ("exp_avg", "exp_avg_sq"):
            assert torch.allclose(ka[field], kb[field], atol=1e-9), field
    assert a["rng"]["torch"].sum() == b["rng"]["torch"].sum()


def test_resume_keeps_earlier_best_candidate(tmp_path):
    """恢复前的最优候选优于恢复后所有轮次时，best 不得被丢弃。"""
    import torch as _torch

    from native_stand.learning.distillation.loop import resume_distillation, run_distillation

    job = _job(tmp_path)
    teacher = _teacher_checkpoint(tmp_path, job, hidden=(32, 32))
    dataset = _distill_dataset(tmp_path)
    first = run_distillation(
        job,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "run",
        rounds=3,
        batch_size=16,
    )
    best_before = _torch.load(
        tmp_path / "run" / "distillation_best.pt", map_location="cpu", weights_only=False
    )
    second = resume_distillation(job, run_dir=tmp_path / "run", rounds=1)
    if second["best_validation_loss"] < first["best_validation_loss"]:
        # 新轮次确实更好：best 更新到新轮，旧best不再作为候选。
        assert second["best_round"] == first["completed_rounds"] + 1
    else:
        # 旧best更好：必须保留原候选与权重。
        assert second["best_round"] == first["best_round"]
        kept = _torch.load(
            tmp_path / "run" / "distillation_best.pt", map_location="cpu", weights_only=False
        )
        assert kept["round"] == best_before["round"]
        for key in best_before["student"]:
            assert _torch.equal(best_before["student"][key], kept["student"][key])


def _wait_terminal(run: Path, timeout=900.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = json.loads((run / "state.json").read_text())
        if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            return state
        time.sleep(0.2)
    raise TimeoutError("worker未在时限内结束")


def _bc_run(tmp_path, run_id, *, quota, job):
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.owner import create_run

    run = create_run(job, runs_root=tmp_path / "runs", run_id=run_id)
    Budget.create(run / "budget.json", smoke_updates=0, formal_updates=0, distill_rounds=quota)
    RunStateStore(run).create(run_id=run_id, owner={"pid": 0}, stage="distill")
    return run


def _read_state(run: Path) -> dict:
    return json.loads((run / "state.json").read_text())


def _bc_fixture(tmp_path, *, student=(32,)):
    """共用BC fixture：小Student、合成数据、Teacher；对照使用同一拓扑与seed。

    使用显式model字段的旧式配置（cpu_motion_job.json），使小网络可写入
    CLI实际读取的JSON，而不是藏在preset里。
    """
    payload = json.loads((PROJECT / "tests" / "fixtures" / "cpu_motion_job.json").read_text())
    payload["model"]["actor_hidden_dims"] = list(student)
    payload["model"]["critic_hidden_dims"] = list(student)
    payload["limits"]["smoke_updates"] = 0
    payload["limits"]["formal_updates"] = 0
    bc_config = tmp_path / "bc_job.json"
    bc_config.write_text(json.dumps(payload))
    small = load_job_config(bc_config)
    teacher = _teacher_checkpoint(tmp_path, small, hidden=(32, 32))
    dataset = _distill_dataset(tmp_path)
    distill_job, cfg = build_config(
        small,
        dataset=dataset,
        teacher=teacher,
        rounds=1,
        batch_size=16,
        learning_rate=1e-3,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=7,
    )
    return small, distill_job, cfg, dataset, teacher, bc_config


def _read_state(run: Path) -> dict:
    return json.loads((run / "state.json").read_text())


def _wait_terminal(run: Path, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _read_state(run)
        if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            return state
        time.sleep(0.2)
    raise AssertionError(
        f"worker未在{timeout}s内结束; state={_read_state(run).get('status')} "
        f"budget={json.loads((run / 'budget.json').read_text())['used']}"
    )


def _wait_worker_exit(pid: int, timeout: float = 60.0) -> None:
    """回收自己启动的worker：waitpid收割，Z状态视为已退出。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return
        status = (
            Path(f"/proc/{pid}/status").read_text() if Path(f"/proc/{pid}/status").is_file() else ""
        )
        if status.split("\n")[0].startswith("State:\tZ"):
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
            return
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        time.sleep(0.2)
    raise AssertionError(
        f"worker {pid} 未退出（state={status.split(chr(10))[0] if status else 'unknown'}）"
    )


def _sha(path: Path) -> str:
    from native_stand.artifacts.io import file_sha256

    return file_sha256(path)


def test_bc_cli_submit_single_round(tmp_path):
    """真实CLI提交：owner启动CPU worker完成单轮；退出码0、回执正确。"""

    small, _dj, _cfg, dataset, teacher, job = _bc_fixture(tmp_path)
    result = _wait_cli_submit(tmp_path, job, dataset, teacher, run_id="bc_cli")
    run = Path(result["run"])
    assert result["worker_pid"] > 0
    state = _wait_terminal(run)
    assert state["status"] == "COMPLETED" and state["rounds_used"] == 1
    assert not state.get("updates_used"), "BC不得写PPO updates"
    _wait_worker_exit(result["worker_pid"])
    budget = json.loads((run / "budget.json").read_text())
    assert budget["limits"]["distill"] == 1 and budget["used"]["distill"] == 1
    # 提交预算一致：BC无PPO可执行额度
    assert budget["limits"]["smoke"] == 0 and budget["limits"]["formal"] == 0


def _wait_cli_submit(tmp_path, config, dataset, teacher, *, run_id, rounds=1) -> dict:
    import contextlib
    import io

    from native_stand.cli import main as cli_main

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = cli_main(
            [
                "distill",
                "--config",
                str(config),
                "--json",
                "--runs-root",
                str(tmp_path / "runs"),
                "--dataset",
                str(dataset),
                "--teacher",
                str(teacher),
                "--rounds",
                str(rounds),
                "--run-id",
                run_id,
            ]
        )
    assert code == 0, (
        f"distill退出码{code}: stdout={stdout.getvalue()[-300:]} stderr={stderr.getvalue()[-1500:]}"
    )

    # 回执直接来自CLI输出JSON（--json为pretty-print，整体解析）
    text = stdout.getvalue()
    begin = text.index("{")
    payload = json.loads(text[begin : text.rindex("}") + 1])
    if payload.get("action") == "distill-submitted":
        return payload["summary"]
    raise AssertionError(f"没有提交回执: {text[-400:]}")


def test_bc_cli_stop_then_resume(tmp_path):
    """after_round=2确定性停止 → CLI resume剩余2轮；额度不刷新。"""
    from native_stand.cli import main as cli_main
    from native_stand.experiments import state as state_mod

    small, _dj, _cfg, dataset, teacher, job = _bc_fixture(tmp_path)
    result = _wait_cli_submit(tmp_path, job, dataset, teacher, run_id="bc_stop_resume", rounds=4)
    run = Path(result["run"])
    # 确定性停止：第1轮完成（rounds_used=1）后写入after_round=2，
    # 轮边界检查保证恰好停在累计第2轮（不依赖线程抢时序）。
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if (_read_state(run).get("rounds_used") or 0) >= 1:
            break
        time.sleep(0.02)
    state_mod.write_stop_request(run, reason="test", requester="pytest", after_round=2)
    state = _wait_terminal(run)
    assert state["status"] == "STOPPED_BOUNDED", state
    assert state["rounds_used"] == 2, state
    _wait_worker_exit(result["worker_pid"])

    # CLI resume：剩余2轮
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = cli_main(["resume", "--json", "--run", str(run)])
    assert code == 0, (
        f"resume退出码{code}: stdout={stdout.getvalue()[-300:]} stderr={stderr.getvalue()[-500:]}"
    )
    state = _wait_terminal(run)
    assert state["status"] == "COMPLETED" and state["rounds_used"] == 4
    budget = json.loads((run / "budget.json").read_text())
    assert budget["limits"]["distill"] == 4 and budget["used"]["distill"] == 4
    latest = sorted(run.glob("distill*/**/distillation_latest.pt"))[-1]
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert [row["round"] for row in payload["history"]] == [1, 2, 3, 4]
    # 身份：SHA与实际文件一致
    assert state["latest_model"]["sha256"] == _sha(latest)
    actor = sorted(run.glob("distill*/**/student_actor.pt"))[-1]
    assert state["best_actor"]["sha256"] == _sha(actor)


def test_bc_zero_round_stop_and_resume(tmp_path):
    from native_stand.experiments.worker import main as worker_main

    """启动即停止：0轮、无虚假checkpoint；CLI恢复完成原额度。"""
    from native_stand.cli import main as cli_main
    from native_stand.experiments import state as state_mod
    from native_stand.experiments.distillation_job import build_config

    small, _dj, _cfg, dataset, teacher, job = _bc_fixture(tmp_path)
    distill_job, _cfg = build_config(
        small,
        dataset=dataset,
        teacher=teacher,
        rounds=2,
        batch_size=16,
        learning_rate=1e-3,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=7,
    )
    run = _bc_run(tmp_path, "bc_zero", quota=2, job=distill_job)
    state_mod.write_stop_request(run, reason="test", requester="pytest")
    assert worker_main(["--run-dir", str(run)]) == 0
    state = _read_state(run)
    assert state["status"] == "STOPPED_BOUNDED"
    assert not state.get("rounds_used")
    assert not list(run.glob("distill*/**/distillation_latest.pt"))
    budget = json.loads((run / "budget.json").read_text())
    assert budget["used"]["distill"] == 0

    # CLI恢复：完成原额度2轮
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = cli_main(["resume", "--json", "--run", str(run)])
    assert code == 0, (
        f"resume退出码{code}: {stdout.getvalue()[-200:]} ERR={stderr.getvalue()[-500:]}"
    )
    state = _wait_terminal(run)
    assert state["status"] == "COMPLETED" and state["rounds_used"] == 2
    budget = json.loads((run / "budget.json").read_text())
    assert budget["used"]["distill"] == 2 and budget["limits"]["distill"] == 2


def test_bc_resume_rejections(tmp_path):
    from native_stand.experiments.worker import main as worker_main

    """额度耗尽与RUNNING均拒绝；不启动额外进程。"""
    from native_stand.experiments.distillation_job import _precheck_resume

    small, _dj, _cfg, dataset, teacher, job = _bc_fixture(tmp_path)
    distill_job, _cfg = build_config(
        small,
        dataset=dataset,
        teacher=teacher,
        rounds=1,
        batch_size=16,
        learning_rate=1e-3,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=7,
    )
    run = _bc_run(tmp_path, "bc_reject", quota=1, job=distill_job)
    assert worker_main(["--run-dir", str(run)]) == 0
    with pytest.raises(ValueError, match="额度已用尽"):
        _precheck_resume(run)
    state = _read_state(run)
    state["status"] = "RUNNING"
    (run / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="RUNNING"):
        _precheck_resume(run)


def test_bc_identity_round_boundary(tmp_path):
    from native_stand.experiments.worker import main as worker_main

    """round边界保存的latest身份被state登记且SHA一致；best Actor单独登记。"""
    from native_stand.experiments.distillation_job import build_config

    small, _dj, _cfg, dataset, teacher, job = _bc_fixture(tmp_path)
    distill_job, _cfg = build_config(
        small,
        dataset=dataset,
        teacher=teacher,
        rounds=2,
        batch_size=16,
        learning_rate=1e-3,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=7,
    )
    run = _bc_run(tmp_path, "bc_identity", quota=2, job=distill_job)
    assert worker_main(["--run-dir", str(run)]) == 0
    state = _read_state(run)
    latest = sorted(run.glob("distill*/**/distillation_latest.pt"))[-1]
    actor = sorted(run.glob("distill*/**/student_actor.pt"))[-1]
    assert state["latest_model"]["checkpoint"] == str(latest.resolve())
    assert state["latest_model"]["sha256"] == _sha(latest)
    assert state["best_actor"]["sha256"] == _sha(actor)


def test_bc_failure_round_boundary(tmp_path):
    from native_stand.experiments.worker import main as worker_main

    """第2轮后异常：上一已发布状态可读取；消耗如实；FAILED无RUNNING。"""
    from native_stand.experiments.distillation_job import build_config

    small, _dj, _cfg, dataset, teacher, job = _bc_fixture(tmp_path)
    distill_job, _cfg = build_config(
        small,
        dataset=dataset,
        teacher=teacher,
        rounds=2,
        batch_size=16,
        learning_rate=1e-3,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=7,
    )
    run = _bc_run(tmp_path, "bc_fail", quota=2, job=distill_job)

    # 在第2轮首个optimizer step注入：以state的rounds_used>=1为明确边界
    original_step = torch.optim.Adam.step
    injection = {"fired": False}

    def counting_step(self, *args, **kwargs):
        state_file = run / "state.json"
        if state_file.is_file():
            current = json.loads(state_file.read_text()).get("rounds_used") or 0
            if current >= 1:
                injection["fired"] = True
                raise RuntimeError("注入的失败")
        return original_step(self, *args, **kwargs)

    torch.optim.Adam.step = counting_step
    try:
        code = worker_main(["--run-dir", str(run)])
    finally:
        torch.optim.Adam.step = original_step
    assert code == 1
    assert injection["fired"], "注入未触发"
    state = _read_state(run)
    assert state["status"] == "FAILED"
    budget = json.loads((run / "budget.json").read_text())
    assert all(row["status"] != "RUNNING" for row in budget["attempts"].values())
    # 已完成轮次有登记的latest且可读取
    if state.get("rounds_used"):
        registered = state["latest_model"]
        assert Path(registered["checkpoint"]).is_file()
        assert registered["sha256"] == _sha(Path(registered["checkpoint"]))
        payload = torch.load(registered["checkpoint"], map_location="cpu", weights_only=False)
        assert payload["round"] == state["rounds_used"]


def test_bc_init_failure_settles_orphan(tmp_path, monkeypatch):
    from native_stand.experiments.worker import main as worker_main

    """预算预留后MetricsWriter失败：无残留RUNNING/SUBMITTED。"""
    import dataclasses as _dc

    from native_stand.experiments import worker as worker_mod
    from native_stand.experiments.distillation_job import build_config

    job = _job(tmp_path)
    small = _dc.replace(
        job, model=_dc.replace(job.model, actor_hidden_dims=(32,), critic_hidden_dims=(32,))
    )
    teacher = _teacher_checkpoint(tmp_path, small, hidden=(32, 32))
    dataset = _distill_dataset(tmp_path)
    distill_job, _cfg = build_config(
        small,
        dataset=dataset,
        teacher=teacher,
        rounds=1,
        batch_size=16,
        learning_rate=1e-3,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=7,
    )
    run = _bc_run(tmp_path, "bc_init", quota=1, job=distill_job)

    class FailingMetrics:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("注入的MetricsWriter失败")

    monkeypatch.setattr(worker_mod, "MetricsWriter", FailingMetrics)
    assert worker_main(["--run-dir", str(run)]) == 1
    state = _read_state(run)
    assert state["status"] == "FAILED", state
    budget = json.loads((run / "budget.json").read_text())
    assert all(row["status"] != "RUNNING" for row in budget["attempts"].values())


def test_bc_continuous_matches_segmented(tmp_path):
    """同架构/seed/超参/划分：连续4轮 与 2+2 一致。"""
    from native_stand.learning.distillation.loop import resume_distillation, run_distillation

    small, _dj, _cfg, dataset, teacher, job = _bc_fixture(tmp_path)
    run_distillation(
        small,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "continuous",
        rounds=4,
        batch_size=16,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=7,
    )
    run_distillation(
        small,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "segmented",
        rounds=2,
        batch_size=16,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=7,
    )
    resume_distillation(small, run_dir=tmp_path / "segmented", rounds=2)
    a = torch.load(
        tmp_path / "continuous" / "distillation_latest.pt", map_location="cpu", weights_only=False
    )
    b = torch.load(
        tmp_path / "segmented" / "distillation_latest.pt", map_location="cpu", weights_only=False
    )
    assert b["round"] == a["round"] == 4
    for key in a["student"]:
        assert torch.allclose(a["student"][key], b["student"][key], atol=1e-8), key
    assert torch.equal(a["shuffle_rng"], b["shuffle_rng"])
    assert a["split"] == b["split"]
    assert a["best"]["round"] == b["best"]["round"]
    assert [row["round"] for row in a["history"]] == [row["round"] for row in b["history"]]


def test_missing_dataset_rejected_before_run_creation(tmp_path):
    from native_stand.experiments.distillation_job import build_config

    job = _job(tmp_path)
    teacher = _teacher_checkpoint(tmp_path, job, hidden=(32, 32))
    with pytest.raises(FileNotFoundError, match="数据集不存在"):
        build_config(
            job,
            dataset=tmp_path / "missing.npz",
            teacher=teacher,
            rounds=1,
            batch_size=16,
            learning_rate=1e-3,
            validation_fraction=0.2,
            holdout_fraction=0.2,
            seed=1,
        )
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())


def test_bc_single_round(tmp_path):
    """单轮BC：COMPLETED、消耗1、latest round=1且含训练后参数。"""
    from native_stand.experiments.distillation_job import build_config
    from native_stand.experiments.worker import main as worker_main

    small, _dj, _cfg, dataset, teacher, bc_config = _bc_fixture(tmp_path)
    distill_job, _cfg = build_config(
        small,
        dataset=dataset,
        teacher=teacher,
        rounds=1,
        batch_size=16,
        learning_rate=1e-3,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=7,
    )
    run = _bc_run(tmp_path, "bc_single", quota=1, job=distill_job)
    assert worker_main(["--run-dir", str(run)]) == 0
    state = _read_state(run)
    assert state["status"] == "COMPLETED" and state["rounds_used"] == 1
    assert not state.get("updates_used")
    budget = json.loads((run / "budget.json").read_text())
    assert budget["used"]["distill"] == 1
    latest = sorted(run.glob("distill*/**/distillation_latest.pt"))[-1]
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert payload["round"] == 1
    assert [row["round"] for row in payload["history"]] == [1]
    # 训练后的参数与Adam状态确实在snapshot里
    from native_stand.learning.ppo import assembly as _assembly

    probe_job = small
    probe_actor = _assembly.build_actor(probe_job)
    trained = payload["student"]
    assert any(not torch.equal(trained[k], probe_actor.state_dict()[k]) for k in trained)
    assert payload["optimizer"]["state"], "Adam必须有实际优化状态"


def test_bc_four_rounds_progress(tmp_path):
    """四轮BC：round按1..4推进；预算最终为4；history完整。"""
    from native_stand.experiments.distillation_job import build_config
    from native_stand.experiments.worker import main as worker_main

    small, _dj, _cfg, dataset, teacher, bc_config = _bc_fixture(tmp_path)
    distill_job, _cfg = build_config(
        small,
        dataset=dataset,
        teacher=teacher,
        rounds=4,
        batch_size=16,
        learning_rate=1e-3,
        validation_fraction=0.2,
        holdout_fraction=0.2,
        seed=7,
    )
    run = _bc_run(tmp_path, "bc_four", quota=4, job=distill_job)
    assert worker_main(["--run-dir", str(run)]) == 0
    state = _read_state(run)
    assert state["status"] == "COMPLETED" and state["rounds_used"] == 4
    budget = json.loads((run / "budget.json").read_text())
    assert budget["used"]["distill"] == 4 and budget["limits"]["distill"] == 4
    latest = sorted(run.glob("distill*/**/distillation_latest.pt"))[-1]
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert [row["round"] for row in payload["history"]] == [1, 2, 3, 4]
