"""Portable production-boundary tests; physics allocation is never performed."""

from __future__ import annotations

import copy
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from test_motion_production_integration import make_job

from native_stand.artifacts.branch_initialization import VERSION as BRANCH_VERSION
from native_stand.artifacts.checkpoint import load_complete_state, save_full
from native_stand.artifacts.io import file_sha256, read_json, write_json
from native_stand.experiments.config import load_job_config
from native_stand.learning.ppo.assembly import build_actor, build_critic
from native_stand.learning.ppo.dual_adam import DualAdam
from native_stand.learning.ppo.source_anchor import KEY
from native_stand.learning.ppo.supported import assemble_from_full
from native_stand.simulation.isaac.motion_backend import (
    MotionProductionAdapter,
    _check_source_progress,
    _resolve_source,
    candidate_strategy,
    preflight,
    validate_job,
)
from native_stand.students.resume import rng_state
from native_stand.tasks.motion import maneuver_course_v2, split_candidates
from native_stand.tasks.motion.action_contract import ActionDecoder
from native_stand.tasks.motion.course_strategy import resolve
from native_stand.tasks.motion.maneuver_statistics import RESPONSE_VERSION, episode_class

FROZEN_TEST_SCOPE = "PORTABLE"
ROOT = Path(__file__).resolve().parents[1]


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, tuple | list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


def course_configuration(branch):
    # 配方本身是受管的测试输入，完全不读取外部模型/历史目录。
    raw = read_json(ROOT / f"configs/jobs/{branch}48_u3000_prepared.json")
    return raw["assets"]["branch_course"]


def fixture(tmp_path, branch="motion", *, envs=4):
    job, contract = make_job(tmp_path, warmup=50, retention=False)
    cfg = course_configuration(branch)
    prepared = read_json(ROOT / f"configs/jobs/{branch}48_u3000_prepared.json")["assets"]
    cpu_v4_boundary = tmp_path / "physics_io_boundary.json"
    write_json(cpu_v4_boundary, {"version": "test_only_no_physics"})
    template = read_json(job.assets["runtime_template"])
    template["motion_identified_response"] = dict(
        contract=str(cpu_v4_boundary), contract_sha256=file_sha256(cpu_v4_boundary)
    )
    write_json(job.assets["runtime_template"], template)
    job = replace(
        job,
        name="branch_fixture",
        runtime=replace(job.runtime, envs=envs),
        model=replace(
            job.model,
            critic_input=877,
            critic_privileged_version="motion48_height_bank_planned_causal877_v2",
        ),
        observation=replace(job.observation, critic_dim=877),
        curriculum=replace(job.curriculum, version=cfg["version"], stages=(cfg,)),
        limits=replace(job.limits, smoke_updates=0, formal_updates=0, sample_limit=0),
        diagnostics={"training_candidate_selection": split_candidates.configuration()},
    )
    actor, critic = build_actor(job), build_critic(job)
    actor.log_std.requires_grad_(False)
    optimizer = DualAdam(
        actor.parameters(),
        critic.parameters(),
        actor_lr=job.ppo.learning_rate,
        critic_lr=job.ppo.critic_learning_rate,
    )
    # 小型数学更新仅使两Adam带真实momentum，非Motion rollout/训练。
    (
        actor.mean(torch.ones(2, 31)).square().mean()
        + critic({"student_critic": torch.ones(2, 877)}).square().mean()
    ).backward()
    optimizer.step()
    optimizer.zero_grad()
    teacher = tmp_path / "teacher.pt"
    torch.save({"actor": actor.state_dict(), "actor_only": True}, teacher)
    bank_path, meta_path = tmp_path / "bank.npz", tmp_path / "bank.json"
    commands = [[0.0, 0.0], [3.6, 0.0], [-3.6, 0.0]]
    x = torch.zeros(384, 31)
    x[:, 22:25] = torch.tensor([[v / 4, w / 4, 0.68] for v, w in commands]).repeat_interleave(
        128, 0
    )
    np.savez(
        bank_path,
        actor_obs=x.numpy(),
        source_mean=actor.mean(x).detach().numpy(),
        cohort=np.arange(3).repeat(128),
    )
    write_json(meta_path, {"commands": commands, "data_role": "train_only"})

    def ref(path):
        return dict(path=str(path), sha256=file_sha256(path))

    retention = dict(
        version="motion48_per_command_nominal_source_KL_v4",
        coefficient=0.02,
        source_sha256=file_sha256(teacher),
        commands=commands,
        maximum_states_per_cohort=64,
        bank_states_per_command=64,
        record_first_gradient=True,
        bank=ref(bank_path),
        bank_metadata=ref(meta_path),
        mask="qualified nominal causal student",
        bank_fraction_when_onpolicy=0.5,
        task_reduction="mean_over_declared_commands",
    )
    old_state = maneuver_course_v2.initial_state()
    old_state["stage"] = 33
    old_state["streaks"]["turn"] = 1
    old_state["DR_episode_counters"] = [17] * job.runtime.envs
    source = dict(
        actor=copy.deepcopy(actor.state_dict()),
        critic=copy.deepcopy(critic.state_dict()),
        optimizer=copy.deepcopy(optimizer.state_dict()),
        std=actor.log_std.exp().detach().clone(),
        rng=torch.get_rng_state(),
        training_rng=rng_state(),
        counters={},
        network_architecture={"actor_hidden_dims": [8], "critic_hidden_dims": [8]},
        action_version=job.action.version,
        output_contract=contract,
        critic_privileged_version=job.model.critic_privileged_version,
        motion_course_state=old_state,
        reward_refinement_state=[],
        symmetry_mix_state=[],
        source_anchor_actor=copy.deepcopy(actor.state_dict()),
        source_policy_retention=retention,
        warmup_updates_done=50,
        update=3000,
        recipe_updates=5097,
        online_updates_including_smoke=5097,
        source_anchor_identity=ref(teacher),
        training_candidates={"old_comparison": "not a new-branch candidate"},
        completed_episode_statistics={"old_partial": True},
    )
    selected, final = tmp_path / "selected.pt", tmp_path / "final.pt"
    save_full(selected, source)
    save_full(final, {**source, "update": 3014, "online_updates_including_smoke": 5111})
    selected_ref = dict(
        **ref(selected), model_online_updates=5097, model_local_update=3000, full_checkpoint=True
    )
    final_ref = dict(
        **ref(final), model_online_updates=5111, model_local_update=3014, full_checkpoint=True
    )
    settlement_path = tmp_path / "settlement.json"
    write_json(
        settlement_path,
        dict(
            status="STOPPED_BOUNDED",
            failure=None,
            new_budget_created=False,
            selected_checkpoint=selected_ref,
            final_checkpoint=final_ref,
            total_online_updates_consumed=5111,
            remaining_updates=4889,
            original_limit=10000,
            rolled_back_updates_are_not_refunded=True,
        ),
    )
    binding_path = tmp_path / "center_binding.json"
    write_json(
        binding_path,
        dict(
            policy_sha256=file_sha256(selected),
            large_contract_sha256=job.assets["output_contract_sha256"],
        ),
    )
    selection_path = tmp_path / "selection.json"
    write_json(
        selection_path,
        dict(
            branch=branch,
            source=selected_ref,
            source_center_binding=ref(binding_path),
            output_contract=job.assets["output_contract"],
            output_contract_sha256=job.assets["output_contract_sha256"],
        ),
    )
    branch_path = tmp_path / "branch.json"
    write_json(
        branch_path,
        dict(
            version=BRANCH_VERSION,
            branch=branch,
            selection=ref(selection_path),
            settlement=ref(settlement_path),
            source_model_online_updates=5097,
            authorization_spent_updates=5111,
            shared_remaining_updates=4889,
            budget_status="UNALLOCATED",
            branch_updates=0,
        ),
    )
    job = replace(
        job,
        initialization=replace(
            job.initialization, checkpoint=selected, sha256=file_sha256(selected)
        ),
        assets={
            **job.assets,
            "branch_course": cfg,
            "branch_initialization": ref(branch_path),
            "support_center_policy_binding": ref(binding_path),
            "historical_online_updates": 5111,
            "source_online_updates": 5097,
            "new_independent_authorization": False,
            "source_policy_retention": retention,
            "source_anchor_identity": ref(teacher),
            "runtime_template_sha256": file_sha256(job.assets["runtime_template"]),
            "delayed_execution": prepared["delayed_execution"],
            "motor_limits": prepared["motor_limits"],
            "wheel_identification_sha256": "f" * 64,
        },
    )
    job = replace(
        job,
        control=replace(
            job.control, effort_limits_nm=tuple(prepared["motor_limits"]["motor_caps_nm"])
        ),
    )
    from native_stand.tasks.motion.branch_state import confirmation_execution_identity

    cfg["confirmation_execution_sha256"] = confirmation_execution_identity(job)
    obs = TensorDict(
        dict(
            student_actor=torch.zeros(envs, 31),
            student_critic=torch.zeros(envs, 877),
            recovery_actor_valid=torch.ones(envs, 1, dtype=torch.bool),
            motion_diagnostic_cohort=torch.zeros(envs, 1),
            **{KEY: torch.full((envs, 1), -1, dtype=torch.long)},
        ),
        [envs],
    )
    return job, source, obs


@pytest.mark.parametrize("branch", ["motion", "spin"])
def test_full_selected_branch_preserves_learning_state_resets_evidence(tmp_path, branch):
    job, source, obs = fixture(tmp_path, branch)
    validate_job(job)
    loaded = _resolve_source(job, None)
    state = loaded.payload["motion_course_state"]
    assert state["stage"] == 0 and state["last_training_update"] == -1
    assert not any(state["streaks"].values()) and state["accepted_by_stage"] == {}
    assert loaded.payload["course_updates"] == 0
    assert _check_source_progress(job, loaded.payload, None) == 0
    assert loaded.payload["model_online_updates"] == 5097
    assert job.assets["historical_online_updates"] == 5111
    assert "completed_episode_statistics" not in loaded.payload
    assert source["motion_course_state"]["stage"] == 33
    algorithm = assemble_from_full(loaded.payload, obs, job)
    assert algorithm.actor_update_enabled
    equal_tree(source["actor"], algorithm.actor.state_dict())
    equal_tree(source["critic"], algorithm.critic.state_dict())
    equal_tree(source["optimizer"], algorithm.optimizer.state_dict())
    assert torch.equal(source["std"], algorithm.actor.log_std.exp())
    equal_tree(source["training_rng"], loaded.payload["training_rng"])
    with pytest.raises(ValueError, match="共同余额尚未分配"):
        preflight(job)


@pytest.mark.parametrize("branch", ["motion", "spin"])
def test_full_branch_real_adapter_capture_and_same_run_restore(tmp_path, monkeypatch, branch):
    from test_phase3_fixes import _production_context

    job, source, obs = fixture(tmp_path, branch)
    context = _production_context(tmp_path / "first", job, smoke=1)

    def allocate_cpu(adapter):
        stage = adapter.spec["curriculum"]["stages"][0]
        adapter.env = NS(
            num_envs=4,
            device="cpu",
            stage_spec=stage,
            identified_dr=NS(episodes=torch.full((4,), 17)),
            transient_events=NS(summary=lambda: {}),
        )
        adapter.env.onpolicy_episodes = episode_class(object, version=RESPONSE_VERSION)(adapter.env)
        adapter.obs = obs
        adapter.modules = {"rng_state": rng_state}

    monkeypatch.setattr(MotionProductionAdapter, "_open_environment", allocate_cpu)
    monkeypatch.setattr(MotionProductionAdapter, "_write_restoration", lambda self: None)
    adapter = MotionProductionAdapter()
    try:
        adapter.build(context, "smoke", None)
        saved = adapter.capture(context, {})
        assert saved["counters"]["new_run_updates"] == 0
        assert saved["counters"]["online_updates"] == 5111
        assert saved["model_online_updates"] == 5097
        assert saved["warmup_updates_done"] == 50
        equal_tree(source["optimizer"], saved["optimizer"])
        reference = context.save(saved, update=0, stage="smoke")
        loaded = load_complete_state(reference["checkpoint"])
        restored = MotionProductionAdapter()
        restored.build(context, "smoke", loaded)
        equal_tree(saved["actor"], restored.algorithm.actor.state_dict())
        equal_tree(saved["critic"], restored.algorithm.critic.state_dict())
        equal_tree(saved["optimizer"], restored.algorithm.optimizer.state_dict())
        assert restored.state == adapter.state
        assert (
            restored.algorithm.source_anchor.state_dict()
            == adapter.algorithm.source_anchor.state_dict()
        )
    finally:
        context.metrics.close()


@pytest.mark.parametrize("branch", ["motion", "spin"])
def test_formal_adapter_first_candidate_publish_save_and_recovery(tmp_path, monkeypatch, branch):
    """真实统计/课程/候选/save边界；结束回合累加量由CPU I/O fixture供给。"""
    from test_phase3_fixes import _production_context

    from native_stand.tasks.motion.legacy_spec_adapter import build_environment_spec

    job, original, obs = fixture(tmp_path, branch, envs=128)
    context = _production_context(tmp_path / "candidate", job, smoke=200)

    def allocate_cpu(adapter):
        stage = adapter.spec["curriculum"]["stages"][0]
        adapter.env = NS(
            num_envs=128,
            device="cpu",
            stage_spec=stage,
            identified_dr=NS(episodes=torch.full((128,), 17)),
            transient_events=NS(summary=lambda: {}),
        )
        adapter.env.onpolicy_episodes = episode_class(object, version=RESPONSE_VERSION)(adapter.env)
        adapter.obs = obs
        adapter.modules = {"rng_state": rng_state, "build_environment_spec": build_environment_spec}

    monkeypatch.setattr(MotionProductionAdapter, "_open_environment", allocate_cpu)
    monkeypatch.setattr(MotionProductionAdapter, "_write_restoration", lambda self: None)
    monkeypatch.setattr(
        MotionProductionAdapter, "physical_readback", lambda self: {"physics_steps": 0}
    )
    adapter = MotionProductionAdapter()
    try:
        adapter.build(context, "smoke", None)
        episodes = adapter.env.onpolicy_episodes
        stage = adapter.env.stage_spec
        active = torch.tensor(stage["slot_counts"]) > 0
        # 提供已结束的统计输入，不运行积分/rollout，也不把它称为30秒能力认证。
        for name in ("count", "complete", "passed", "quality"):
            getattr(episodes, name)[active] = 16
        episodes.group_sums[active, 0] = 16  # double support
        commands = torch.tensor(stage["commands"])
        episodes.group_sums[:, 1:3] = commands * 16
        episodes.group_sums[active, 8] = 0.01**2 * 16  # integrated physical HF RMS
        episodes.group_sums[active, 12] = 16  # opposite rolling
        episodes.group_sums[active, 14] = 16  # completed
        episodes.response_groups[active, 4:6] = 16
        adapter._update_scope = dict(
            stats=NS(export=lambda: {}),
            execution_stats=NS(export=lambda: {"scope": "scripted fixture"}),
            diagnostics={},
            detail={},
            channel_clips=torch.zeros(6),
            valid_samples=torch.tensor(128.0),
        )
        context.updates_used = 100  # temporary fixture counter, no real optimizer updates
        adapter.after_update(context, 100, {})
        assert context.refs.candidate_index(context.run_dir.name) is None
        assert adapter._candidate_pending["window"]["start_update"] == 1
        assert adapter._candidate_pending["window"]["end_update"] == 100
        assert adapter._candidate_pending["eligible_by_objective"] == {
            "standing": True,
            "motion": True,
        }
        snapshot = adapter.capture(context, {})
        assert snapshot["model_online_updates"] == 5197
        assert snapshot["counters"]["online_updates"] == 5211
        equal_tree(snapshot["actor"], original["actor"])
        equal_tree(snapshot["optimizer"], original["optimizer"])
        reference = context.save(snapshot, update=100, stage="smoke")
        assert file_sha256(reference["checkpoint"]) == reference["sha256"]
        assert context.refs.candidate_index(context.run_dir.name) is None
        adapter.after_save(context, reference)
        authority = context.refs.candidate_index(context.run_dir.name)
        for name in ("best_candidate", "best_standing_candidate"):
            assert authority[name]["checkpoint"] == reference["checkpoint"]
            assert authority[name]["sha256"] == reference["sha256"]
            assert (
                authority[name]["evidence"]["window"]
                == adapter.candidates[name]["evidence"]["window"]
            )
            assert authority[name]["independent_certificate"] is False
        assert str(Path(reference["checkpoint"]).resolve()) in context.refs.consumer_paths()
        # 保存发生在发布前；从含pending的真实checkpoint恢复，复用权威引用，不重复晋级。
        loaded = load_complete_state(reference["checkpoint"])
        resumed = MotionProductionAdapter()
        resumed.build(context, "smoke", loaded)
        assert resumed.state == adapter.state
        assert resumed.candidates == authority
        assert resumed._candidate_pending is None
        equal_tree(snapshot["optimizer"], resumed.algorithm.optimizer.state_dict())
    finally:
        context.metrics.close()


def test_retention_relocated_bank_and_teacher_restore_gradient_cursor(tmp_path):
    job, _, obs = fixture(tmp_path)
    loaded = _resolve_source(job, None)
    first = assemble_from_full(loaded.payload, obs, job)
    with torch.no_grad():
        first.actor.mlp[-1].bias.add_(0.02)
    frozen = copy.deepcopy(first.source_anchor.teacher.state_dict())
    rng = torch.get_rng_state().clone()
    cache = first.actor._current_distribution
    loss = first.source_anchor.loss(first.actor, obs, torch.ones(4, dtype=torch.bool))
    loss.backward()
    assert loss > 0 and first.actor.mlp[-1].bias.grad.abs().sum() > 0
    assert torch.equal(rng, torch.get_rng_state()) and first.actor._current_distribution is cache
    equal_tree(frozen, first.source_anchor.teacher.state_dict())
    assert all(p.grad is None for p in first.source_anchor.teacher.parameters())
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    mapping = {}
    for path in [
        job.assets["source_anchor_identity"]["path"],
        job.assets["source_policy_retention"]["bank"]["path"],
        job.assets["source_policy_retention"]["bank_metadata"]["path"],
    ]:
        target = relocated / Path(path).name
        shutil.copyfile(path, target)
        mapping[path] = str(target)
    mapping_file = tmp_path / "mapping.json"
    write_json(mapping_file, dict(version="runtime_path_mapping_v1", files=mapping))
    assets = {
        **job.assets,
        "path_mapping": dict(path=str(mapping_file), sha256=file_sha256(mapping_file)),
    }
    from native_stand.artifacts.runtime_paths import RuntimePaths

    paths = RuntimePaths.from_assets(assets)
    assets["source_policy_retention"] = paths.view(job.assets["source_policy_retention"])
    assets["source_anchor_identity"] = paths.view(job.assets["source_anchor_identity"])
    moved_job = replace(job, assets=assets)
    payload = dict(
        loaded.payload,
        actor=first.actor.state_dict(),
        source_anchor_runtime_state=first.source_anchor.state_dict(),
    )
    second = assemble_from_full(payload, obs, moved_job)
    assert second.source_anchor.state_dict() == first.source_anchor.state_dict()
    expected = first.source_anchor.loss(first.actor, obs, torch.ones(4, dtype=torch.bool))
    actual = second.source_anchor.loss(second.actor, obs, torch.ones(4, dtype=torch.bool))
    assert torch.equal(expected, actual)
    assert second.source_anchor.state_dict() == first.source_anchor.state_dict()


@pytest.mark.parametrize("change", ["coefficient", "mask", "commands", "sha"])
def test_changed_retention_identity_rejected(tmp_path, change):
    job, _, obs = fixture(tmp_path)
    loaded = _resolve_source(job, None)
    config = copy.deepcopy(job.assets["source_policy_retention"])
    if change == "coefficient":
        config[change] = 0.03
    elif change == "mask":
        config[change] = "all DR frames"
    elif change == "commands":
        config[change] = [[0.0, 0.0]]
    else:
        config["bank"]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="身份不同|语义声明不一致"):
        assemble_from_full(
            loaded.payload,
            obs,
            replace(job, assets={**job.assets, "source_policy_retention": config}),
        )


@pytest.mark.parametrize(
    "change", ["final_instead_of_selected", "wrong_contract", "warmup", "refund"]
)
def test_branch_identity_and_non_refundable_count_guards(tmp_path, change):
    job, _, _ = fixture(tmp_path)
    if change == "final_instead_of_selected":
        final = tmp_path / "final.pt"
        job = replace(
            job,
            initialization=replace(job.initialization, checkpoint=final, sha256=file_sha256(final)),
        )
    elif change == "wrong_contract":
        job = replace(job, assets={**job.assets, "output_contract_sha256": "c" * 64})
    elif change == "warmup":
        job = replace(job, ppo=replace(job.ppo, critic_warmup_updates=51))
    else:
        job = replace(job, assets={**job.assets, "historical_online_updates": 5097})
    with pytest.raises(ValueError, match="不符合新分支初始化声明"):
        _resolve_source(job, None)


def test_legacy_initialization_still_rejects_older_model_without_explicit_branch(tmp_path):
    job, _, _ = fixture(tmp_path)
    assets = dict(job.assets)
    assets.pop("branch_initialization")
    no_branch = replace(job, assets=assets)
    with pytest.raises(ValueError, match="小于历史基数"):
        _check_source_progress(no_branch, {"online_updates_including_smoke": 5097}, None)


def test_declared_std_parameterization_cannot_change_sampling_distribution(tmp_path):
    job, _, obs = fixture(tmp_path)
    source = _resolve_source(job, None).payload
    bad = {**source, "std": source["std"], "std_parameterization": "log_std"}
    with pytest.raises(ValueError, match="实际采样分布不一致"):
        assemble_from_full(bad, obs, job)


def test_complete_branch_relocation_uses_no_original_paths(tmp_path):
    from native_stand.simulation.isaac.motion_backend import anchor_identity

    origin, moved = tmp_path / "origin", tmp_path / "cloud"
    origin.mkdir()
    job, _, obs = fixture(origin)
    raw = job.as_run_record()
    raw = {key: value for key, value in raw.items() if value is not None}
    for key in ("config_path", "config_sha256"):
        raw.pop(key)
    mapping = tmp_path / "mapping.json"
    write_json(mapping, dict(version="runtime_path_mapping_v1", prefixes={str(origin): str(moved)}))
    raw["assets"]["path_mapping"] = dict(path=str(mapping), sha256=file_sha256(mapping))
    origin.rename(moved)
    relocated_config = tmp_path / "relocated_job.json"
    write_json(relocated_config, raw)
    relocated = load_job_config(relocated_config)
    validate_job(relocated)
    loaded = _resolve_source(relocated, None)
    assert not origin.exists() and loaded.path.parent == moved
    identity = anchor_identity(relocated, loaded.payload)
    assert Path(identity["path"]).parent == moved
    restored = assemble_from_full(loaded.payload, obs, relocated)
    assert restored.actor_update_enabled
    assert restored.source_anchor.config["bank"]["path"] == str(moved / "bank.npz")


def test_command_bank_v4_enters_actual_masked_PPO_loss(tmp_path):
    from test_motion_production_integration import rollout

    job, _, obs = fixture(tmp_path)
    source = _resolve_source(job, None).payload
    algorithm = assemble_from_full(source, obs, job)
    frozen = copy.deepcopy(algorithm.source_anchor.teacher.state_dict())
    with torch.no_grad():
        algorithm.actor.mlp[-1].bias.add_(0.02)
    before_actor = copy.deepcopy(algorithm.actor.state_dict())
    before_critic = copy.deepcopy(algorithm.critic.state_dict())
    rollout(algorithm, obs, job.runtime.steps_per_env)
    summary = algorithm.mask_summary["source_policy_retention"]
    assert summary["version"] == "motion48_per_command_nominal_source_KL_v4"
    assert summary["weighted_loss_mean"] > 0
    assert summary["gradient_comparison"]["weighted_source_KL_gradient_norm"] > 0
    assert all(n > 0 for n in summary["actual_bank_selected_visits_by_cohort"])
    assert any(
        not torch.equal(value, algorithm.actor.state_dict()[key])
        for key, value in before_actor.items()
    )
    assert any(
        not torch.equal(value, algorithm.critic.state_dict()[key])
        for key, value in before_critic.items()
    )
    equal_tree(frozen, algorithm.source_anchor.teacher.state_dict())


def report(state, branch, *, unknown=False):
    strategy = resolve(state["version"])
    stage = strategy.stage_config(state["stage"], 2560, state=state)
    rows = []
    for key, purpose, axis, count in zip(
        stage["cohort_keys"],
        stage["cohort_purposes"],
        stage["maneuver_tasks"],
        stage["slot_counts"],
        strict=True,
    ):
        rows.append(
            dict(
                key=key,
                purpose=purpose,
                axis=axis,
                slot_count=count,
                ended_episodes=0 if unknown else 16,
                completed_episodes=0 if unknown else 16,
                pass_fraction=None if unknown else 1.0,
                quality_pass_fraction=None if unknown else 1.0,
                mean_dual_support_fraction=None if unknown else 1.0,
                physical_HF_RMS_rad_s=None if unknown else 0.01,
            )
        )
    return dict(
        version=RESPONSE_VERSION, stage_revision=state["stage"], through_update=100, rows=rows
    )


@pytest.mark.parametrize("branch", ["motion", "spin"])
def test_sampler_independent_frontiers_unknown_and_no_epoch_double_count(branch):
    strategy = resolve(course_configuration(branch)["version"])
    state = strategy.initial_state({"branch_course": course_configuration(branch)})
    stage = strategy.stage_config(0, 2560, state=state)
    assert sum(stage["slot_counts"]) == 2560 and len(stage["cohort_keys"]) <= 128
    assert len({tuple(key) for key in stage["cohort_keys"]}) == len(stage["cohort_keys"])
    assert all(p == 0 for p in stage["prefix_intervals"])
    if branch == "motion":
        assert all(abs(v) <= 3.6 and (v != 0 or abs(w) <= 2) for v, w in stage["commands"])
    else:
        assert all(v == 0 and np.isfinite(w) for v, w in stage["commands"])
    unknown, decision = strategy.advance(state, report(state, branch, unknown=True), 100, envs=2560)
    assert not decision["state_revision_changed"]
    assert not any(x["passed"] for x in decision["decisions"].values())
    first = report(state, branch)
    first["window"] = dict(
        run=branch, comparison_sha256=branch, start_update=1, end_update=100, stage_revision=0
    )
    one, _ = strategy.advance(state, first, 100, envs=2560)
    again, duplicated = strategy.advance(one, first, 100, envs=2560)
    assert again == one and duplicated["duplicate_window"]
    second = report(one, branch)
    second["window"] = dict(
        run=branch, comparison_sha256=branch, start_update=101, end_update=200, stage_revision=0
    )
    chosen = next(iter(one["levels"]))
    for row in second["rows"]:
        if row["axis"] != chosen and row["purpose"].startswith("focus_"):
            row["ended_episodes"] = row["completed_episodes"] = 0
            row["pass_fraction"] = None
    two, _ = strategy.advance(one, second, 200, envs=2560)
    assert two["levels"][chosen] == 1
    assert all(level == 0 for axis, level in two["levels"].items() if axis != chosen)
    assert two["general_DR_frontier"] == 2
    assert (
        two["assessment_keys"]
        == strategy.stage_config(two["stage"], 2560, state=two)["cohort_keys"]
    )


@pytest.mark.parametrize("branch", ["motion", "spin"])
def test_completed_episode_export_unknown_and_early_failure_denominator(branch):
    strategy = resolve(course_configuration(branch)["version"])
    state = strategy.initial_state({"branch_course": course_configuration(branch)})
    stage = strategy.stage_config(0, 2560, state=state)
    env = NS(
        num_envs=1,
        device="cpu",
        stage_spec=stage,
        student_actor_valid=torch.ones(1, dtype=torch.bool),
        command_index=torch.zeros(1, dtype=torch.long),
        rise_hold=NS(episode=NS(steps=torch.ones(1)), offset=torch.zeros(1)),
        motion_action_command=torch.tensor([[0.0, 0.0, 0.34]]),
        recovery_snapshot=torch.zeros(1, 31),
        standing_snapshot=torch.zeros(1, 6),
        antiosc_physical_stats_post=torch.zeros(1, 8),
        maneuver_goal_height=torch.full((1,), 0.34),
        maneuver_swing_difference=torch.zeros(1),
        motion_spin_measurement_post={"rolling_speed_forward_mps": torch.zeros(1, 2)},
    )
    episodes = episode_class(object, version=RESPONSE_VERSION)(env)
    assert all(row["pass_fraction"] is None for row in episodes.export(0)["rows"])
    episodes.observe_final((torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)))
    rows = episodes.export(100)["rows"]
    assert rows[0]["ended_episodes"] == 1 and rows[0]["completed_episodes"] == 0
    assert rows[0]["pass_fraction"] == 0 and rows[1]["pass_fraction"] is None


def test_motion_spin_candidate_domains_and_standing_eligibility_independent(tmp_path):
    motion = load_job_config(ROOT / "configs/jobs/motion48_u3000_prepared.json")
    spin = load_job_config(ROOT / "configs/jobs/spin48_u3000_prepared.json")
    assert candidate_strategy(motion) is candidate_strategy(spin) is split_candidates
    assert split_candidates.comparison_context(
        motion.as_run_record()
    ) != split_candidates.comparison_context(spin.as_run_record())
    state = resolve(motion.curriculum.version).initial_state(motion.assets)
    review = report(state, "motion")
    for row in review["rows"]:
        if row["purpose"].startswith("anchor_"):
            row["quality_pass_fraction"] = 0.0
            row["pass_fraction"] = 0.0
    evidence = split_candidates.summarize(review, DR_level=1, comparison_sha256="a" * 64)
    assert evidence["eligible_by_objective"] == {"standing": True, "motion": False}
    index = split_candidates.empty_index("a" * 64, version=split_candidates.VERSION)
    index = split_candidates.consider(
        index, evidence, dict(checkpoint=str(tmp_path / "saved.pt"), sha256="b" * 64), {}
    )
    assert index["best_candidate"] is None and index["best_standing_candidate"] is not None


def test_decoder_physical_conversion_and_out_of_range_labels():
    contract = read_json(ROOT / "tests/fixtures/motion_C0_control_contract.json")
    decoder = ActionDecoder.from_contract(contract)
    heights = torch.tensor([0.30, 0.34, 0.36, 0.40])
    means = torch.tensor([[0.2, -0.3, 0.4, -0.2, -0.3, -0.4]]).expand(4, -1)
    target = decoder.decode(means, heights)["requested_target"]
    converted = decoder.physical_to_mean(target, heights)
    assert converted["representable"].all() and torch.allclose(converted["mean"], means, atol=1e-6)
    target[:, 2] = 81.0
    with pytest.raises(ValueError, match="超出Student动作合同"):
        decoder.physical_to_mean(target, heights)
    assert not decoder.physical_to_mean(target, heights, require_representable=False)[
        "representable"
    ].any()
