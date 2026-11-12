"""Recovery continuation consumers; only CPU and temporary artifacts."""

import copy
import dataclasses
import shutil
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from native_stand.artifacts.checkpoint import load_source, save_full
from native_stand.artifacts.io import file_sha256, write_json
from native_stand.artifacts.model_refs import ModelIndex
from native_stand.experiments.config import load_job_config
from native_stand.simulation.driver import Backend
from native_stand.simulation.recovery_adapter import RecoveryAdapter
from native_stand.simulation.recovery_isaac.backend import SNAPSHOT, IsaacRecovery
from native_stand.tasks.recovery.phase_readback import PhaseReadback

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path):
    job = load_job_config(ROOT / "configs/jobs/recovery_preentry48_remaining.json")
    contract = tmp_path / "contract.json"
    write_json(
        contract,
        dict(
            action_version=job.action.version,
            center_rad=[0.0] * 4,
            leg_lower_rad=[-1.5] * 4,
            leg_upper_rad=[1.5] * 4,
        ),
    )
    job = dataclasses.replace(
        job,
        model=dataclasses.replace(job.model, actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        runtime=dataclasses.replace(job.runtime, envs=2, steps_per_env=2, device="cpu"),
        initialization=dataclasses.replace(
            job.initialization, mode="scratch", checkpoint=None, sha256=None
        ),
        assets={
            "recovery_training_contract": dict(path=str(contract), sha256=file_sha256(contract))
        },
    )
    observation = TensorDict(
        dict(
            student_actor=torch.zeros(2, 31),
            student_critic=torch.zeros(2, 909),
            recovery_actor_valid=torch.ones(2, 1, dtype=torch.bool),
        ),
        [2],
    )
    io = NS(
        env=NS(completed=[], completed_count=0),
        observations=lambda: observation,
        runtime_state=lambda: {},
        load_runtime_state=lambda state: None,
    )
    context = NS(
        job=job,
        run_dir=tmp_path / "run",
        updates_used=6,
        refs=ModelIndex(tmp_path / "refs.json"),
        store=NS(update=lambda **kw: None),
        total_updates_before=lambda stage: 6,
        restore_runtime_state=lambda state: None,
        curriculum_state=lambda: {},
    )
    context.run_dir.mkdir()
    adapter = RecoveryAdapter(Backend("CPU-I/O", lambda job: io))
    adapter.build(context, "formal", None)
    return adapter, context, io


@pytest.mark.parametrize("pending", [False, True])
def test_saved_candidates_republish_before_rotation_with_empty_project_index(tmp_path, pending):
    adapter, context, io = fixture(tmp_path)
    candidate = context.run_dir / "checkpoints/candidate.pt"
    save_full(candidate, adapter.capture(context, {"updates": 6}))
    ref = dict(
        checkpoint=str(candidate),
        sha256=file_sha256(candidate),
        update=6,
        objective="standing",
        evidence=dict(rank=[1.0, 1.0]),
    )
    adapter.candidates["candidates_by_objective"]["standing"]["known"] = ref
    adapter.candidates["best_standing_candidate"] = ref
    if pending:
        adapter.pending = dict(
            end_update=6, episode_cursor=0, comparison_sha256=adapter.domain, rows={}
        )
    cp = context.run_dir / "checkpoints/resume.pt"
    state = adapter.capture(context, {"updates": 6})
    state["run_identity"] = str(context.run_dir)
    save_full(cp, state)
    assert context.refs.candidate_index(context.run_dir.name) is None
    restored = RecoveryAdapter(Backend("CPU-I/O", lambda job: io))
    restored.build(context, "formal", load_source(cp))
    assert restored.candidates["best_standing_candidate"] == ref
    assert str(candidate) in context.refs.consumer_paths()
    assert restored.candidates["best_candidate"] is None
    assert restored.pending is None
    generation = context.refs._load()["generation"]
    third = RecoveryAdapter(Backend("CPU-I/O", lambda job: io))
    third.build(context, "formal", load_source(cp))
    assert context.refs._load()["generation"] == generation
    # Production's periodic+terminal same-update rewrite invalidated a nominee
    # SHA. The shared driver must keep the already published inode untouched.
    from native_stand.simulation.driver import _final

    writes = []
    _final(third, context, lambda *args: writes.append(args), context.updates_used)
    assert not writes and file_sha256(candidate) == ref["sha256"]


def test_unknown_and_early_fault_count_once_without_success(tmp_path):
    adapter, context, io = fixture(tmp_path)
    base = dict(
        group="full_Swing",
        nominal_anchor=True,
        command_height_m=0.34,
        height_dynamic=False,
        student_seconds=0.01,
        teacher_seconds=0.0,
        DR_strength=0.0,
        steps=1,
    )
    io.env.completed = [
        dict(base, success=False, physical_quality=False),
        dict(base, success=None, physical_quality=None),
    ]
    io.env.completed_count = 2
    adapter.after_update(context, 6, {})
    row = next(iter(adapter.window.values()))
    assert row["ended"] == 2 and row["student_success"] == 0 and row["quality"] == 0
    before = copy.deepcopy(adapter.window)
    adapter.after_update(context, 6, {})
    assert adapter.window == before


def test_empty_saved_pool_does_not_publish_a_fake_review(tmp_path):
    adapter, context, io = fixture(tmp_path)
    cp = context.run_dir / "checkpoints/resume.pt"
    state = adapter.capture(context, {"updates": 6})
    state["run_identity"] = str(context.run_dir)
    save_full(cp, state)
    restored = RecoveryAdapter(Backend("CPU-I/O", lambda job: io))
    restored.build(context, "formal", load_source(cp))
    assert context.refs.candidate_index(context.run_dir.name) is None
    restored.pending = dict(
        end_update=6, episode_cursor=0, rows={}, comparison_sha256=restored.domain
    )
    restored.after_save(context, dict(checkpoint=str(cp), sha256=file_sha256(cp), update=6))
    assert context.refs.candidate_index(context.run_dir.name)["last_review"]["through_update"] == 6


def test_imported_candidate_mapping_restores_into_empty_index(tmp_path):
    from native_stand.artifacts.handoff import VERSION, import_checkpoint

    adapter, context, io = fixture(tmp_path)
    original = context.run_dir / "checkpoints/original.pt"
    state = adapter.capture(context, {"updates": 6})
    state.update(run_identity=str(context.run_dir), config_contract_sha256="c" * 64)
    save_full(original, state)
    row = dict(
        checkpoint=str(original),
        sha256=file_sha256(original),
        update=6,
        objective="Swing",
        evidence=dict(rank=[1.0, 1.0]),
    )
    adapter.candidates["candidates_by_objective"]["Swing"]["nominal"] = row
    adapter.candidates["best_candidate"] = row
    adapter.candidates["last_review"] = dict(through_update=6, window={}, checkpoint=row)
    state["training_candidates"] = copy.deepcopy(adapter.candidates)
    cp = context.run_dir / "checkpoints/full.pt"
    save_full(cp, state)
    target = tmp_path / "relocated/checkpoints/import.pt"
    relocated = target.parent / "original.pt"
    relocated.parent.mkdir(parents=True)
    shutil.copyfile(original, relocated)
    manifest = tmp_path / "fixture-handoff.json"
    # Only a non-running /tmp fixture, never a declaration about the real owner.
    write_json(
        manifest,
        dict(
            version=VERSION,
            run_id="fixture",
            owner_terminated=True,
            checkpoint=dict(path=str(cp), sha256=file_sha256(cp)),
            original_run_identity=str(context.run_dir),
            dependencies=[],
            config_contract_sha256="c" * 64,
            budget=dict(updates_used=6, remaining=94, limits={"formal": 100}),
            parent_settlement={},
        ),
    )
    imported = import_checkpoint(
        manifest,
        destination=target,
        mapping={str(cp): str(cp), str(original): str(relocated)},
        expected_run_id="fixture",
    )
    context.run_dir = target.parent.parent
    write_json(context.run_dir / "run.json", {"logical_run_id": "fixture"})
    restored = RecoveryAdapter(Backend("CPU-I/O", lambda job: io))
    restored.build(context, "formal", imported)
    assert restored.candidates["best_candidate"]["checkpoint"] == str(relocated)
    assert str(relocated) in context.refs.consumer_paths()
    assert str(original) not in context.refs.consumer_paths()


def test_legacy_single_device_cuda_rng_normalized_at_real_adapter_boundary(tmp_path, monkeypatch):
    adapter, context, io = fixture(tmp_path)
    state = adapter.capture(context, {"updates": 6})
    state.pop("training_rng")
    state["CUDA_rng"] = torch.arange(16, dtype=torch.uint8)
    state["run_identity"] = str(context.run_dir)
    cp = context.run_dir / "checkpoints/legacy.pt"
    save_full(cp, state)
    received = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: received.extend(states))
    restored = RecoveryAdapter(Backend("CPU-I/O", lambda job: io))
    restored.build(context, "formal", load_source(cp))
    assert len(received) == 1 and torch.equal(received[0], state["CUDA_rng"])
    assert restored._prior_warmup == 5
    # Wrong mapping still fails, never silently drops a device stream.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="CUDA RNG映射"):
        restored.build(context, "formal", load_source(cp))


def test_phase_readback_exact_save_load_and_partial_reset(tmp_path):
    phase = PhaseReadback(2, "cpu")
    geometry = torch.tensor([[0.3, 0.3, 0.0, 0.0], [0.2, 0.2, 0.0, 0.0]])
    phase.reset(torch.arange(2), geometry, torch.tensor([0.18, 0.2]))
    phase.update(
        geometry + torch.tensor([-0.05, -0.06, 0.5, 0.6]),
        torch.tensor([0.34, 0.3]),
        torch.tensor([2.0, -1.0]),
        torch.tensor([1.0, 0.0]),
    )
    path = tmp_path / "phase.pt"
    torch.save(phase.state_dict(), path)
    restored = PhaseReadback(2, "cpu")
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.record(0) == phase.record(0)
    second = restored.record(1)
    restored.reset(torch.tensor([0]), geometry[:1], torch.tensor([0.18]))
    assert restored.record(0)["first_entry_s"] is None
    assert restored.record(1) == second


@pytest.mark.parametrize("same_run", [False, True])
def test_actual_fresh_reset_keeps_course_rng_discards_unfinished_evidence(same_run):
    from native_stand.tasks.recovery.swing_priority_course import RecoveryCourse

    saved = dict(
        version=SNAPSHOT,
        courses=[RecoveryCourse(7), RecoveryCourse(8)],
        DR_rng=[np.random.default_rng(i).bit_generator.state for i in range(2)],
        completed=[{"success": False}],
        completed_count=17,
    )
    for course in saved["courses"]:
        course.record("full_Swing", student_seconds=1.0, teacher_seconds=0.0, passed=False)
    env = NS(completed=[], completed_count=0, DR_rng=[np.random.default_rng(99) for _ in range(2)])
    calls = []

    def reset():
        calls.append(True)
        assert env.episode_spec == [None, None]
        return {"x": torch.zeros(2, 1)}, {}

    env.reset = reset
    wrapper = object.__new__(IsaacRecovery)
    wrapper.env, wrapper.num_envs, wrapper.device = env, 2, "cpu"
    wrapper.initialize_from_training_state({"execution": saved}, same_run=same_run)
    assert calls == [True]
    for a, b in zip(env.courses, saved["courses"], strict=True):
        assert a is not b and a.counts == b.counts and a.choose() == b.choose()
        assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert env.completed_count == (17 if same_run else 0)
    assert env.completed == (saved["completed"] if same_run else [])
    for rng, state in zip(env.DR_rng, saved["DR_rng"], strict=True):
        assert rng.bit_generator.state == state
