"""Actual aggregation, collection, lifecycle and physical-label regression consumers."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from test_public_migration import model_file, supervision_job

from native_stand.artifacts.io import file_sha256, read_json, write_json, write_npz
from native_stand.experiments.supervision import PhaseConfig
from native_stand.experiments.supervision_flow import execute

FROZEN_TEST_SCOPE = "PORTABLE"


def test_two_collections_accumulate_deduplicate_and_resume(tmp_path):
    job = supervision_job(tmp_path)
    replay = tmp_path / "replay.npz"
    write_npz(
        replay,
        dict(
            states=np.ones((100, 31), dtype="float32"),
            episodes=np.arange(100),
            roles=np.full(100, "train"),
        ),
    )
    replay_ref = dict(
        path=str(replay), sha256=file_sha256(replay), collection_id="old-physical-replay"
    )
    recipe = dataclasses.replace(
        job.supervision,
        dataset=replay_ref,
        rehearsal=(replay_ref, replay_ref),
        initial_student=dict(
            path=job.supervision.teachers[0].path, sha256=job.supervision.teachers[0].sha256
        ),
        phases=(
            PhaseConfig("collect", steps=1),
            PhaseConfig("label"),
            PhaseConfig("fit", rounds=1),
        ),
        repetitions=2,
    )
    job = dataclasses.replace(job, supervision=recipe)
    collected = []

    def scripted_io(job, source, spec, steps, directory, charge):
        number = len(collected) + 1
        path = directory / "data.npz"
        write_npz(
            path,
            dict(states=np.full((100, 31), 0.1 * number, dtype="float32"), episodes=np.arange(100)),
        )
        ref = dict(
            path=str(path), sha256=file_sha256(path), collection_id=f"physical-collection-{number}"
        )
        collected.append(ref)
        return ref

    rounds = []
    state, done = execute(
        job,
        tmp_path / "flow",
        collector=scripted_io,
        stop_requested=lambda: len(rounds) == 1,
        on_round_complete=rounds.append,
        on_round_saved=lambda r: None,
    )
    assert not done and len(state["collections"]) == 2
    state, done = execute(
        job,
        tmp_path / "flow",
        collector=scripted_io,
        stop_requested=lambda: False,
        on_round_complete=rounds.append,
        on_round_saved=lambda r: None,
    )
    assert done and len(collected) == 2 and len(state["collections"]) == 3
    with (
        np.load(tmp_path / "flow/phase_001_label/labels.npz") as a,
        np.load(tmp_path / "flow/phase_004_label/labels.npz") as b,
    ):
        assert len(a["states"]) == 200 and len(b["states"]) == 300
        roles = dict(zip(a["physical_episode_keys"], a["roles"], strict=True))
        assert all(
            roles.get(k, r) == r
            for k, r in zip(b["physical_episode_keys"], b["roles"], strict=True)
        )
        assert len(set(b["sample_ids"].tolist())) == 300
    assert len(read_json(state["aggregate"]["path"])["sources"]) == 3
    saved = torch.load(state["latest"]["path"], weights_only=False)
    assert saved["comparison_history"][-1]["incumbent_loss"] >= 0
    assert saved["comparison_history"][-1]["previous"]["dataset_sha256"] != saved["dataset_sha256"]


def test_rehearsal_cannot_override_persisted_holdout(tmp_path):
    from native_stand.artifacts.dataset import create_view, rehearsal_sources

    replay = tmp_path / "raw.npz"
    write_npz(replay, dict(states=np.zeros((20, 31)), episodes=np.arange(20)))
    ref = dict(path=str(replay), sha256=file_sha256(replay), collection_id="same")
    with pytest.raises(ValueError, match="persisted"):
        rehearsal_sources(dict(ref, role="train"))
    view = tmp_path / "view.json"
    record = create_view(view, [ref], seed=9)
    refs = rehearsal_sources(
        dict(path=str(view), sha256=file_sha256(view), kind="episode_dataset_view_v1", role="train")
    )
    new = create_view(tmp_path / "new.json", refs, seed=3)
    assert new["episode_roles"] == record["episode_roles"]
    assert "holdout" in new["episode_roles"].values()


class ScriptedCollectionIO:
    num_envs = 2

    def __init__(self, fault=False, missing=False):
        self.tick = 0
        self.closed = False
        self.fault = fault
        self.missing = missing

    def observations(self):
        return {"student_actor": torch.full((2, 31), self.tick / 100, dtype=torch.float32)}

    def step(self, action):
        self.tick += 1
        return dict(
            terminated=torch.tensor([[self.tick % 2 == 0], [False]]),
            truncated=torch.tensor([False, self.tick % 3 == 0]),
            extras=dict(
                actual_physics_steps=[2 if self.fault and self.tick == 2 else 10, 10],
                control_owner=torch.tensor([0, 1]),
                collection_conditions={} if self.missing else {"mode": torch.tensor([0, 1])},
            ),
        )

    def close(self):
        self.closed = True


def test_collection_actual_rows_partial_resets_owner_fault_and_stop(tmp_path, monkeypatch):
    from native_stand.simulation import collection

    job = supervision_job(tmp_path)
    job = dataclasses.replace(job, runtime=dataclasses.replace(job.runtime, envs=4))
    spec, path = model_file(tmp_path)
    source = dict(path=str(path), sha256=file_sha256(path))
    env = ScriptedCollectionIO(fault=True)
    monkeypatch.setattr(collection, "make_environment", lambda job: env)
    charged = []
    ref = collection.collect(
        job,
        source,
        spec.record(),
        8,
        tmp_path / "collect",
        charged.append,
        stop_requested=lambda: env.tick == 4,
        chunk_steps=2,
        required_conditions=("mode",),
    )
    assert env.closed and charged == [20, 12, 20, 20] and not ref["complete"]
    manifest = read_json(ref["path"])
    assert manifest["requested_envs"] == 4 and manifest["actual_envs"] == 2
    parts = [dict(np.load(r["path"])) for r in manifest["sources"]]
    assert np.concatenate([p["control_owner"] for p in parts]).tolist() == [0, 1] * 4
    assert np.concatenate([p["episodes"] for p in parts]).tolist() == [0, 0, 0, 0, 1, 0, 1, 1]
    assert not parts[0]["transition_valid"][2]
    old_keys = set(np.concatenate([p["physical_episode_keys"] for p in parts]))
    next_env = ScriptedCollectionIO()
    monkeypatch.setattr(collection, "make_environment", lambda job: next_env)
    completed = collection.collect(
        job,
        source,
        spec.record(),
        8,
        tmp_path / "collect",
        charged.append,
        chunk_steps=2,
        required_conditions=("mode",),
    )
    assert completed["complete"] and next_env.closed and sum(charged) == 152
    data = read_json(completed["path"])
    assert len(data["sources"]) == 4
    new = dict(np.load(data["sources"][-1]["path"]))
    assert not old_keys.intersection(new["physical_episode_keys"])


def test_collection_missing_metadata_charges_and_closes(tmp_path, monkeypatch):
    from native_stand.simulation import collection

    job = supervision_job(tmp_path)
    spec, path = model_file(tmp_path)
    env = ScriptedCollectionIO(missing=True)
    monkeypatch.setattr(collection, "make_environment", lambda job: env)
    billed = []
    with pytest.raises(ValueError, match="routing"):
        collection.collect(
            job,
            dict(path=str(path), sha256=file_sha256(path)),
            spec.record(),
            2,
            tmp_path / "collect",
            billed.append,
            required_conditions=("mode",),
        )
    assert env.closed and billed == [20]
    assert read_json(tmp_path / "collect/collection_state.json")["physical_steps"] == 20


def test_recovery_teacher_student_physical_space_is_zero_error(tmp_path):
    from native_stand.experiments.supervision import TeacherConfig
    from native_stand.learning.inference import InferenceSpec, build_actor

    job = supervision_job(tmp_path)
    contract = dict(
        action_version="recovery_cubic_physical_Gaussian_v2",
        center_rad=[0.0] * 4,
        leg_lower_rad=[-1.2] * 4,
        leg_upper_rad=[1.2] * 4,
    )
    spec = InferenceSpec("recovery_physical", 31, (8,), 6, {"version": "physical31"}, contract)
    actor = build_actor(spec)
    with torch.no_grad():
        actor.mlp[-1].weight.zero_()
        actor.mlp[-1].bias.zero_()
        actor.mlp[-1].bias[[2, 5]] = 0.5493061443340548
    source = tmp_path / "recovery.pt"
    torch.save(
        dict(actor=actor.state_dict(), output_contract=contract, inference_spec=spec.record()),
        source,
    )
    teacher = TeacherConfig(
        "recovery", str(source), file_sha256(source), spec.record(), {"kind": "physical"}
    )
    recipe = dataclasses.replace(
        job.supervision,
        teachers=(teacher,),
        student=spec.record(),
        student_labels={"kind": "physical"},
        initial_student=dict(path=str(source), sha256=file_sha256(source)),
        phases=(PhaseConfig("label"), PhaseConfig("fit", rounds=1)),
    )
    job = dataclasses.replace(job, supervision=recipe)
    state, done = execute(
        job,
        tmp_path / "physical",
        stop_requested=lambda: False,
        on_round_complete=lambda r: None,
        on_round_saved=lambda r: None,
    )
    checkpoint = torch.load(state["latest"]["path"], weights_only=False)
    assert done and checkpoint["history"][0]["train_loss"] < 1e-10
    with np.load(tmp_path / "physical/phase_000_label/labels.npz") as data:
        np.testing.assert_allclose(data["teacher_actions"][:, [2, 5]], 50.0, atol=1e-5)
        assert json.loads(str(data["label_provenance"]))["label_space"] == {"kind": "physical"}


@pytest.mark.parametrize(
    "operation,steps,rounds", [("label", 0, 1), ("export", 1, 0), ("fit", 1, 1), ("collect", 1, 1)]
)
def test_phase_counts_only_apply_to_executed_operation(operation, steps, rounds):
    with pytest.raises(ValueError):
        PhaseConfig(operation, steps=steps, rounds=rounds)


def _wait_fixture_terminal(run):
    import time

    for _ in range(200):
        state = read_json(run / "state.json") or {}
        if state.get("status") in ("COMPLETED", "FAILED", "STOPPED_BOUNDED"):
            return state
        time.sleep(0.05)
    raise AssertionError("bounded fixture worker did not finish")


@pytest.mark.parametrize("bound", [1, 3])
def test_formal_cli_worker_stop_resume_and_zero_round_postprocessing(tmp_path, capsys, bound):
    from native_stand.cli import main

    job = supervision_job(tmp_path)
    raw = job.as_run_record()
    for key in ("config_path", "config_sha256", "preset"):
        raw.pop(key)
    config = tmp_path / "job.json"
    write_json(config, raw)
    assert (
        main(
            [
                "distill",
                "--config",
                str(config),
                "--runs-root",
                str(tmp_path / "runs"),
                "--run-id",
                "fixture",
                "--max-rounds",
                str(bound),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    run = tmp_path / "runs/fixture"
    first = _wait_fixture_terminal(run)
    assert first["status"] == "STOPPED_BOUNDED", read_json(run / "worker_failure.json")
    assert read_json(run / "budget.json")["used"]["distill"] == bound
    assert main(["resume", "--run", str(run), "--json"]) == 0
    final = _wait_fixture_terminal(run)
    assert final["status"] == "COMPLETED", read_json(run / "worker_failure.json")
    budget = read_json(run / "budget.json")
    assert budget["used"] == {"smoke": 0, "formal": 0, "distill": 3}
    assert len(budget["attempts"]) == 2
    assert (run / "supervision/phase_003_export/package/manifest.json").is_file()


def test_package_published_before_phase_commit_can_resume(tmp_path, monkeypatch):
    import native_stand.experiments.supervision_flow as flow

    job = supervision_job(tmp_path)
    original = flow.write_json

    def fail_commit(path, value):
        if Path(path).name == "supervision_state.json" and value.get("phase_cursor") == 4:
            raise OSError("injected phase commit crash after package publication")
        return original(path, value)

    monkeypatch.setattr(flow, "write_json", fail_commit)
    with pytest.raises(OSError, match="phase commit"):
        execute(
            job,
            tmp_path / "flow",
            stop_requested=lambda: False,
            on_round_complete=lambda r: None,
            on_round_saved=lambda r: None,
        )
    monkeypatch.setattr(flow, "write_json", original)
    state, done = execute(
        job,
        tmp_path / "flow",
        stop_requested=lambda: False,
        on_round_complete=lambda r: pytest.fail("fit repeated"),
        on_round_saved=lambda r: None,
    )
    assert done and state["rounds"] == 3


def test_scene_config_reaches_material_query_and_reset_validation(tmp_path):
    from native_stand.simulation.scenes import (
        FlatHeight,
        SceneConfig,
        initial_catalog,
        validate_reset_pose,
    )

    config = SceneConfig(friction=0.7, ground_z_m=0.1)
    root = torch.tensor([[0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]])
    validate_reset_pose(root, torch.zeros(1, 6), torch.zeros(1, 6), FlatHeight(config))
    bad = root.clone()
    bad[:, 2] = 0.05
    with pytest.raises(ValueError, match="below"):
        validate_reset_pose(bad, torch.zeros(1, 6), torch.zeros(1, 6), FlatHeight(config))
    catalog = {"mechanics": {"id": "same"}, "poses": {"Swing": {"identity": "legal-frozen"}}}
    path = tmp_path / "catalog.json"
    write_json(path, catalog)
    configured = dataclasses.replace(
        config, initial_state=dict(path=str(path), sha256=file_sha256(path))
    )
    assert initial_catalog(configured, catalog) == catalog
    assert configured.friction == 0.7
    wrong = dict(catalog, mechanics={"id": "different"})
    write_json(path, wrong)
    with pytest.raises(ValueError):
        initial_catalog(configured, catalog)


def test_failed_fit_charge_is_not_refunded_or_silently_replayed(tmp_path, monkeypatch):
    import native_stand.learning.distillation.loop as loop
    from native_stand.experiments import owner, worker
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.distillation_job import _precheck_resume, run_distillation_worker
    from native_stand.experiments.state import RunStateStore

    job = supervision_job(tmp_path)
    run = owner.create_run(job, runs_root=tmp_path / "runs", run_id="failure")
    Budget.create(run / "budget.json", smoke_updates=0, formal_updates=0, distill_rounds=3)
    RunStateStore(run).create(
        run_id="failure", owner={"pid": 0, "role": "fixture"}, stage="distill"
    )
    monkeypatch.setattr(worker, "STOP_REQUESTED", False)

    def fail_save(*a, **kw):
        raise OSError("injected after optimization and charge")

    monkeypatch.setattr(loop, "_atomic_save_latest", fail_save)
    assert run_distillation_worker(run, job) == 1
    assert read_json(run / "budget.json")["used"]["distill"] == 1
    with pytest.raises(ValueError, match="latest|恢复点"):
        _precheck_resume(run)


def test_saved_fit_before_publication_is_repaired_without_new_rounds(tmp_path, monkeypatch):
    from native_stand.artifacts.model_refs import ModelIndex
    from native_stand.experiments import owner, worker
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.distillation_job import _precheck_resume, run_distillation_worker
    from native_stand.experiments.state import RunStateStore

    job = supervision_job(tmp_path)
    run = owner.create_run(job, runs_root=tmp_path / "runs", run_id="saved")
    Budget.create(run / "budget.json", smoke_updates=0, formal_updates=0, distill_rounds=3)
    RunStateStore(run).create(run_id="saved", owner={"pid": 0, "role": "fixture"}, stage="distill")
    monkeypatch.setattr(worker, "STOP_REQUESTED", False)
    original = ModelIndex.register_output

    def fail_publish(*a, **kw):
        raise OSError("injected after fit saved before reference publication")

    monkeypatch.setattr(ModelIndex, "register_output", fail_publish)
    assert run_distillation_worker(run, job) == 1
    monkeypatch.setattr(ModelIndex, "register_output", original)
    _, _, remaining, latest = _precheck_resume(run)
    assert remaining == 2 and latest["round"] == 1
    assert run_distillation_worker(run, job) == 0
    assert read_json(run / "budget.json")["used"]["distill"] == 3
    assert read_json(run / "state.json")["status"] == "COMPLETED"


def test_semantic_identity_ignores_only_identified_reference_paths(tmp_path):
    from native_stand.experiments.supervision_flow import semantic_identity

    job = supervision_job(tmp_path)
    recipe = job.supervision
    teacher = dataclasses.replace(recipe.teachers[0], path="/relocated/same-content.pt")
    assert semantic_identity(dataclasses.replace(recipe, teachers=(teacher,))) == semantic_identity(
        recipe
    )
    changed = dataclasses.replace(teacher, conditions={"mask": 1})
    assert semantic_identity(dataclasses.replace(recipe, teachers=(changed,))) != semantic_identity(
        recipe
    )
    # Arbitrary semantic keys named path are not blanket-excluded.
    first = dataclasses.replace(
        recipe, student={**recipe.student, "input_contract": {"path": "semantic-A"}}
    )
    second = dataclasses.replace(
        recipe, student={**recipe.student, "input_contract": {"path": "semantic-B"}}
    )
    assert semantic_identity(first) != semantic_identity(second)


def test_actual_recovery_player_forward_is_not_cubic_decoded_twice(tmp_path):
    from collections import deque

    from native_stand.learning.inference import InferenceSpec, build_actor
    from native_stand.simulation.control_core import ControlCore, build
    from native_stand.simulation.mujoco_playback import MotionPlayer
    from native_stand.simulation.playback import Controls
    from native_stand.simulation.playback_execution import PlaybackExecution
    from native_stand.simulation.recovery_isaac.execution import nominal_timing

    contract = dict(
        action_version="recovery_cubic_physical_Gaussian_v2",
        center_rad=[0.0] * 4,
        leg_lower_rad=[-1.2] * 4,
        leg_upper_rad=[1.2] * 4,
    )
    spec = InferenceSpec("recovery_physical", 31, (8,), 6, {"version": "physical31"}, contract)
    actor = build_actor(spec)
    with torch.no_grad():
        actor.mlp[-1].weight.zero_()
        actor.mlp[-1].bias.zero_()
        actor.mlp[-1].bias[[2, 5]] = 0.5493061443340548
    job = supervision_job(tmp_path)
    control = dataclasses.replace(job.control, effort_limits_nm=(40, 40, 4.8, 40, 40, 4.8))
    names = [f"p{i}" for i in range(13)]
    params = [1.0, 0.1, 0.2, 4.0, 6.0, 1.0, -1.0, 2.0, 3.0, 0.4, 0.0, 0.0, 0.0]
    row = dict(
        parameters=dict(zip(names, params, strict=False)),
        effective_observer=dict(
            transport_ms=8, q_age_ms=2, v_age_ms=4, q_reference=[0.0, 0.0], velocity_tau_ms=3.0
        ),
    )
    core = ControlCore(build(tmp_path / "control.so"), contract)
    execution = PlaybackExecution(
        core,
        dict(parameter_names=names, sides={"L": row, "R": row}),
        nominal_timing(),
        control,
        [-1.5, -1.5, -100, -1.5, -1.5, -100],
        [1.5, 1.5, 100, 1.5, 1.5, 100],
    )
    execution.reset(np.zeros(6), np.zeros(6))
    plant = MotionPlayer.__new__(MotionPlayer)
    plant.diagnostic_target_transform = None
    plant.terrain = None
    plant.recording = True
    plant.plan = NS(job=NS(task="recovery", control=control))
    plant.actor = actor
    plant.actor_spec = spec
    plant.execution = execution
    plant.robot_body = 0
    plant.qids = np.arange(6)
    plant.vids = np.arange(6)
    plant.policy_ticks = 0
    plant.actual_physics_steps = 0
    plant.records = deque()
    plant.data = NS(
        xmat=np.eye(3).reshape(1, 9),
        cvel=np.zeros((1, 6)),
        qpos=np.zeros(6),
        qvel=np.zeros(6),
        ctrl=np.zeros(6),
        qfrc_applied=np.zeros(6),
    )
    plant.model = NS(neq=0)
    # Sole physical I/O boundary, no real MuJoCo import or integration.
    plant.mj = NS(mj_step=lambda *a: None, mj_forward=lambda *a: None)
    plant.step(Controls())
    np.testing.assert_allclose(plant.last_record["requested"][[2, 5]], 50.0, atol=1e-5)


def test_formal_bc_handoff_moves_all_dependencies_then_resumes(tmp_path, capsys):
    import shutil

    from native_stand.cli import main

    original = tmp_path / "original"
    original.mkdir()
    job = supervision_job(original)
    raw = job.as_run_record()
    for key in ("config_path", "config_sha256", "preset"):
        raw.pop(key)
    config = original / "job.json"
    write_json(config, raw)
    assert (
        main(
            [
                "distill",
                "--config",
                str(config),
                "--runs-root",
                str(original / "runs"),
                "--run-id",
                "handoff",
                "--max-rounds",
                "1",
                "--json",
            ]
        )
        == 0
    )
    run = original / "runs/handoff"
    assert _wait_fixture_terminal(run)["status"] == "STOPPED_BOUNDED"
    source_record = read_json(run / "run.json")
    logical = source_record["logical_run_id"]
    source_latest = read_json(run / "supervision/supervision_state.json")["latest"]
    # The entire old root disappears, including teachers and data, not only the checkpoint.
    moved = tmp_path / "persistent"
    original.rename(moved)
    target = moved / "runs/handoff"
    files = {str(original / p.relative_to(moved)): str(p) for p in moved.rglob("*") if p.is_file()}
    evidence = moved / "settled"
    evidence.mkdir()
    refs = {}
    for name, file in [
        ("source_run", "run.json"),
        ("source_state", "state.json"),
        ("source_budget", "budget.json"),
    ]:
        old = str(run / file)
        saved = evidence / file
        shutil.copyfile(target / file, saved)
        files[old] = str(saved)
        refs[name] = dict(path=old, sha256=file_sha256(saved))
    source_copy = evidence / "source_checkpoint.pt"
    destination = Path(files[source_latest["path"]])
    shutil.copyfile(destination, source_copy)
    destination.unlink()  # /tmp fixture only; original bytes remain in settled evidence.
    files[source_latest["path"]] = str(source_copy)
    path_map = moved / "runtime_paths.json"
    write_json(
        path_map, dict(version="runtime_path_mapping_v1", prefixes={str(original): str(moved)})
    )
    record = read_json(target / "run.json")
    record["config"]["assets"]["path_mapping"] = dict(
        path=str(path_map), sha256=file_sha256(path_map)
    )
    record["model_index"] = str(moved / "runs/models.json")
    write_json(target / "run.json", record)
    flow_path = target / "supervision/supervision_state.json"
    dependencies = [
        dict(path=old, sha256=file_sha256(new))
        for old, new in files.items()
        if Path(new).is_file()
        and old not in (str(run / "run.json"), str(run / "state.json"), str(run / "budget.json"))
    ]
    book = read_json(target / "budget.json")
    manifest = moved / "handoff.json"
    write_json(
        manifest,
        dict(
            version="checkpoint_directory_handoff_v1",
            model_type="bc_full",
            run_id=logical,
            target_run=str(target),
            owner_terminated=True,
            checkpoint=source_latest,
            dependencies=dependencies,
            budget=dict(rounds_used=1, remaining=2, limits=book["limits"]),
            **refs,
        ),
    )
    mapping = moved / "mapping.json"
    write_json(mapping, files)
    assert (
        main(
            [
                "import-checkpoint",
                "--manifest",
                str(manifest),
                "--mapping",
                str(mapping),
                "--destination",
                str(destination),
                "--run-id",
                logical,
                "--json",
            ]
        )
        == 0
    )
    assert not original.exists()
    assert read_json(flow_path)["latest"]["sha256"] == file_sha256(destination)
    assert main(["resume", "--run", str(target), "--json"]) == 0
    terminal = _wait_fixture_terminal(target)
    assert terminal["status"] == "COMPLETED", read_json(target / "worker_failure.json")
    assert read_json(target / "budget.json")["used"]["distill"] == 3
    assert (target / "supervision/phase_003_export/package/manifest.json").exists()
