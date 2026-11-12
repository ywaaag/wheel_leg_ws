"""Final public consumers with scripted physical I/O; no integration or real run."""

import copy

import numpy as np
import pytest
import torch
from test_phase3_fixes import _cpu_job, _production_context, _sentinel_full_state
from test_public_migration import model_file

from native_stand.artifacts.io import file_sha256, read_json, write_json
from native_stand.simulation import driver, sim2sim

FROZEN_TEST_SCOPE = "PORTABLE"


def test_diagnostic_flat_scene_readback_rejects_material_drift():
    from types import SimpleNamespace

    from native_stand.simulation.scenes import SceneConfig

    plant = SimpleNamespace(scene_config=SceneConfig())
    plane = dict(
        collider="plane",
        height_m=0.0,
        normal_axis="Z",
        width_m=128.0,
        length_m=128.0,
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.0,
        material_path="/World/material",
    )
    assert sim2sim._scene_readback(plant, plane) == plane
    with pytest.raises(ValueError, match="scene differs"):
        sim2sim._scene_readback(plant, {**plane, "dynamic_friction": 0.6})


def test_live_plan_rejects_deferred_application_close_before_engine(tmp_path):
    config = protocol(tmp_path)
    path = config["backends"][0]["config"]
    raw = read_json(path)
    raw.setdefault("assets", {})["application_lifecycle"] = dict(owner="worker_atexit")
    write_json(path, raw)
    with pytest.raises(ValueError, match="synchronous engine close"):
        sim2sim.live_plan(config)


def test_live_raised_short_fault_keeps_consumption_and_primary_error(tmp_path, monkeypatch):
    events = []

    class IO:
        physics_dt = 0.001

        def __init__(self, endpoint, config, directory):
            self.actual = {k: copy.deepcopy(config[k]) for k in sim2sim.REQUIRED[1:-1]}
            self.actual.update(
                {k: endpoint[k] for k in ("engine", "integration", "contact_sampling")}
            )
            events.append("open")

        def observations(self):
            return torch.zeros(1, 31)

        def step(self, mean):
            error = RuntimeError("engine after2ms")
            error.actual_physics_steps = [2]
            raise error

        def close(self):
            events.append("close")
            raise ValueError("cleanup")

    monkeypatch.setattr(sim2sim, "make_diagnostic_session", IO)
    with pytest.raises(RuntimeError, match="after2ms") as failure:
        sim2sim.run_live(protocol(tmp_path), tmp_path / "result.json")
    report = read_json(tmp_path / "result_traces/backend_0_failure.json")
    assert report["actual_physics_steps"] == 2 and report["samples"] == 0
    assert events == ["open", "close"]
    assert any("cleanup" in note for note in failure.value.__notes__)


@pytest.mark.parametrize("missing_route", [False, True])
def test_motion_production_collection_through_aggregate_and_actual_bc(
    tmp_path, monkeypatch, missing_route
):
    from dataclasses import replace

    from test_motion_collection import install

    from native_stand.artifacts.checkpoint import save_full
    from native_stand.artifacts.io import write_npz
    from native_stand.experiments.supervision import PhaseConfig, SupervisionConfig, TeacherConfig
    from native_stand.experiments.supervision_flow import execute
    from native_stand.learning.inference import spec_for_job

    job, source, created = install(tmp_path, monkeypatch)
    saved = tmp_path / "initial.pt"
    save_full(saved, source)
    ref = dict(path=str(saved), sha256=file_sha256(saved))
    spec = spec_for_job(job, source["output_contract"])
    data = tmp_path / "base.npz"
    states = np.zeros((100, 31), dtype=np.float32)
    states[:, 24] = 0.68
    base = dict(
        states=states,
        episodes=np.arange(100),
        condition_control_owner=np.zeros(100, dtype=np.int64),
    )
    if missing_route:
        base.pop("condition_control_owner")
    write_npz(data, base)
    labels = dict(kind="motion_contract", contract=source["output_contract"])
    recipe = SupervisionConfig(
        teachers=tuple(
            TeacherConfig(
                name,
                **ref,
                inference=spec.record(),
                labels=labels,
                conditions={"control_owner": owner},
            )
            for name, owner in [("student", 0), ("prefix", 1)]
        ),
        student=spec.record(),
        student_labels=labels,
        dataset=dict(path=str(data), sha256=file_sha256(data)),
        initial_student=ref,
        phases=(
            PhaseConfig("collect", steps=2),
            PhaseConfig("label"),
            PhaseConfig("fit", rounds=1),
        ),
        repetitions=2,
        batch_size=16,
    )
    charged, rounds = [], []
    if missing_route:
        with pytest.raises(ValueError, match="routing conditions are incomplete"):
            execute(
                replace(job, algorithm="distillation", supervision=recipe),
                tmp_path / "flow",
                stop_requested=lambda: False,
                on_round_complete=rounds.append,
                on_round_saved=lambda *args: None,
                on_physical_steps=charged.append,
            )
        assert rounds == [] and sum(charged) == 60 and created.count("closed") == 1
        return
    state, done = execute(
        replace(job, algorithm="distillation", supervision=recipe),
        tmp_path / "flow",
        stop_requested=lambda: False,
        on_round_complete=rounds.append,
        on_round_saved=lambda *args: None,
        on_physical_steps=charged.append,
    )
    assert done and state["rounds"] == 2 and len(state["collections"]) == 3
    assert sum(charged) == 120 and created.count("closed") == 2
    with np.load(state["dataset"]["path"]) as data:
        import json

        assert set(data["teacher_owner"].tolist()) == {-1, 0, 1}
        provenance = json.loads(str(data["label_provenance"]))
        assert [t["name"] for t in provenance["teachers"]] == ["student", "prefix"]


def test_collection_exception_preserves_measured_short_fault_consumption(tmp_path, monkeypatch):
    from native_stand.simulation import collection

    job = _cpu_job(tmp_path)
    spec, model = model_file(tmp_path)
    charged, closed = [], []

    class IO:
        num_envs = 2

        def observations(self):
            return {"student_actor": torch.zeros(2, 31)}

        def step(self, mean):
            error = RuntimeError("fault after 2ms")
            error.actual_physics_steps = [2, 2]
            raise error

        def close(self):
            closed.append(True)

    monkeypatch.setattr(collection, "make_environment", lambda job: IO())
    with pytest.raises(RuntimeError, match="2ms"):
        collection.collect(
            job,
            {"path": str(model), "sha256": file_sha256(model)},
            spec.record(),
            3,
            tmp_path / "collection",
            charged.append,
        )
    state = read_json(tmp_path / "collection/collection_state.json")
    assert charged == [4] and closed == [True]
    assert state["physical_steps"] == 4 and state["policy_steps"] == 0
    assert state["status"] == "ENGINEERING_FAULT"
    assert not state["failed_step_consumption"]["transition_committed"]


def test_resume_stopped_before_first_update_keeps_existing_checkpoint(tmp_path):
    job = _cpu_job(tmp_path, limits={"smoke_updates": 3, "formal_updates": 0})
    ctx = _production_context(tmp_path, job, smoke=3)
    ctx.charge(updates=1)
    payload = _sentinel_full_state(job, sentinel=0.1)
    payload["counters"]["updates"] = 1
    reference = ctx.save(payload, update=1, actor_changed=False, stage="smoke")
    ctx.stop_requested = lambda: True

    class Adapter:
        def build(self, *args):
            pass

        def capture(self, *args):
            raise AssertionError("Stopped resume must not recapture an existing update")

        def close(self):
            pass

    result = driver.run_stage(ctx, Adapter(), stage="smoke")
    assert result.stopped and result.updates == 0
    assert ctx.updates_used == 1
    assert file_sha256(reference["checkpoint"]) == reference["sha256"]


def test_playback_diagnostic_reads_last_actual_force_not_previous_policy_record():
    from types import SimpleNamespace

    data = SimpleNamespace(time=0.0, qpos=np.zeros(13), qvel=np.zeros(12))
    execution = SimpleNamespace(motor=np.zeros(6), NET=np.zeros(6))
    record = dict(
        motor=np.full(6, -99), NET=np.full(6, -99), observation=np.zeros(31), raw=np.zeros(6)
    )

    def step(controls):
        data.time += 0.001
        execution.motor += 1
        execution.NET += 2

    diagnostic = sim2sim.PlaybackDiagnostic.__new__(sim2sim.PlaybackDiagnostic)
    diagnostic.plant = SimpleNamespace(
        data=data,
        execution=execution,
        step=step,
        records=[record],
        qids=np.arange(6),
        vids=np.arange(6),
    )
    diagnostic.controls = None
    diagnostic.physics_dt = 0.001
    diagnostic.policy_substeps = 2
    diagnostic.np = np
    result = diagnostic.step(None)
    assert result["extras"]["actual_physics_steps"] == [2]
    np.testing.assert_array_equal(result["extras"]["diagnostic"]["motor"], np.full((1, 6), 2))
    np.testing.assert_array_equal(result["extras"]["diagnostic"]["NET"], np.full((1, 6), 4))


def test_deployment_nested_runtime_dependencies_move_without_original_root(tmp_path):
    from types import SimpleNamespace

    from native_stand.artifacts.checkpoint import load_actor_source
    from native_stand.artifacts.deployment import export_package, job_dependencies, verify_package

    old = tmp_path / "old"
    old.mkdir()
    leaf = old / "scene.json"
    write_json(leaf, {"physical_identity": "frozen"})
    runtime = old / "runtime.json"
    write_json(
        runtime, {"bindings": {"scene_spec": {"path": str(leaf), "sha256": file_sha256(leaf)}}}
    )
    moved = tmp_path / "relocated"
    old.rename(moved)
    mapping = tmp_path / "mapping.json"
    write_json(
        mapping,
        dict(
            version="runtime_path_mapping_v1",
            files={str(old / name): str(moved / name) for name in ("runtime.json", "scene.json")},
        ),
    )
    job = SimpleNamespace(
        task="recovery",
        assets={
            "path_mapping": {"path": str(mapping), "sha256": file_sha256(mapping)},
            "recovery_isaac_runtime": {
                "path": str(runtime),
                "sha256": file_sha256(moved / "runtime.json"),
            },
        },
    )
    refs = job_dependencies(job)
    assert next(r for r in refs if r["role"] == "scene_spec")["path"] == str(moved / "scene.json")
    spec, source = model_file(tmp_path)
    package = tmp_path / "deployment"
    export_package(
        load_actor_source(source), spec, package, execution={"wheel_cap_nm": 4.8}, dependencies=refs
    )
    assert verify_package(package)["files"]["golden.npz"]


@pytest.mark.parametrize(
    "fault,expected_saved,consumed",
    [("rollout", 1, 2), ("postprocess", 1, 2), ("metrics", 2, 2), ("save", 1, 2)],
)
def test_fault_checkpoint_boundary_does_not_publish_partial_state(
    tmp_path, fault, expected_saved, consumed
):
    job = _cpu_job(
        tmp_path,
        limits={"smoke_updates": 3, "formal_updates": 0},
        storage={"checkpoint_every_updates": 100},
    )
    ctx = _production_context(tmp_path, job, smoke=3)
    events = []
    old_record = ctx.record

    def record(value):
        if fault == "metrics" and ctx.updates_used == 2:
            raise RuntimeError("metrics fault")
        old_record(value)

    ctx.record = record
    old_save = ctx.save

    def save(*args, **kwargs):
        if fault == "save" and ctx.updates_used == 2:
            raise OSError("save fault")
        return old_save(*args, **kwargs)

    ctx.save = save
    ctx.save_due = lambda update, **kw: fault == "save" and update == 2

    class Adapter:
        def build(self, *args):
            pass

        def update_once(self, context, index):
            if fault == "rollout" and index == 3:
                raise RuntimeError("rollout fault after unknown partial work")
            return dict(reward_mean=1.0, actor_steps=1)

        def after_update(self, context, index, outcome):
            if fault == "postprocess" and index == 2:
                raise RuntimeError("postprocess fault")

        def capture(self, context, counters):
            events.append(("capture", context.updates_used))
            value = _sentinel_full_state(job, sentinel=0.1)
            value["counters"] = counters
            return value

        def after_save(self, context, reference):
            events.append(("publish", context.updates_used))

        def close(self):
            events.append(("close", None))

    with pytest.raises((RuntimeError, OSError)):
        driver.run_stage(ctx, Adapter(), stage="smoke")
    assert ctx.updates_used == consumed
    assert ctx.latest_reference()["update"] == expected_saved
    assert events.count(("close", None)) == 1
    assert ("publish", 2) in events if expected_saved == 2 else ("publish", 2) not in events
    assert events.count(("capture", 2)) <= 1  # save error is never retried


def test_primary_engine_error_survives_cleanup_error(tmp_path):
    job = _cpu_job(tmp_path, limits={"smoke_updates": 1, "formal_updates": 0})
    ctx = _production_context(tmp_path, job)

    class Adapter:
        def build(self, *args):
            raise ValueError("original engine error")

        def close(self):
            raise RuntimeError("cleanup error")

    with pytest.raises(ValueError, match="original engine error") as failure:
        driver.run_stage(ctx, Adapter(), stage="smoke")
    assert any("cleanup error" in note for note in failure.value.__notes__)
    assert ctx.updates_used == 0


def protocol(tmp_path):
    spec, path = model_file(tmp_path)
    config_path = tmp_path / "job.json"
    _cpu_job(tmp_path, runtime={"envs": 1})
    return dict(
        version="sim2sim_live_protocol_v1",
        model=dict(path=str(path), sha256=file_sha256(path), inference_spec=spec.record()),
        input_contract=spec.input_contract,
        output_contract=spec.output_contract,
        scene={"flat": True},
        command=[0.0, 0.0, 0.34],
        initial_state={"legal": "fixture"},
        timing={"physics_ms": 1, "policy_ms": 10},
        V4={"source": "declared"},
        motor_caps=[40.0, 40.0, 4.8, 40.0, 40.0, 4.8],
        scoring_window_s=[0.01, 0.03],
        steps=3,
        backends=[
            dict(
                adapter="collection",
                config=str(config_path),
                engine=name,
                integration={"dt": 0.001},
                contact_sampling={"kind": name},
            )
            for name in ("PhysX", "MuJoCo")
        ],
    )


def test_live_sim2sim_formal_cli_sequential_collection_to_diagnostics(
    tmp_path, monkeypatch, capsys
):
    from native_stand.cli import main

    config = protocol(tmp_path)
    events = []

    class IO:
        physics_dt = 0.001

        def __init__(self, endpoint, protocol, directory):
            assert not events or events[-1].startswith("close")
            self.name = endpoint["engine"]
            self.tick = 0
            events.append("open " + self.name)
            self.actual = {key: copy.deepcopy(protocol[key]) for key in sim2sim.REQUIRED[1:-1]}
            self.actual.update(
                {k: endpoint[k] for k in ("engine", "integration", "contact_sampling")}
            )

        def observations(self):
            return torch.full((1, 31), self.tick * 0.01)

        def step(self, mean):
            self.tick += 1
            return dict(
                terminated=[False],
                truncated=[False],
                extras=dict(
                    actual_physics_steps=[10],
                    diagnostic={
                        key: torch.full((1, 6 if key != "state" else 13), float(self.tick))
                        for key in ("q", "dq", "motor", "NET", "state")
                    },
                ),
            )

        def close(self):
            events.append("close " + self.name)

    monkeypatch.setattr(sim2sim, "make_diagnostic_session", IO)
    path = tmp_path / "protocol.json"
    write_json(path, config)
    out = tmp_path / "result.json"
    assert (
        main(["sim2sim", "--config", str(path), "--output", str(out), "--dry-run", "--json"]) == 0
    )
    assert not events and not out.exists()
    assert main(["sim2sim", "--config", str(path), "--output", str(out), "--json"]) == 0
    result = read_json(out)
    assert events == ["open PhysX", "close PhysX", "open MuJoCo", "close MuJoCo"]
    assert result["extra_simulation_steps"] == 60 and result["status"] == "COMPLETE"
    assert all(
        row["metrics"]["policy_mean_replay_max_abs_error"] <= 1e-7 for row in result["engines"]
    )
    assert not result["physical_equivalence_claimed"]


def test_live_condition_conflict_closes_without_step(tmp_path, monkeypatch):
    config = protocol(tmp_path)
    closed = []

    class IO:
        def __init__(self, *args):
            self.actual = {key: config[key] for key in sim2sim.REQUIRED[1:-1]}
            self.actual.update(
                {k: config["backends"][0][k] for k in ("engine", "integration", "contact_sampling")}
            )
            self.actual["motor_caps"] = [5.5] * 6

        def close(self):
            closed.append(True)

        def step(self, *args):
            pytest.fail("condition mismatch stepped physics")

    monkeypatch.setattr(sim2sim, "make_diagnostic_session", IO)
    with pytest.raises(ValueError, match="motor_caps"):
        sim2sim.run_live(config, tmp_path / "result.json")
    assert closed == [True]


def test_live_short_fault_retains_actual_substeps_and_unknown_coverage(tmp_path, monkeypatch):
    config = protocol(tmp_path)

    class IO:
        physics_dt = 0.001

        def __init__(self, endpoint, protocol, directory):
            self.actual = {k: protocol[k] for k in sim2sim.REQUIRED[1:-1]}
            self.actual.update(
                {k: endpoint[k] for k in ("engine", "integration", "contact_sampling")}
            )

        def observations(self):
            return torch.zeros(1, 31)

        def step(self, mean):
            return dict(
                terminated=[False],
                truncated=[False],
                extras=dict(
                    actual_physics_steps=[2],
                    diagnostic={
                        key: torch.zeros(1, 6) for key in ("q", "dq", "motor", "NET", "state")
                    },
                ),
            )

        def close(self):
            pass

    monkeypatch.setattr(sim2sim, "make_diagnostic_session", IO)
    result = sim2sim.run_live(config, tmp_path / "short.json")
    assert result["extra_simulation_steps"] == 4 and result["status"] == "UNKNOWN"
    assert all(row["termination_reason"] == "short_step_fault" for row in result["engines"])


def test_completed_optimizer_receipt_charges_once_but_cannot_publish_partial_postprocess(tmp_path):
    job = _cpu_job(tmp_path, limits={"smoke_updates": 2, "formal_updates": 0})
    ctx = _production_context(tmp_path, job)

    class Adapter:
        def build(self, *args):
            pass

        def update_once(self, context, index):
            self.completed_optimization = {"update": context.updates_used + 1}
            raise RuntimeError("telemetry failed after real completed optimization")

        def capture(self, *args):
            pytest.fail("unverified current state must not be saved")

        def close(self):
            pass

    with pytest.raises(RuntimeError, match="telemetry"):
        driver.run_stage(ctx, Adapter(), stage="smoke")
    assert ctx.updates_used == 1


def test_formal_collect_label_aggregate_fit_uses_public_collector(tmp_path, monkeypatch):
    from dataclasses import replace

    from test_public_migration import supervision_job
    from test_supervision_consumers import ScriptedCollectionIO

    from native_stand.experiments.supervision import PhaseConfig
    from native_stand.experiments.supervision_flow import execute
    from native_stand.simulation import collection

    job = supervision_job(tmp_path)
    recipe = replace(
        job.supervision,
        initial_student={
            "path": job.supervision.teachers[0].path,
            "sha256": job.supervision.teachers[0].sha256,
        },
        phases=(
            PhaseConfig("collect", steps=3),
            PhaseConfig("label"),
            PhaseConfig("fit", rounds=1),
        ),
        repetitions=2,
    )
    job = replace(job, supervision=recipe)
    engines = []

    class IO(ScriptedCollectionIO):
        def step(self, action):
            outcome = super().step(action)
            outcome["extras"][
                "collection_conditions"
            ] = {}  # this fixture has no conditional Teacher
            return outcome

    def physical_io(job):
        env = IO()
        engines.append(env)
        return env

    monkeypatch.setattr(collection, "make_environment", physical_io)
    consumption = []
    rounds = []
    state, done = execute(
        job,
        tmp_path / "flow",
        stop_requested=lambda: False,
        on_round_complete=rounds.append,
        on_round_saved=lambda *a: None,
        on_physical_steps=consumption.append,
    )
    assert (
        done and state["rounds"] == 2 and len(engines) == 2 and all(env.closed for env in engines)
    )
    assert len(state["collections"]) == 3 and sum(consumption) == 120
    with np.load(state["dataset"]["path"]) as labels:
        assert len(set(labels["sample_ids"])) == len(labels["sample_ids"])
        assert "teacher_owner" in labels and "physical_episode_keys" in labels


def test_incomplete_rollout_never_recaptures_current_world(tmp_path):
    job = _cpu_job(tmp_path, limits={"smoke_updates": 3, "formal_updates": 0})
    ctx = _production_context(tmp_path, job, smoke=3)
    captures = []

    class Adapter:
        def build(self, *args):
            pass

        def update_once(self, context, index):
            return {"incomplete": True} if index == 3 else {"actor_steps": 1}

        def after_update(self, *args):
            pass

        def capture(self, context, counters):
            captures.append(context.updates_used)
            return _sentinel_full_state(job, sentinel=0.1)

        def close(self):
            pass

    ctx.save_due = lambda *a, **kw: False
    result = driver.run_stage(ctx, Adapter(), stage="smoke")
    assert result.limit == "rollout_incomplete" and ctx.updates_used == 2
    assert captures == [1] and ctx.latest_reference()["update"] == 1


def test_worker_checkpoint_path_cannot_overwrite_published_identity(tmp_path):
    job = _cpu_job(tmp_path)
    ctx = _production_context(tmp_path, job)
    value = _sentinel_full_state(job, sentinel=0.1)
    first = ctx.save(value, update=1, stage="smoke")
    before = file_sha256(first["checkpoint"])
    assert ctx.save(_sentinel_full_state(job, sentinel=0.9), update=1, stage="smoke") is None
    assert "FileExistsError" in read_json(ctx.run_dir / "save_error.json")["error"]
    assert file_sha256(first["checkpoint"]) == before
    assert ctx.latest_reference()["sha256"] == before
