"""Public CPU migration consumers; no physical integration or real training."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from native_stand.artifacts.checkpoint import load_actor_source
from native_stand.artifacts.io import file_sha256, write_json, write_npz
from native_stand.experiments.config import load_job_config
from native_stand.learning.inference import InferenceSpec, build_actor, load_model

FROZEN_TEST_SCOPE = "PORTABLE"
ROOT = Path(__file__).resolve().parents[1]


def model_file(tmp_path, kind="direct", width=8):
    spec = InferenceSpec(
        kind, 31, (width,), 6, {"version": "test31"}, {"action_version": "test_v1"}
    )
    actor = build_actor(spec)
    path = tmp_path / f"{kind}{width}.pt"
    torch.save(dict(actor=actor.state_dict(), inference_spec=spec.record()), path)
    return spec, path


def test_loaded_identity_is_bound_to_open_weights(tmp_path, monkeypatch):
    spec, path = model_file(tmp_path)
    sha = file_sha256(path)
    original = torch.load

    def replace_after_read(*a, **kw):
        value = original(*a, **kw)
        replacement = tmp_path / "replacement.pt"
        torch.save({"actor": {}, "different": True}, replacement)
        replacement.replace(path)
        return value

    monkeypatch.setattr(torch, "load", replace_after_read)
    from native_stand.artifacts.checkpoint import CheckpointError

    with pytest.raises(CheckpointError, match="加载期间"):
        load_actor_source(path)
    assert file_sha256(path) != sha


def test_inference_direct_and_deployment_move_without_source(tmp_path):
    from native_stand.artifacts.deployment import export_package, verify_package

    spec, path = model_file(tmp_path)
    source = load_actor_source(path)
    model = load_model(path, spec, sha256=source.sha256)
    assert not any(p.requires_grad for p in model.actor.parameters())
    with torch.no_grad():
        model.actor.mlp[-1].weight.zero_()
        model.actor.mlp[-1].bias.fill_(3)
    assert (model.mean(torch.zeros(1, 31)) == 3).all()
    package = tmp_path / "deploy"
    export_package(source, spec, package, execution={"wheel_cap_nm": 4.8})
    moved = tmp_path / "moved"
    package.rename(moved)
    path.unlink()
    assert verify_package(moved)["checks"]["ARM"] == "NOT_RUN"
    assert not (moved / "critic.pt").exists()


def test_multi_teacher_explicit_physical_labels_and_unknown(tmp_path):
    from native_stand.learning.distillation.routing import ActionLabels, Teacher, label

    spec, a = model_file(tmp_path, width=8)
    other, b = model_file(tmp_path, width=12)
    teachers = []
    for i, (s, p) in enumerate([(spec, a), (other, b)]):
        m = load_model(p, s, sha256=file_sha256(p))
        with torch.no_grad():
            m.actor.mlp[-1].weight.zero_()
            m.actor.mlp[-1].bias.fill_(0.5)
        teachers.append(
            Teacher(
                str(i), m, ActionLabels("affine", (2, 2, 80, 2, 2, 80), (0,) * 6), i, {"mode": i}
            )
        )
    x = torch.zeros(3, 31)
    before = x.clone()
    rng = torch.get_rng_state().clone()
    labels, owner, valid = label(
        teachers,
        x,
        {"mode": torch.tensor([0, 1, 2])},
        ActionLabels("affine", (1, 1, 40, 1, 1, 40), (0,) * 6),
    )
    assert owner.tolist() == [0, 1, -1] and valid.tolist() == [True, True, False]
    assert labels[0, 2] == 1 and labels[0, 0] == 1 and torch.isnan(labels[2]).all()
    assert torch.equal(x, before) and torch.equal(torch.get_rng_state(), rng)


def test_dataset_episode_roles_survive_aggregation_and_relocation(tmp_path):
    from native_stand.artifacts.dataset import create_view, load_view
    from native_stand.artifacts.runtime_paths import RuntimePaths

    data = tmp_path / "data.npz"
    write_npz(data, dict(states=np.zeros((100, 31)), episodes=np.arange(100)))
    ref = dict(path=str(data), sha256=file_sha256(data), collection_id="physical-one")
    view = tmp_path / "view.json"
    first = create_view(view, [ref], seed=9)
    second = create_view(view, [ref, ref], seed=9, previous=first)
    assert first["episode_roles"] == second["episode_roles"]
    with pytest.raises(ValueError, match="holdout"):
        load_view(view, roles=("holdout",), rehearsal=True)
    moved = tmp_path / "relocated.npz"
    data.rename(moved)
    _, batches = load_view(view, paths=RuntimePaths(files=((str(data), str(moved)),)))
    assert len(batches) == 1 and all(len(x["states"]) < 100 for x in batches)


def supervision_job(tmp_path):
    from native_stand.experiments.supervision import SupervisionConfig

    spec, path = model_file(tmp_path)
    data = tmp_path / "states.npz"
    write_npz(
        data,
        dict(
            states=np.random.default_rng(9).normal(size=(100, 31)).astype("float32"),
            episodes=np.arange(100),
        ),
    )
    job = load_job_config(ROOT / "configs/jobs/motion_supported015_dqgain.json")
    config = dict(
        teachers=[
            dict(
                name="one",
                path=str(path),
                sha256=file_sha256(path),
                inference=spec.record(),
                labels=dict(kind="affine", gains=[1] * 6, center=[0] * 6),
            )
        ],
        student=spec.record(),
        student_labels=dict(kind="affine", gains=[1] * 6, center=[0] * 6),
        dataset=dict(path=str(data), sha256=file_sha256(data)),
        phases=[
            dict(operation="label"),
            dict(operation="fit", rounds=3),
            dict(operation="compare"),
            dict(operation="export"),
        ],
        batch_size=16,
    )
    return dataclasses.replace(
        job,
        algorithm="distillation",
        runtime=dataclasses.replace(job.runtime, device="cpu"),
        supervision=SupervisionConfig.from_record(config),
    )


def test_real_flow_fit_stop_resume_and_export_no_recharge(tmp_path):
    from native_stand.experiments.supervision_flow import execute
    from native_stand.experiments.budget import Budget

    job = supervision_job(tmp_path)
    budget = Budget.create(
        tmp_path / "budget.json", smoke_updates=0, formal_updates=0, distill_rounds=3
    )
    directory = tmp_path / "flow"
    budget.reserve("attempt", "distill", directory)

    def complete(row):
        budget.charge_rounds("attempt", directory, rounds=1)
        budget.charge_resources("attempt", directory, supervised_updates=row["optimizer_steps"])

    state, done = execute(
        job,
        directory,
        stop_requested=lambda: budget.used()["distill"] >= 1,
        on_round_complete=complete,
        on_round_saved=lambda row: None,
    )
    assert not done and state["rounds"] == 1
    state, done = execute(
        job,
        directory,
        stop_requested=lambda: False,
        on_round_complete=complete,
        on_round_saved=lambda row: None,
    )
    assert (
        done and state["rounds"] == 3 and budget.used() == {"smoke": 0, "formal": 0, "distill": 3}
    )
    state, done = execute(
        job,
        directory,
        stop_requested=lambda: False,
        on_round_complete=lambda r: pytest.fail("recharged"),
        on_round_saved=lambda row: None,
    )
    assert done and state["phase_cursor"] == 4
    assert (directory / "phase_003_export/package/manifest.json").exists()


def test_recovery_actual909_observation_route_and_terminal(tmp_path):
    from native_stand.simulation.recovery_isaac.execution import Execution
    from native_stand.simulation.recovery_isaac.observations import observe
    from native_stand.simulation.recovery_isaac.preentry_lifecycle import Lifecycle
    from native_stand.simulation.v4_response import Response

    names = [f"p{i}" for i in range(13)]
    pars = [1.0, 0.1, 0.2, 4.0, 6.0, 1.0, -1.0, 2.0, 3.0, 0.4, 0.0, 0.0, 0.0]
    row = dict(
        parameters=dict(zip(names, pars)),
        effective_observer=dict(
            transport_ms=8, q_age_ms=2, v_age_ms=4, q_reference=[0.0, 0.0], velocity_tau_ms=3.0
        ),
    )
    ex = Execution(
        dict(sides={"L": row, "R": row}, parameter_names=names),
        Response,
        2,
        effort_caps=[40, 40, 4.8, 40, 40, 4.8],
    )
    q = torch.zeros(2, 6)
    ex.reset(torch.arange(2), q, q)
    life = Lifecycle(2)
    life.effort_caps = ex.caps
    life.reset(torch.arange(2), torch.full((2,), 0.18), torch.zeros(2))
    physical = torch.zeros(2, 13)
    physical[:, 3] = 0.34
    physical[:, 9:11] = 3
    physical[:, 12] = 1
    env = NS(
        execution=ex,
        life=life,
        height_conditioned=True,
        height_initial=torch.tensor([0.32, 0.36]),
        height_dynamic=torch.tensor([False, True]),
        scene=NS(env_origins=torch.zeros(2, 3)),
        physics=NS(coulomb=q, armature=q),
        ids=list(range(6)),
        prefix_s=torch.zeros(2),
        pulse_onset=torch.zeros(2),
        pulse_pattern=torch.zeros(2, 4),
        pulse_enabled=torch.zeros(2, dtype=torch.bool),
        initialized=torch.ones(2, dtype=torch.bool),
        cfg=NS(recovery_critic_version="recovery_standalone_height_causal909_v5"),
        handover_mode=False,
        measured=lambda: (
            torch.zeros(2, 16),
            q,
            q,
            physical,
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.full((2,), 0.1),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, 3),
        ),
        commanded_height=lambda: torch.tensor([0.32, 0.34]),
        recovery_control_owner=lambda: torch.ones(2, dtype=torch.bool),
        physical_profile=lambda: torch.zeros(2, 32),
    )
    ordinary = observe(env)
    assert ordinary["student_critic"].shape == (2, 909)
    assert ordinary["student_actor"].shape == (2, 31)
    assert torch.equal(observe(env)["student_critic"], ordinary["student_critic"])
    assert torch.allclose(
        ordinary["student_critic"][:, -5:],
        torch.tensor([[0.64, 0.64, 0.64, 0.72, 0], [0.72, 0.68, 0.64, 0.72, 1]]),
    )


def test_recovery_worker_real_policy_budget_save_publish_and_resume(tmp_path, monkeypatch):
    from tensordict import TensorDict
    from native_stand.experiments import owner, worker
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.state import RunStateStore
    from native_stand.artifacts.checkpoint import load_source
    from native_stand.simulation.driver import Backend
    from native_stand.learning.ppo.recovery_cubic import PhysicalActor

    job = load_job_config(ROOT / "configs/jobs/recovery_preentry48_remaining.json")
    contract = {
        "action_version": "recovery_cubic_physical_Gaussian_v2",
        "center_rad": [0.0] * 4,
        "leg_lower_rad": [-1.2] * 4,
        "leg_upper_rad": [1.2] * 4,
    }
    contract_path = tmp_path / "contract.json"
    write_json(contract_path, contract)
    actor = PhysicalActor(contract, hidden_dims=(8,))
    initial = tmp_path / "initial.pt"
    torch.save(dict(actor=actor.state_dict(), output_contract=contract), initial)
    runtime = tmp_path / "runtime.json"
    write_json(runtime, {"bindings": {}})
    anchor_path = tmp_path / "full-pool.npz"
    bx = np.zeros((400, 31), dtype="float32")
    bx[:, 24] = 0.68
    bx[:, 25:31] = [0, 1, 0, 1, 0, 0]
    write_npz(
        anchor_path, dict(observation=bx, physical_target=np.zeros((400, 6), dtype="float32"))
    )
    prior_spec, prior_path = model_file(tmp_path, kind="tanh", width=8)
    assets = {
        "qualified_physical_BC_anchor": dict(
            path=str(anchor_path),
            sha256=file_sha256(anchor_path),
            sampling="full_pool_height_dq_stratified",
            coefficient=0.1,
            batch_size=512,
        ),
        "old_Recovery_action_prior": dict(
            path=str(prior_path),
            sha256=file_sha256(prior_path),
            actor_hidden_dims=[8],
            coefficient=0.01,
            max_rows=64,
        ),
        "recovery_training_contract": dict(
            path=str(contract_path), sha256=file_sha256(contract_path)
        ),
        "recovery_isaac_runtime": dict(path=str(runtime), sha256=file_sha256(runtime)),
        "recovery_standalone_course": job.assets["recovery_standalone_course"],
    }
    job = dataclasses.replace(
        job,
        model=dataclasses.replace(job.model, actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        initialization=dataclasses.replace(
            job.initialization, checkpoint=initial, sha256=file_sha256(initial)
        ),
        assets=assets,
        runtime=dataclasses.replace(job.runtime, device="cpu", envs=2, steps_per_env=192),
        limits=dataclasses.replace(job.limits, smoke_updates=0, formal_updates=6),
        storage=dataclasses.replace(
            job.storage, checkpoint_every_updates=1, metrics_every_updates=1, tensorboard=False
        ),
        ppo=dataclasses.replace(job.ppo, epochs=1, minibatches=1),
        diagnostics={"training_candidate_selection": {"every_updates": 1, "minimum_ended": 1}},
    )
    from native_stand.simulation.recovery_isaac.preentry_lifecycle import Lifecycle
    from native_stand.tasks.recovery.swing_priority_course import RecoveryCourse
    import copy

    class SensorIO:
        num_envs = 2
        infinite_horizon = False

        def __init__(self):
            self.env = self
            self.completed = []
            self.completed_count = 0
            self.closed = False
            self.life = Lifecycle(2)
            self.life.effort_caps = torch.tensor([40, 40, 4.8, 40, 40, 4.8]).expand(2, -1)
            self.life.reset(torch.arange(2), torch.full((2,), 0.18), torch.zeros(2))
            self.courses = [RecoveryCourse(7 + i) for i in range(2)]
            self.specs = [c.choose() for c in self.courses]
            self.ages = torch.zeros(2, dtype=torch.long)
            from native_stand.simulation.recovery_isaac.execution import Execution
            from native_stand.simulation.v4_response import Response

            names = [f"p{i}" for i in range(13)]
            params = [1.0, 0.1, 0.2, 4.0, 6.0, 1.0, -1.0, 2.0, 3.0, 0.4, 0.0, 0.0, 0.0]
            profile = dict(
                parameters=dict(zip(names, params)),
                effective_observer=dict(
                    transport_ms=8,
                    q_age_ms=2,
                    v_age_ms=4,
                    q_reference=[0.0, 0.0],
                    velocity_tau_ms=3.0,
                ),
            )
            self.execution = Execution(
                dict(parameter_names=names, sides={"L": profile, "R": profile}),
                Response,
                2,
                effort_caps=[40, 40, 4.8, 40, 40, 4.8],
            )
            self.q = torch.tensor([[0.1, 0, 0, 0.1, 0, 0]]).expand(2, -1).clone()
            self.execution.reset(torch.arange(2), self.q, torch.zeros_like(self.q))
            self.cfg = NS(recovery_critic_version="recovery_standalone_height_causal909_v5")
            self.height_conditioned = True
            self.height_initial = torch.full((2,), 0.34)
            self.height_dynamic = torch.zeros(2, dtype=torch.bool)
            self.scene = NS(env_origins=torch.zeros(2, 3))
            self.physics = NS(coulomb=torch.zeros(2, 6), armature=torch.zeros(2, 6))
            self.ids = list(range(6))
            self.prefix_s = torch.zeros(2)
            self.pulse_onset = torch.zeros(2)
            self.pulse_pattern = torch.zeros(2, 4)
            self.pulse_enabled = torch.zeros(2, dtype=torch.bool)
            self.initialized = torch.ones(2, dtype=torch.bool)
            self.handover_mode = False

        def commanded_height(self):
            return torch.full((2,), 0.34)

        def recovery_control_owner(self):
            return torch.ones(2, dtype=torch.bool)

        def physical_profile(self):
            return torch.zeros(2, 32)

        def measured(self):
            physical = torch.zeros(2, 13)
            physical[:, 3] = 0.34
            physical[:, 9:11] = 3
            physical[:, 12] = 1
            physical[1, 3] = torch.where(self.ages[1] >= 2, 0.08, 0.18)
            frame = torch.zeros(2, 16)
            frame[:, 4] = -1
            frame[:, 6:10] = self.q[:, [0, 1, 3, 4]] / 1.5
            return (
                frame,
                self.q,
                torch.zeros_like(self.q),
                physical,
                torch.zeros(2, 3),
                torch.zeros(2, 3),
                torch.full((2,), 0.15),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, 3),
            )

        def observations(self):
            from native_stand.simulation.recovery_isaac.observations import observe

            return TensorDict(observe(self), batch_size=[2])

        def step(self, actions):
            self.ages += 1
            physical = torch.zeros(2, 13)
            physical[:, 3] = 0.34
            physical[:, 9:11] = 3
            physical[:, 12] = 1
            physical[1, 3] = torch.where(self.ages[1] >= 2, 0.08, 0.18)
            result = self.life.observe(
                time_s=self.ages * 0.01,
                state=physical,
                target=actions,
                motor=torch.zeros(2, 6),
                student=torch.ones(2, dtype=torch.bool),
                terminal=physical[:, 3] < 0.1,
                truncated=self.ages * 0.01 >= 23,
                hp10=torch.zeros(2, 4),
                angular_velocity=torch.zeros(2, 3),
                physical_q=torch.tensor([[0.1, 0, 0, 0.1, 0, 0]]).expand(2, -1),
                clearance=torch.full((2,), 0.15),
            )
            for i in torch.where(result["done"])[0].tolist():
                spec = self.specs[i]
                row = dict(
                    student_seconds=float(self.ages[i]) * 0.01,
                    teacher_seconds=0.0,
                    DR_strength=spec["DR_strength"],
                    steps=int(self.ages[i]),
                    group=spec["group"],
                    success=bool(result["success"][i]),
                    physical_quality=bool(physical[i, 3] >= 0.3),
                    nominal_anchor=spec["nominal_anchor"],
                    command_height_m=spec["height_m"],
                    height_dynamic=spec["height_dynamic"],
                )
                self.completed.append(row)
                self.completed_count += 1
                self.courses[i].record(
                    spec["group"],
                    student_seconds=row["student_seconds"],
                    teacher_seconds=0.0,
                    passed=row["success"],
                )
                self.specs[i] = self.courses[i].choose()
            terminal = self.observations()["student_critic"].clone()
            reward = result["reward"].clone()
            done = result["done"].clone()
            terminated = result["terminated"].clone()
            truncated = result["truncated"].clone()
            ids = torch.where(done)[0]
            if len(ids):
                self.life.reset(ids, torch.full((len(ids),), 0.18), torch.zeros(len(ids)))
                self.ages[ids] = 0
            return dict(
                rewards=reward,
                dones=done,
                terminated=terminated,
                truncated=truncated,
                terminal_observations={"student_critic": terminal},
                extras={},
            )

        def runtime_state(self):
            return copy.deepcopy(
                dict(
                    completed=self.completed,
                    completed_count=self.completed_count,
                    life=self.life.state_dict(),
                    execution=self.execution.state_dict(),
                    ages=self.ages,
                    courses=self.courses,
                    specs=self.specs,
                )
            )

        def load_runtime_state(self, state):
            self.completed = state["completed"]
            self.completed_count = state["completed_count"]
            self.life.load_state_dict(state["life"])
            self.execution.load_state_dict(state["execution"])
            self.ages = state["ages"]
            self.courses = state["courses"]
            self.specs = state["specs"]

        def close(self):
            self.closed = True

    sensors = []

    def make_env(job):
        env = SensorIO()
        sensors.append(env)
        return env

    monkeypatch.setattr(
        worker, "resolve_backend", lambda job: Backend("scripted-physical-IO", make_env)
    )
    run = owner.create_run(job, runs_root=tmp_path / "runs", run_id="recovery-cpu")
    budget = Budget.create(run / "budget.json", smoke_updates=0, formal_updates=6)
    store = RunStateStore(run)
    store.create(run_id=run.name, owner={"pid": 0, "role": "fixture"}, stage="formal")
    context = worker._new_context(run, "formal")
    context.stop_at_update = 3
    try:
        result = worker._execute_stage(context, "formal", None)
    finally:
        context.metrics.close()
    budget.finalize(context.attempt_id, context.output, "STOPPED_BOUNDED")
    assert result.stopped and budget.used()["formal"] == 3 and sensors[0].closed
    source = load_source(run / "checkpoints/model_update_3.pt")
    assert source.payload["warmup_updates_done"] == 3
    assert source.payload["training_candidates"]["best_candidate"] is None
    assert source.payload["recovery_episode_cursor"] > 0
    context = worker._new_context(run, "formal")
    try:
        result = worker._execute_stage(context, "formal", source)
    finally:
        context.metrics.close()
    assert result.updates == 3 and budget.used()["formal"] == 6 and sensors[-1].closed
    final = load_source(run / "checkpoints/model_update_6.pt")
    assert final.payload["warmup_updates_done"] == 5
    assert final.payload["recovery_episode_cursor"] > source.payload["recovery_episode_cursor"]
    assert context.refs.candidate_index(run.name)["best_candidate"] is not None
    assert final.payload["recovery_pending_publication"]["end_update"] == 6
    assert final.payload["actor_update_enabled"]
    assert set(final.payload["optimizer"]) == {"actor", "critic"}
    assert any(
        not torch.equal(source.payload["actor"][k], v) for k, v in final.payload["actor"].items()
    )


def test_scene_provider_and_recorded_sim2sim_cli(tmp_path, capsys):
    from native_stand.simulation.scenes import height_above_ground
    from native_stand.cli import main

    roots = torch.tensor([[1.0, 2.0, 0.5], [2.0, 1.0, 0.8]])
    assert torch.allclose(
        height_above_ground(roots, lambda xy: xy[:, 0] * 0.1), torch.tensor([0.4, 0.6])
    )
    trace = tmp_path / "trace.npz"
    write_npz(
        trace,
        dict(
            time=np.array([0.0, 0.01, 0.02]),
            q=np.ones((3, 6)),
            dq=np.zeros((3, 6)),
            motor=np.ones((3, 6)) * 4.8,
            NET=np.zeros((3, 6)),
            state=np.zeros((3, 13)),
        ),
    )
    config = dict(
        version="sim2sim_recorded_protocol_v1",
        model={"sha256": "a" * 64},
        input_contract="31",
        output_contract="physical",
        scene="flat",
        command=[0, 0, 0.34],
        initial_state="bound",
        timing=[1, 5, 10],
        V4="full",
        motor_caps=[40, 40, 4.8, 40, 40, 4.8],
        scoring_window_s=[0, 0.02],
        traces=[
            dict(
                path=str(trace),
                sha256=file_sha256(trace),
                engine=e,
                integration="explicit",
                contact_sampling="engine-native",
            )
            for e in ("PhysX", "MuJoCo")
        ],
    )
    path = tmp_path / "sim2sim.json"
    write_json(path, config)
    assert main(["sim2sim", "--config", str(path), "--dry-run", "--json"]) == 0
    output = tmp_path / "result.json"
    assert main(["sim2sim", "--config", str(path), "--output", str(output), "--json"]) == 0
    assert json.loads(output.read_text())["extra_simulation_steps"] == 0


def test_supervision_worker_uses_public_owner_state_and_no_isaac_preflight(tmp_path, monkeypatch):
    from native_stand.experiments import owner, worker
    from native_stand.experiments.distillation_job import run_distillation_worker
    from native_stand.experiments.budget import Budget
    from native_stand.experiments.state import RunStateStore

    job = supervision_job(tmp_path)
    # It remains an Isaac collection recipe, but its actual phases are offline.
    job = dataclasses.replace(job, backend="isaac")
    run = owner.create_run(job, runs_root=tmp_path / "runs", run_id="bc-worker")
    Budget.create(run / "budget.json", smoke_updates=0, formal_updates=0, distill_rounds=3)
    RunStateStore(run).create(run_id=run.name, owner={"pid": 0, "role": "fixture"}, stage="distill")
    monkeypatch.setattr(worker, "STOP_REQUESTED", False)
    assert run_distillation_worker(run, job) == 0
    state = json.loads((run / "state.json").read_text())
    book = json.loads((run / "budget.json").read_text())
    assert (
        state["status"] == "COMPLETED" and state["rounds_used"] == 3 and state["updates_used"] == 0
    )
    assert book["used"] == {"smoke": 0, "formal": 0, "distill": 3}
    assert book["resources_used"]["supervised_updates"] > 3
    assert "physical_steps" not in book.get("resources_used", {})


def test_all_supervision_dependencies_relocate_with_original_unavailable(tmp_path):
    import shutil
    from native_stand.experiments.supervision_flow import execute

    original = tmp_path / "old"
    original.mkdir()
    job = supervision_job(original)
    rounds = []
    execute(
        job,
        original / "flow",
        stop_requested=lambda: len(rounds) == 1,
        on_round_complete=rounds.append,
        on_round_saved=lambda r: None,
    )
    moved = tmp_path / "new"
    shutil.move(original, moved)
    mapping = tmp_path / "mapping.json"
    write_json(
        mapping, dict(version="runtime_path_mapping_v1", prefixes={str(original): str(moved)})
    )
    job = dataclasses.replace(
        job,
        assets={**job.assets, "path_mapping": dict(path=str(mapping), sha256=file_sha256(mapping))},
    )
    state, done = execute(
        job,
        moved / "flow",
        stop_requested=lambda: False,
        on_round_complete=rounds.append,
        on_round_saved=lambda r: None,
    )
    assert done and state["rounds"] == 3 and len(rounds) == 3 and not original.exists()
    assert str(original) not in state["latest"]["path"]


def test_rehearsal_and_dagger_use_same_bc_loop(tmp_path):
    from native_stand.experiments.supervision_flow import execute
    from native_stand.experiments.supervision import PhaseConfig

    job = supervision_job(tmp_path)
    initial = dict(path=job.supervision.teachers[0].path, sha256=job.supervision.teachers[0].sha256)
    original = job.supervision.dataset
    with np.load(original["path"]) as raw:
        replay = tmp_path / "persisted-rehearsal.npz"
        write_npz(
            replay,
            dict(
                states=raw["states"],
                episodes=raw["episodes"],
                roles=np.full(len(raw["states"]), "train"),
            ),
        )
    rehearsal = dict(path=str(replay), sha256=file_sha256(replay), collection_id="rehearsal")
    recipe = dataclasses.replace(
        job.supervision,
        initial_student=initial,
        rehearsal=(rehearsal,),
        phases=(
            PhaseConfig("collect", steps=2),
            PhaseConfig("label"),
            PhaseConfig("fit", rounds=2),
        ),
    )
    job = dataclasses.replace(job, supervision=recipe)
    counts = []

    def physical_io(job, source, spec, steps, directory, charge):
        assert source == initial and steps == 2
        if charge:
            charge(20)
        return dict(**original, collection_id="student-episode")

    state, done = execute(
        job,
        tmp_path / "dagger",
        stop_requested=lambda: False,
        on_round_complete=counts.append,
        on_round_saved=lambda r: None,
        collector=physical_io,
        on_physical_steps=lambda count: counts.append({"physical": count}),
    )
    assert done and state["rounds"] == 2 and counts[0] == {"physical": 20}
    assert state["phase_cursor"] == 3


def test_recovery_factory_full_pool_bc_and_independent_prior_gradients(tmp_path):
    from tensordict import TensorDict
    from native_stand.learning.policies.factory import build_policy
    from native_stand.learning.ppo.recovery_action_prior import offsets

    spec, prior_path = model_file(tmp_path, kind="tanh", width=8)
    contract = {
        "action_version": "recovery_cubic_physical_Gaussian_v2",
        "center_rad": [0.0] * 4,
        "leg_lower_rad": [-1.2] * 4,
        "leg_upper_rad": [1.2] * 4,
    }
    path = tmp_path / "contract.json"
    write_json(path, contract)
    x = torch.zeros(400, 31)
    x[:, 24] = torch.linspace(0.6, 0.76, 400)
    x[:, 25:31] = torch.tensor([0, 1, 0, 1, 0, 0])
    x[200:, 10] = 0.2
    anchors = tmp_path / "anchor.npz"
    write_npz(
        anchors,
        dict(observation=x.numpy(), physical_target=np.ones((400, 6), dtype="float32") * 0.1),
    )
    job = load_job_config(ROOT / "configs/jobs/recovery_preentry48_remaining.json")
    job = dataclasses.replace(
        job,
        model=dataclasses.replace(job.model, actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        initialization=dataclasses.replace(
            job.initialization, mode="scratch", checkpoint=None, sha256=None
        ),
        runtime=dataclasses.replace(job.runtime, envs=2, steps_per_env=2, device="cpu"),
        assets={
            "recovery_training_contract": dict(path=str(path), sha256=file_sha256(path)),
            "qualified_physical_BC_anchor": dict(
                path=str(anchors),
                sha256=file_sha256(anchors),
                sampling="full_pool_height_dq_stratified",
                coefficient=0.1,
                batch_size=512,
            ),
            "old_Recovery_action_prior": dict(
                path=str(prior_path),
                sha256=file_sha256(prior_path),
                actor_hidden_dims=[8],
                coefficient=0.01,
                max_rows=64,
            ),
        },
    )
    obs = TensorDict(
        {
            "student_actor": x[:2],
            "student_critic": torch.zeros(2, 909),
            "recovery_actor_valid": torch.ones(2, 1, dtype=torch.bool),
        },
        batch_size=[2],
    )
    obs["student_critic"][:, 863] = 0.36
    obs["student_critic"][:, offsets()["kinematics"] + 3] = 0.2
    batch = NS(observations=obs)
    for use_bc in [True, False]:
        assets = dict(job.assets)
        if not use_bc:
            assets.pop("qualified_physical_BC_anchor")
        handle = build_policy(dataclasses.replace(job, assets=assets), obs, device="cpu")
        ppo = handle["ppo"]
        loss = ppo.actor_auxiliary_loss(batch, torch.ones(2, dtype=torch.bool))
        assert loss is not None and loss > 0
        gradients = torch.autograd.grad(loss, list(ppo.actor.parameters()), allow_unused=True)
        assert sum(float(g.abs().sum()) for g in gradients if g is not None) > 0
        assert (
            ppo.action_prior_readback["rows"] == 2
            and ppo.action_prior_readback["coefficient"] == 0.01
        )
        state = handle["capture"]({"updates": 6}, curriculum={})
        assert state["recovery_auxiliary_identity"]["prior"]["sha256"] == file_sha256(prior_path)
        assert (state["recovery_auxiliary_identity"]["BC"] is not None) == use_bc


def test_same_physical_episode_never_changes_partition_on_relabel(tmp_path):
    from native_stand.experiments.supervision_flow import execute

    job = supervision_job(tmp_path)
    recipe = dataclasses.replace(job.supervision, repetitions=2)
    job = dataclasses.replace(job, supervision=recipe)
    rounds = []
    state, done = execute(
        job,
        tmp_path / "repeat",
        stop_requested=lambda: False,
        on_round_complete=rounds.append,
        on_round_saved=lambda r: None,
    )
    assert done and state["rounds"] == 6 and len(rounds) == 6
    a = json.loads((tmp_path / "repeat/phase_000_label/view.json").read_text())
    b = json.loads((tmp_path / "repeat/phase_004_label/view.json").read_text())
    assert a["episode_roles"] == b["episode_roles"]


def test_supervision_continuous_and_segmented_adam_rng_equal(tmp_path):
    from native_stand.experiments.supervision_flow import execute

    job = supervision_job(tmp_path)
    continuous, done = execute(
        job,
        tmp_path / "continuous",
        stop_requested=lambda: False,
        on_round_complete=lambda r: None,
        on_round_saved=lambda r: None,
    )
    rounds = []
    execute(
        job,
        tmp_path / "segmented",
        stop_requested=lambda: len(rounds) == 1,
        on_round_complete=rounds.append,
        on_round_saved=lambda r: None,
    )
    segmented, done = execute(
        job,
        tmp_path / "segmented",
        stop_requested=lambda: False,
        on_round_complete=rounds.append,
        on_round_saved=lambda r: None,
    )
    a = torch.load(continuous["latest"]["path"], weights_only=False)
    b = torch.load(segmented["latest"]["path"], weights_only=False)

    def compare(x, y):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                compare(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            for xx, yy in zip(x, y, strict=True):
                compare(xx, yy)
        elif isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)
        else:
            assert x == y

    for field in ("student", "optimizer", "rng", "shuffle_rng", "split", "round"):
        compare(a[field], b[field])


def test_export_actual_onnx_reference_and_C_source_identity(tmp_path):
    from native_stand.artifacts.deployment import export_package, verify_package, job_dependencies

    spec, path = model_file(tmp_path)
    source = load_actor_source(path)
    job = load_job_config(ROOT / "configs/jobs/motion_supported015_dqgain.json")
    refs = job_dependencies(job)
    assert any(r["role"] == "C_source" for r in refs)
    package = tmp_path / "onnx-package"
    record = export_package(
        source, spec, package, execution={"physics_dt_s": 0.001}, dependencies=refs, onnx=True
    )
    assert record["checks"]["ONNX_reference"] is True
    assert verify_package(package)["checks"]["ORT"] == "NOT_RUN"
    (package / "actor.onnx").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="identity"):
        verify_package(package)


def test_cli_structured_recipe_dry_run_does_not_load_models(tmp_path, monkeypatch, capsys):
    from native_stand.cli import main

    job = supervision_job(tmp_path)
    raw = job.as_run_record()
    for key in ("config_path", "config_sha256"):
        raw.pop(key)
    raw.pop("preset")
    file = tmp_path / "recipe.json"
    write_json(file, raw)

    def forbidden(*a, **kw):
        raise AssertionError("dry-run deserialized or spawned")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    assert main(["distill", "--config", str(file), "--dry-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["PPO_updates"] == 0
    assert not (tmp_path / "runs").exists()
