"""Production collection boundary; only the physical I/O/session is scripted."""

from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from test_motion_control_contract import Teacher
from test_motion_spin_branches import fixture

from native_stand.artifacts.io import read_json
from native_stand.learning.inference import spec_for_job
from native_stand.simulation import collection
from native_stand.simulation.isaac import motion_backend
from native_stand.simulation.isaac.environment import control_router
from native_stand.tasks.motion.course_strategy import resolve
from native_stand.tasks.motion.legacy_spec_adapter import build_environment_spec

FROZEN_TEST_SCOPE = "PORTABLE"


class ScriptedMotionIO:
    def __init__(self, spec, _, *, envs, job, source, prefix, corrections):
        self.num_envs, self.device, self.job = envs, "cpu", job
        self.stage_spec = spec["curriculum"]["stages"][0]
        self.command_index = torch.arange(envs) % len(self.stage_spec["commands"])
        self.target_xy = torch.tensor(self.stage_spec["commands"])[self.command_index].float()
        self.identified_dr = NS(
            profiles=torch.tensor(self.stage_spec["identified_profiles"])[self.command_index],
            amplitude=torch.tensor(self.stage_spec["identified_amplitudes"])[
                self.command_index, None
            ],
        )
        self.transient_events = NS(
            mode=torch.full((envs,), -1), active=torch.zeros(envs, dtype=torch.bool)
        )
        self.rise_hold = NS(episode=NS(steps=torch.zeros(envs, dtype=torch.long)))
        self.height_adapter = NS(target=torch.full((envs,), 0.34))
        modes = torch.full((envs,), 2)
        if envs > 2:
            modes[-1] = 7
        self.fsm = NS(mode=modes)
        self.initialization_checked = torch.ones(envs, dtype=torch.bool)
        self.sim = NS(get_physics_step_count=lambda: self.tick)
        self.tick = self.resets = self.closed = 0
        self.fault = False
        self.curriculum_training = False
        self.motion_cohort_mix = torch.zeros(len(self.stage_spec["commands"]))
        self.router = control_router(job, source, prefix)
        self.router.initialize(self, {})
        self.previous = torch.zeros(envs, 6)
        self.student_terminal_critic = torch.zeros(envs, 877)

    def select_initial_family(self, family, split):
        assert family == "motion_native_normal"
        self.split = split

    def observations(self):
        actor = torch.zeros(self.num_envs, 31)
        actor[:, 22:24] = self.target_xy / 4
        actor[:, 24] = self.height_adapter.target / 0.5
        self.command = torch.cat((self.target_xy, self.height_adapter.target[:, None]), -1)
        actor[:, 16:22] = self.previous / actor.new_tensor([1.5, 1.5, 100, 1.5, 1.5, 100])
        self.student_obs = dict(student_actor=actor, student_critic=torch.zeros(self.num_envs, 877))
        return self.student_obs

    def reset(self):
        self.resets += 1
        if self.num_envs > 1:
            self.r4_prefix_remaining[1] = 1
        return self.observations(), {}

    def step(self, action):
        raw, _ = self.router.actions(self, action)
        self.previous.copy_(raw * raw.new_tensor([1.5, 1.5, 100, 1.5, 1.5, 100]))
        self.tick += 4 if self.fault else 10
        if self.fault:
            raise RuntimeError("scripted physical I/O fault after four steps")
        self.rise_hold.episode.steps += 1
        if hasattr(self, "collection_snapshot"):
            self.collection_snapshot()
        # Partial reset deliberately destroys live owner/cohort metadata.
        self.command_index[0] += 1
        self.r4_owner[0] = 0
        if hasattr(self, "robot"):
            self.robot.data.joint_pos.torch[0] = 999
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = terminated.clone()
        terminated[0] = True
        if self.num_envs > 1:
            truncated[1] = True
        return self.observations(), torch.ones(self.num_envs), terminated, truncated, {}

    def close(self):
        self.closed += 1


def install(tmp_path, monkeypatch, *, envs=3):
    job, _, _ = fixture(tmp_path, envs=envs)
    job = replace(
        job, assets={k: v for k, v in job.assets.items() if k != "source_policy_retention"}
    )
    loaded = motion_backend._resolve_source(job, None)
    source = loaded.payload
    strategy = resolve(job.curriculum.version)
    strategy.prepare(source, job.runtime.envs)
    spec = build_environment_spec(
        read_json(job.assets["runtime_template"]), source, job, stage_builder=strategy.stage_config
    )
    observed = []
    app = NS(close=lambda: observed.append("closed"))

    def physical_io(*args, **kwargs):
        raw = ScriptedMotionIO(*args, **kwargs)
        observed.append(raw)
        return raw

    def session(job, *, train):
        assert train
        return dict(
            source=source,
            spec=spec,
            config=job.reward.fields["scale_corrections"],
            prefix=Teacher(),
            environment=physical_io,
            observations=lambda env, obs: obs,
        ), app

    # Only engine/AppLauncher allocation is replaced; real source, course,
    # _build_environment, Handover, decoder and collection factory are consumed.
    monkeypatch.setattr(motion_backend, "prepare_session", session)
    return job, source, observed


def test_formal_collection_factory_real_owner_reset_conditions_and_resources(tmp_path, monkeypatch):
    job, _, created = install(tmp_path, monkeypatch)
    env = collection.make_environment(job)
    raw = created[0]
    before = env.observations()["student_actor"].clone()
    assert raw.split == "train" and raw.resets == 1
    result = env.step(torch.zeros(3, 6))
    extra = result["extras"]
    assert extra["control_owner"].tolist() == [0, 1, 2]
    assert extra["collection_conditions"]["cohort"].tolist() == [0, 1, 2]
    assert raw.command_index.tolist() == [1, 1, 2]
    torch.testing.assert_close(extra["collection_conditions"]["vx"], before[:, 22] * 4)
    assert extra["actual_physics_steps"] == [10] * 3
    assert env.cumulative_physics_steps == 30
    assert result["terminated"].tolist() == [True, False, False]
    assert result["truncated"].tolist() == [False, True, False]
    # Actual adopted physical targets, not shadow predictions, supply previous.
    torch.testing.assert_close(
        env.observations()["student_actor"][:, 16:22],
        raw.previous / raw.previous.new_tensor([1.5, 1.5, 100, 1.5, 1.5, 100]),
    )
    env.close()
    env.close()
    assert created.count("closed") == raw.closed == 1
    with pytest.raises(ValueError, match="resources"):
        motion_backend.make_collection_env(job, resources=replace(job.runtime, envs=1))


def test_fault_keeps_actual_partial_consumption_and_stops_reuse(tmp_path, monkeypatch):
    job, _, created = install(tmp_path, monkeypatch)
    env = collection.make_environment(job)
    raw = created[0]
    raw.fault = True
    with pytest.raises(RuntimeError, match="four steps") as error:
        env.step(torch.zeros(3, 6))
    assert error.value.actual_physics_steps == [4] * 3
    assert env.cumulative_physics_steps == 12
    with pytest.raises(RuntimeError, match="faulted"):
        env.step(torch.zeros(3, 6))
    env.close()
    assert raw.closed == 1


def test_collect_to_real_teacher_labels_view_and_fit(tmp_path, monkeypatch):
    from native_stand.artifacts.checkpoint import save_full
    from native_stand.artifacts.dataset import create_view, expand_sources, read_source
    from native_stand.artifacts.io import file_sha256
    from native_stand.learning.distillation.routing import ActionLabels, label
    from native_stand.learning.distillation.routing import Teacher as RoutedTeacher
    from native_stand.learning.inference import load_model

    job, source, created = install(tmp_path, monkeypatch)
    saved = tmp_path / "source_actor.pt"
    save_full(saved, source)
    spec = spec_for_job(job, source["output_contract"])
    ref = dict(path=str(saved), sha256=file_sha256(saved))
    charged = []
    manifest = collection.collect(
        job,
        ref,
        spec.record(),
        2,
        tmp_path / "collect",
        charged.append,
        required_conditions=("control_owner", "target_vx", "height"),
        chunk_steps=1,
    )
    assert manifest["complete"] and charged == [30, 30]
    refs = expand_sources(manifest)
    chunks = [read_source(r) for r in refs]
    states = torch.from_numpy(np.concatenate([c["states"] for c in chunks]))
    conditions = {
        k: np.concatenate([c["conditions"][k] for c in chunks]) for k in chunks[0]["conditions"]
    }
    codec = ActionLabels(kind="motion_contract", contract=source["output_contract"])
    model = load_model(saved, spec, sha256=ref["sha256"])
    teachers = [
        RoutedTeacher("student-owned", model, codec, 2, {"control_owner": 0}),
        RoutedTeacher("prefix-owned", model, codec, 1, {"control_owner": 1}),
    ]
    labels, routed, valid = label(teachers, states, conditions, codec, student_kind="tanh")
    assert routed.tolist() == [0, 1, -1, 0, 0, -1]
    torch.testing.assert_close(labels[valid], model.mean(states[valid]))
    # Accumulation uses physical episode keys; no engine/PPO update is created.
    view = create_view(tmp_path / "view.json", refs)
    assert len(view["sources"]) == 2 and len(view["episode_roles"]) == 5
    assert created[-1] == "closed"

    from native_stand.experiments.supervision import PhaseConfig, SupervisionConfig, TeacherConfig
    from native_stand.experiments.supervision_flow import execute

    changed = dict(source)
    changed["actor"] = {k: v.clone() for k, v in source["actor"].items()}
    changed["actor"]["mlp.2.bias"].add_(0.02)
    teacher_path = tmp_path / "label_teacher.pt"
    save_full(teacher_path, changed)
    recipe = SupervisionConfig(
        teachers=tuple(
            TeacherConfig(
                name,
                str(teacher_path),
                file_sha256(teacher_path),
                spec.record(),
                dict(kind="motion_contract", contract=source["output_contract"]),
                conditions={"control_owner": owner},
            )
            for name, owner in (("student-owned", 0), ("prefix-owned", 1))
        ),
        phases=(
            PhaseConfig("collect", steps=24),
            PhaseConfig("label"),
            PhaseConfig("fit", rounds=1),
        ),
        student=spec.record(),
        student_labels=dict(kind="motion_contract", contract=source["output_contract"]),
        dataset=manifest,
        initial_student=ref,
    )
    rounds = []
    state, complete = execute(
        replace(job, supervision=recipe),
        tmp_path / "flow",
        stop_requested=lambda: False,
        on_round_complete=rounds.append,
        on_round_saved=lambda row: None,
    )
    assert complete and len(rounds) == state["rounds"] == 1
    fitted = torch.load(state["latest"]["path"], weights_only=False)
    assert not torch.equal(fitted["student"]["mlp.2.bias"], source["actor"]["mlp.2.bias"])


def test_environment_reset_failure_closes_original_io_without_masking_failure(
    tmp_path, monkeypatch
):
    job, _, created = install(tmp_path, monkeypatch)

    def fail(self):
        raise RuntimeError("scripted reset failure")

    monkeypatch.setattr(ScriptedMotionIO, "reset", fail)
    with pytest.raises(RuntimeError, match="reset failure"):
        collection.make_environment(job)
    assert created[0].closed == 1 and created[-1] == "closed"


@pytest.mark.parametrize("mismatch", [None, "timing", "V4", "initial_state"])
def test_diagnostic_verifies_actual_bindings_and_captures_before_reset(
    tmp_path, monkeypatch, mismatch
):
    from native_stand.artifacts.io import file_sha256, write_json
    from native_stand.simulation.isaac.motion_commands import set_evaluation_command

    job, source, created = install(tmp_path, monkeypatch, envs=1)
    env = collection.make_environment(job)
    e = created[0]
    e.identified_dr = NS(
        packet_age=torch.tensor([20]),
        leg_packet_age=torch.tensor([20]),
        target_delay=torch.tensor([5]),
        profiles=torch.tensor([-1]),
        amplitude=torch.tensor([[0.0]]),
        caps=torch.tensor([[40, 40, 4.8, 40, 40, 4.8]]),
    )
    e.physics_dt, e.step_dt = 0.001, 0.01
    e.motion_height_range_m = (0.30, 0.40)
    e._get_observations = e.observations
    e.set_grouped_evaluation_targets = lambda v: e.target_xy.copy_(v)
    e.set_evaluation_command = lambda v: set_evaluation_command(e, v)
    e.command = torch.tensor([[0.0, 0.0, 0.34]])
    e.ids = list(range(6))
    e.robot = NS(
        data=NS(
            joint_pos=NS(torch=torch.arange(6).float()[None]),
            joint_vel=NS(torch=torch.zeros(1, 6)),
            root_state_w=NS(torch=torch.zeros(1, 13)),
        )
    )
    v4 = dict(version="scripted_physics_io_boundary_no_engine")
    path = tmp_path / "V4.json"
    write_json(path, v4)
    e.motion_timing = NS(
        contract=v4,
        dr=e.identified_dr,
        last_external=torch.zeros(1, 6),
        base=NS(
            motor_ring=NS(delay=torch.tensor([[0, 0, 7, 0, 0, 11]]), tau=torch.zeros(1, 6)),
            motor_actual=torch.zeros(1, 6),
            adopted=torch.zeros(1, 6),
            last_pd_sensors=torch.zeros(1, 12),
        ),
    )
    env.output_contract = source["output_contract"]
    env.template = dict(
        ground={"version": "flat"},
        motion_identified_response=dict(contract=str(path), contract_sha256=file_sha256(path)),
    )
    inference = spec_for_job(job, env.output_contract)
    protocol = dict(
        input_contract=inference.input_contract,
        output_contract=inference.output_contract,
        initial_state=dict(kind="model_bound_reset", seed=job.runtime.seed),
        V4=dict(path=str(path), sha256=file_sha256(path)),
        timing=dict(
            physics_ms=1,
            PD_ms=5,
            policy_ms=10,
            actor_packet_ms=20,
            actor_leg_packet_ms=20,
            target_ms=5,
            wheel_transport_ms=[7, 11],
            wheel_tau_ms=[0, 0],
        ),
        scene={"version": "flat"},
        command=[0.25, 0, 0.34],
        motor_caps=[40, 40, 4.8, 40, 40, 4.8],
    )
    if mismatch:
        if mismatch == "timing":
            protocol["timing"]["actor_packet_ms"] = 10
        elif mismatch == "V4":
            protocol["V4"]["sha256"] = "a" * 64
        else:
            protocol["initial_state"]["seed"] += 1
        with pytest.raises(ValueError):
            env.configure_diagnostic(protocol)
        assert env.cumulative_physics_steps == 0
    else:
        actual = env.configure_diagnostic(protocol)
        assert actual["actual_timing"]["wheel_transport_ms"] == [7, 11]
        result = env.step(torch.zeros(1, 6))
        assert result["extras"]["diagnostic"]["q"].tolist() == [[0, 1, 2, 3, 4, 5]]
        assert e.robot.data.joint_pos.torch.tolist() == [[999] * 6]
        assert result["extras"]["actual_physics_steps"] == [10]
    env.close()
