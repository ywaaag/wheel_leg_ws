"""Real Recovery reset/adapter consumers with scripted physical I/O; zero integration."""

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from native_stand.learning.ppo.performance import Measurement
from native_stand.simulation.recovery_isaac.observations import physical_profile
from native_stand.simulation.recovery_isaac.physics import PhysicalParameters
from native_stand.simulation.recovery_isaac.reset import InitialStates
from native_stand.tasks.recovery.cubic_randomization import draw
from native_stand.tasks.recovery.phase_readback import PhaseReadback
from native_stand.tasks.recovery.swing_priority_course import RecoveryCourse

ROOT = Path(__file__).resolve().parents[1]


class View:
    """Full-buffer material API, compact indexed mass/COM/inertia setters."""

    def __init__(self, n, bodies=3, shapes=4):
        self.mass = torch.ones(n, bodies)
        self.inertia = torch.eye(3).reshape(1, 1, 9).expand(n, bodies, -1).clone()
        self.com = torch.zeros(n, bodies, 7)
        self.com[:, :, 6] = 1
        self.material = torch.full((n, shapes, 3), 0.5)
        self.calls = []

    def get(self, name):
        self.calls.append(("get", name, getattr(self, name).numel()))
        return getattr(self, name)

    def get_masses(self):
        return self.get("mass")

    def get_inertias(self):
        return self.get("inertia")

    def get_coms(self):
        return self.get("com")

    def get_material_properties(self):
        return self.get("material")

    def set_material_properties(self, data, ids):
        assert data.shape == self.material.shape  # Compact material rows forbidden.
        self.calls.append(("set", "material", data.numel()))
        self.material[ids.long()] = data[ids.long()]


class Robot:
    def __init__(self, n):
        self.body_names = ["root", "L", "R"]
        self.joint_names = [f"q{i}" for i in range(6)]
        self.root_view = View(n)
        self.q = torch.zeros(n, 6)
        self.dq = self.q.clone()
        self.root = torch.zeros(n, 7)
        self.velocity = torch.zeros(n, 6)
        self.body_write_order = []

    def set_masses_index(self, masses, body_ids, env_ids):
        self.body_write_order.append("mass")
        self.root_view.mass[env_ids.long()] = masses

    def set_coms_index(self, coms, body_ids, env_ids):
        self.body_write_order.append("COM")
        self.root_view.com[env_ids.long()] = coms
        self.root_view.inertia[env_ids.long()] += 5  # Emulate COM-induced inertia mutation.

    def set_inertias_index(self, inertias, body_ids, env_ids):
        self.body_write_order.append("inertia")
        self.root_view.inertia[env_ids.long()] = inertias

    def write_joint_armature_to_sim_index(self, **kw):
        pass

    def write_joint_damping_to_sim_index(self, **kw):
        pass

    def write_joint_friction_coefficient_to_sim_index(self, **kw):
        pass

    def write_root_link_pose_to_sim_index(self, root_pose, env_ids):
        self.root[env_ids] = root_pose

    def write_root_com_velocity_to_sim_index(self, root_velocity, env_ids):
        self.velocity[env_ids] = root_velocity

    def write_joint_position_to_sim_index(self, position, env_ids):
        self.q[env_ids] = position

    def write_joint_velocity_to_sim_index(self, velocity, env_ids):
        self.dq[env_ids] = velocity


class BaseIO:
    def _reset_idx(self, ids):
        self.reset_count[ids] += 1


def reset_class(monkeypatch, module=None):
    # Only Isaac/asset I/O imports substituted. Task/reward/course/math stay real.
    math = ModuleType("isaaclab.utils.math")
    math.quat_apply = math.quat_apply_inverse = lambda *args: None
    pxr = ModuleType("pxr")
    pxr.Usd, pxr.UsdGeom, pxr.UsdPhysics = NS(), NS(), NS()
    native = ModuleType("native_stand.env")
    native.NativeStandEnv = BaseIO
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "native_stand.env", native)
    if module is None:
        from native_stand.simulation.recovery_isaac import environment as module
    return module.environment_class({}, {}, None, None, [])


def baseline_module(tmp_path, name):
    """Reference is immutable Git source, never old production modules."""
    path = tmp_path / (name + ".py")
    path.write_bytes(
        subprocess.check_output(
            ["git", "show", f"16f81c8:src/native_stand/simulation/recovery_isaac/{name}.py"],
            cwd=ROOT,
        )
    )
    spec = importlib.util.spec_from_file_location(
        "native_stand.simulation.recovery_isaac.reference_" + name, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_env(cls, *, n=8, physics_class=PhysicalParameters, execution_class=None):
    from native_stand.simulation.recovery_isaac.execution import Execution
    from native_stand.simulation.recovery_isaac.preentry_lifecycle import Lifecycle
    from native_stand.simulation.v4_response import Response

    e = object.__new__(cls)
    e.num_envs, e.device, e.recovery_ready, e.height_conditioned, e.handover_mode = (
        n,
        "cpu",
        True,
        True,
        False,
    )
    e.reset_measurement = Measurement(True)
    e.cfg = NS(seed=111101, ground_dynamic_friction=0.5, recovery_probe_cases=None)
    e.robot = Robot(n)
    e.ids = list(range(6))
    e.effort = torch.zeros(n, 6)
    e.scene = NS(env_origins=torch.stack((torch.arange(n), torch.zeros(n), torch.zeros(n)), -1))
    mechanics = dict(
        joint_names=e.robot.joint_names,
        **{
            key: {name: 0.1 for name in e.robot.joint_names}
            for key in ("armature", "damping", "friction_loss_Nm")
        },
    )
    contract = dict(
        sides={
            s: dict(parameters={"leg_mass_inertia_scale": 1.0}, body_changes=[]) for s in ("L", "R")
        }
    )
    e.physics = physics_class(e, contract, mechanics)
    names = [f"p{i}" for i in range(13)]
    params = [1.0, 0.1, 0.2, 4.0, 6.0, 1.0, -1.0, 2.0, 3.0, 0.4, 0.0, 0.0, 0.0]
    row = dict(
        parameters=dict(zip(names, params, strict=False)),
        effective_observer=dict(
            transport_ms=8, q_age_ms=2, v_age_ms=4, q_reference=[0.0, 0.0], velocity_tau_ms=3.0
        ),
    )
    execution_class = execution_class or Execution
    e.execution = execution_class(
        dict(parameter_names=names, sides={"L": row, "R": row}),
        Response,
        n,
        effort_caps=[40, 40, 4.8, 40, 40, 4.8],
    )
    e.life = Lifecycle(n)
    e.life.effort_caps = e.execution.caps
    e.life.height_conditioned = True
    e.life.command_height = torch.full((n,), 0.34)
    e.phase_readback = PhaseReadback(n, "cpu")
    poses = {}
    for name, h in [("Swing", 0.18), ("standing", 0.34)] + [
        (f"standing_h{h:.2f}", h) for h in (0.3, 0.32, 0.34, 0.36, 0.38)
    ]:
        poses[name] = dict(
            joint_position={k: float(i) * 0.01 for i, k in enumerate(e.robot.joint_names)},
            joint_velocity={k: 0.0 for k in e.robot.joint_names},
            root_pose_xyzw=[0.0, 0.0, h, 0.0, 0.0, 0.0, 1.0],
            root_velocity_linear_angular=[0.0] * 6,
        )
    e.init_catalog = {"poses": poses}
    e.initial_states = InitialStates(e.init_catalog, e.robot.joint_names, "cpu")
    e.courses = [RecoveryCourse(111101 + i) for i in range(n)]
    e.DR_rng = [np.random.default_rng(281101 + i) for i in range(n)]
    e.draws = [draw(0.0, rng) for rng in e.DR_rng]
    e.episode_spec = [None] * n
    e.completed = []
    e.completed_count = 0
    for name in ("reset_count", "episode_steps", "HP10_count"):
        setattr(e, name, torch.zeros(n, dtype=torch.long))
    for name in ("prefix_s", "student_seconds", "teacher_seconds", "pulse_onset", "height_initial"):
        setattr(e, name, torch.zeros(n))
    for name in ("initialized", "pulse_enabled", "height_dynamic"):
        setattr(e, name, torch.zeros(n, dtype=torch.bool))
    e.pulse_pattern = torch.zeros(n, 4)
    e.HP10_history = torch.zeros(500, n, 4)
    e.HP10_sum = torch.zeros(n, 4)
    e.ground_height = lambda xy: torch.zeros(len(xy))
    e.sim = NS(forward=lambda: None)

    def closure():
        e.closure_each = torch.zeros(n, 2)

    e._closure = closure
    e.measured = lambda: setattr(
        e, "leg_geometry", torch.tensor([[0.3, 0.3, 0.0, 0.0]]).expand(n, -1)
    )
    return e


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, np.random.Generator):
        assert_equal(a.bit_generator.state, b.bit_generator.state)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, list | tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("ids", [None, [3], [5, 1], []])
def test_actual_reset_reference_equivalence(monkeypatch, tmp_path, ids):
    cls = reset_class(monkeypatch)
    old = reset_class(monkeypatch, baseline_module(tmp_path, "environment"))
    a = make_env(cls)
    b = make_env(old, physics_class=baseline_module(tmp_path, "physics").PhysicalParameters)
    # New empty reset intentionally is a no-op, unlike old needless scene work.
    seq = [None, ids, ids]
    for selected in seq:
        index = None if selected is None else torch.tensor(selected, dtype=torch.long)
        if selected == []:
            before = a.execution.state_dict()
            old_rng = [copy.deepcopy(r.bit_generator.state) for r in a.DR_rng]
            a._reset_idx(index)
            assert_equal(before, a.execution.state_dict())
            assert_equal(old_rng, [r.bit_generator.state for r in a.DR_rng])
            continue
        # Real completed episode/early failure summaries, no artificial course passes.
        for e in (a, b):
            e.episode_steps[:] = torch.arange(e.num_envs) + 1
            e.student_seconds[:] = e.episode_steps * 0.01
            e.life.success[:] = torch.arange(e.num_envs) % 2 == 0
            e.life.last_result = {"done": torch.ones(e.num_envs, dtype=torch.bool)}
            e.HP10_count[:] = 500
            e.HP10_sum[:] = 0
        a._reset_idx(index)
        b._reset_idx(index)
        assert_equal(a.completed, b.completed)
        assert_equal([vars(c) for c in a.courses], [vars(c) for c in b.courses])
        assert_equal(
            [r.bit_generator.state for r in a.DR_rng], [r.bit_generator.state for r in b.DR_rng]
        )
        assert_equal(a.execution.state_dict(), b.execution.state_dict())
        for name in ("q", "dq", "root", "velocity"):
            assert_equal(getattr(a.robot, name), getattr(b.robot, name))
        assert_equal(a.physics.state_dict(), b.physics.state_dict())
        assert_equal(a.life.state_dict(), b.life.state_dict())
        for name in (
            "height_initial",
            "height_dynamic",
            "pulse_pattern",
            "pulse_onset",
            "prefix_s",
            "reset_count",
        ):
            assert_equal(getattr(a, name), getattr(b, name))
        assert a.robot.body_write_order[-3:] == ["mass", "COM", "inertia"]
        assert_equal(
            physical_profile(a), baseline_module(tmp_path, "observations").physical_profile(b)
        )


def test_partial_profile_rebuild_and_shape_dtype_invalidation(monkeypatch):
    e = make_env(reset_class(monkeypatch))
    e._reset_idx(None)
    before = physical_profile(e).clone()
    identity = e._physical_profile_cache.data_ptr()
    e._reset_idx(torch.tensor([2]))
    assert e._physical_profile_dirty == {2}
    after = physical_profile(e)
    assert after.data_ptr() == identity
    other = [i for i in range(e.num_envs) if i != 2]
    assert torch.equal(before[other], after[other])
    e.effort = e.effort.double()
    assert physical_profile(e).dtype == torch.float64
    # Explicit complete load invalidation remains supported by existing backend.
    e._physical_profile_cache = None
    assert torch.equal(physical_profile(e).float(), after.float())


def test_readback_fault_not_hidden_by_work_buffer(monkeypatch):
    e = make_env(reset_class(monkeypatch))
    e._reset_idx(None)
    original = e.robot.root_view.get_coms
    e.robot.root_view.get_coms = lambda: original() + 1
    with pytest.raises(RuntimeError, match="COM"):
        e.physics.reset(torch.tensor([2]), [e.draws[2]], host_ids=[2])


def test_real_adapter_rollout_reset_loss_rng_two_adam_equal(monkeypatch, tmp_path):
    from tensordict import TensorDict
    from test_recovery_continuation import fixture

    from native_stand.simulation.driver import Backend
    from native_stand.simulation.recovery_adapter import RecoveryAdapter

    # Two models only differ in reset implementation; all policy/reward/course
    # consumers remain public implementations. Physical observations are scripted.
    old = reset_class(monkeypatch, baseline_module(tmp_path, "environment"))
    new = reset_class(monkeypatch)
    environments = [
        make_env(new),
        make_env(old, physics_class=baseline_module(tmp_path, "physics").PhysicalParameters),
    ]

    class IO:
        infinite_horizon = False

        def __init__(self, env):
            self.env = env
            self.num_envs = env.num_envs
            self.count = 0
            self.raw = []
            self.rewards = []
            self.terminals = []
            env.cfg.recovery_critic_version = "recovery_standalone_height_causal909_v5"
            env.cfg.recovery_preentry_reward = True
            env.reset_measurement.enabled = False
            env._reset_idx(None)
            env.C0, env.C0_center = None, None
            env.HP10_cursor = 0
            env.policy_raw = torch.zeros(env.num_envs, 6)
            env.control_physical = env.policy_raw.clone()
            env.student_mask = torch.ones(env.num_envs, dtype=torch.bool)
            env.last_external = env.policy_raw.clone()
            env.last_passive = env.policy_raw.clone()
            env.robot.data = NS(applied_torque=NS(torch=env.policy_raw.clone()))
            env.actual_physics_steps = 0

            def measured():
                q, dq = env.robot.q, env.robot.dq
                state = torch.zeros(env.num_envs, 13)
                state[:, 3] = env.robot.root[:, 2]
                state[:, 9:11] = 3
                state[:, 12] = 1
                frame = torch.zeros(env.num_envs, 16)
                frame[:, 4] = -1
                frame[:, 6:10] = q[:, [0, 1, 3, 4]] / 1.5
                frame[:, 10:16] = dq * 0.05
                env.leg_geometry = torch.tensor([[0.3, 0.3, 0.0, 0.0]]).expand(env.num_envs, -1)
                return (
                    frame,
                    q,
                    dq,
                    state,
                    env.robot.root[:, :3],
                    torch.zeros(env.num_envs, 3),
                    torch.full((env.num_envs,), 0.15),
                    torch.zeros(env.num_envs, dtype=torch.bool),
                    torch.zeros(env.num_envs, 3),
                )

            env.measured = measured

        def observations(self):
            return TensorDict(self.env._get_observations(), batch_size=[self.num_envs])

        def step(self, raw):
            e = self.env
            self.count += 1
            self.raw.append(raw.clone())
            e._pre_physics_step(raw)
            for _ in range(10):
                e.execution.step(
                    e.robot.q, e.robot.dq
                )  # Causal tensor math only, no physics integrator.
            if self.count % 2 == 0:
                e.robot.root[0, 2] = 0.08  # Scripted early-fault observation.
            terminated, truncated = e._get_dones()
            self.terminals.append(e.terminal_critic.clone())
            assert e.terminal_critic.shape == (self.num_envs, 909)
            reward = e._get_rewards()
            self.rewards.append(reward.clone())
            ids = torch.where(terminated | truncated)[0]
            if len(ids):
                e._reset_idx(ids)
            return dict(
                rewards=reward,
                terminated=terminated,
                truncated=truncated,
                dones=terminated | truncated,
                extras={"terminal_critic": e.terminal_critic},
            )

        def runtime_state(self):
            return dict(execution=self.env.execution.state_dict(), life=self.env.life.state_dict())

        def close(self):
            pass

    io = [IO(e) for e in environments]
    _, context, _ = fixture(tmp_path)
    import dataclasses

    context.job = dataclasses.replace(
        context.job, runtime=dataclasses.replace(context.job.runtime, envs=8, steps_per_env=4)
    )
    from native_stand.artifacts.io import file_sha256
    from native_stand.artifacts.runtime_paths import RuntimePaths
    from native_stand.experiments.config import load_job_config

    source_job = load_job_config(ROOT / "configs/jobs/recovery_preentry48_remaining.json")
    mapping = Path("/home/kellen/wheel_leg_training_data/runtime_paths.json")
    paths = RuntimePaths.from_assets(
        {"path_mapping": dict(path=str(mapping), sha256=file_sha256(mapping))}
    )
    context.job.assets.update(
        {
            name: paths.view(source_job.assets[name])
            for name in ("qualified_physical_BC_anchor", "old_Recovery_action_prior")
        }
    )
    seed = torch.get_rng_state().clone()
    adapters = []
    for item in io:
        torch.set_rng_state(seed)
        adapter = RecoveryAdapter(Backend("scripted IO only", lambda job, item=item: item))
        adapter.build(context, "formal", None)
        adapters.append(adapter)
    state = torch.get_rng_state().clone()
    outcomes = []
    for adapter in adapters:
        torch.set_rng_state(state)
        result = adapter.update_once(context, 7)
        outcomes.append((result, torch.get_rng_state().clone()))
    assert_equal(outcomes[0][1], outcomes[1][1])
    for key in ("actor", "critic"):
        assert_equal(
            getattr(adapters[0].ppo, key).state_dict(), getattr(adapters[1].ppo, key).state_dict()
        )
    assert_equal(adapters[0].ppo.optimizer.state_dict(), adapters[1].ppo.optimizer.state_dict())
    for name in ("actions", "returns", "advantages", "rewards", "values"):
        assert_equal(getattr(adapters[0].ppo.storage, name), getattr(adapters[1].ppo.storage, name))
    assert_equal(io[0].raw, io[1].raw)
    assert_equal(io[0].rewards, io[1].rewards)
    assert_equal(io[0].terminals, io[1].terminals)
    assert_equal(environments[0].completed, environments[1].completed)
    assert len(environments[0].completed) == 2
    assert_equal(adapters[0].ppo.action_prior_readback, adapters[1].ppo.action_prior_readback)
    assert adapters[0].ppo.action_prior_readback["total_queried_rows"] > 0


def test_inactive_rows_and_empty_reset_are_unchanged(monkeypatch):
    e = make_env(reset_class(monkeypatch))
    e._reset_idx(None)
    idx = torch.tensor([4, 1])
    untouched = [0, 2, 3, 5, 6, 7]
    physical = copy.deepcopy(e.physics.state_dict())
    course = [copy.deepcopy(vars(c)) for c in e.courses]
    rng = [copy.deepcopy(g.bit_generator.state) for g in e.DR_rng]
    phase = [e.phase_readback.record(i) for i in untouched]
    execution = e.execution.state_dict()
    e._reset_idx(idx)
    assert_equal([vars(e.courses[i]) for i in untouched], [course[i] for i in untouched])
    assert_equal([e.DR_rng[i].bit_generator.state for i in untouched], [rng[i] for i in untouched])
    assert [e.phase_readback.record(i) for i in untouched] == phase
    for name, value in physical.items():
        assert_equal(e.physics.state_dict()[name][untouched], value[untouched])
    new = e.execution.state_dict()
    for name, value in execution["tensors"].items():
        actual = new["tensors"][name]
        if value.ndim and value.shape[0] == e.num_envs:
            assert_equal(value[untouched], actual[untouched])
        elif value.ndim > 1 and value.shape[1] == e.num_envs:
            assert_equal(value[:, untouched], actual[:, untouched])
    calls = len(e.robot.root_view.calls)
    material_ptr = e.physics.material.data_ptr()
    e._reset_idx(torch.empty(0, dtype=torch.long))
    assert len(e.robot.root_view.calls) == calls
    assert e.physics.material.data_ptr() == material_ptr


def test_tensor_frontend_keeps_selected_rows_and_rejects_unknown():
    from native_stand.simulation.recovery_isaac.physics import value

    x = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    proxy = NS(torch=x)
    actual = value(proxy, "cpu", torch.tensor([7, 2]))
    assert torch.equal(actual, x[[7, 2]]) and actual.data_ptr() != x.data_ptr()
    # Complete state still owns independent storage.
    actual.fill_(0)
    assert x[7].sum() > 0


def test_physical_restore_reuses_full_material_buffer_and_invalidates_profile(monkeypatch):
    e = make_env(reset_class(monkeypatch))
    e._reset_idx(None)
    expected = e.physics.state_dict()
    physical_profile(e)
    e.physics.restore(expected)
    assert_equal(e.physics.state_dict(), expected)
    assert e._physical_profile_cache is None
    assert e.robot.body_write_order[-3:] == ["mass", "COM", "inertia"]


@pytest.mark.parametrize("rows", [[3, 1], [], [0, 1, 2, 3, 4, 5, 6, 7]])
def test_compact_execution_reset_and_future_queue_exact(tmp_path, monkeypatch, rows):
    a = make_env(reset_class(monkeypatch))
    reference = baseline_module(tmp_path, "execution").Execution
    b = copy.deepcopy(a.execution)
    b.__class__ = reference
    indices = torch.tensor(rows, dtype=torch.long)
    q = torch.linspace(-0.3, 0.3, len(rows) * 6).reshape(-1, 6)
    dq = q * 0.2
    a.execution.reset(indices, q, dq)
    b.reset(indices, q, dq)
    assert_equal(a.execution.state_dict(), b.state_dict())
    x = torch.zeros(8, 6)
    for tick in range(32):
        if tick % 10 == 0:
            target = x + tick * 0.001
            a.execution.submit(target)
            b.submit(target)
        assert_equal(a.execution.step(x, x), b.step(x, x))
    assert_equal(a.execution.state_dict(), b.state_dict())
