"""Real ABI4 and script physical I/O; fail on any real integration or CUDA."""

import ctypes as C
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
import torch
from fixtures.jump_cpu_io import mapped_job, script_physics, small_job

from native_stand.artifacts.io import file_sha256, json_digest
from native_stand.learning.policies.factory import build_policy
from native_stand.simulation import jump_backend, jump_build
from native_stand.simulation.jump_shared_c import array

FROZEN_TEST_SCOPE = "HOST"
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("WHEEL_LEG_DATA_ROOT", "/home/kellen/wheel_leg_training_data"))


@pytest.fixture(autouse=True)
def no_physics_or_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real integration/GPU forbidden in Jump CPU acceptance")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


@pytest.fixture
def job():
    return mapped_job(ROOT, DATA)


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif hasattr(a, "keys"):
        assert set(a.keys()) == set(b.keys())
        for key in list(a.keys()):
            equal(a[key], b[key])
    elif isinstance(a, tuple | list):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            equal(left, right)
    else:
        assert a == b


def test_one_build_per_job_and_rebuild_reuses_without_overwrite(tmp_path, job, monkeypatch):
    calls = []
    run = subprocess.run

    def track(command, *args, **kwargs):
        if "-shared" in command:
            calls.append(command)
        return run(command, *args, **kwargs)

    monkeypatch.setattr(jump_build.subprocess, "run", track)
    job = replace(job, runtime=replace(job.runtime, envs=3))
    batch = jump_backend.make_env(job, output=tmp_path / "job")
    first = batch.envs[0].plant.binding
    assert len(calls) == 1
    assert len({e.plant.binding["path"] for e in batch.envs}) == 1
    assert len({C.addressof(e.plant.execution.c.states) for e in batch.envs}) == 3
    assert len({e.plant.execution.v4.targets.data_ptr() for e in batch.envs}) == 3
    assert len({id(e.plant.model) for e in batch.envs}) == 3
    assert len({id(e.plant.data) for e in batch.envs}) == 3
    digest = file_sha256(first["path"])
    second = jump_backend.make_env(job, output=tmp_path / "job")
    assert len(calls) == 1 and file_sha256(first["path"]) == digest
    assert second.envs[0].plant.binding == first
    original = batch.envs[1].plant.model.actuator_forcerange.copy()
    batch.envs[0].plant.model.actuator_forcerange[0] = [-3, 3]
    np.testing.assert_array_equal(batch.envs[1].plant.model.actuator_forcerange, original)
    batch.close()
    second.close()


def test_build_identity_headers_flags_toolchain_and_failure(tmp_path, monkeypatch):
    path = tmp_path / "immutable.so"
    first = jump_build.build(path)
    identity = first["build_identity"]
    assert identity["ABI"] == 4 and "rl_jump_core.h" in identity["sources"]
    assert identity["toolchain"]["sha256"] and identity["toolchain"]["version"]
    with monkeypatch.context() as m:
        m.setattr(jump_build, "FLAGS", (*jump_build.FLAGS, "-fno-math-errno"))
        with pytest.raises(ValueError, match="不覆盖"):
            jump_build.build(path)
    assert file_sha256(path) == first["sha256"]
    real_run = subprocess.run

    def failure(command, *args, **kwargs):
        if "-shared" in command:
            raise subprocess.CalledProcessError(1, command, stderr="fixture compile error")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(jump_build.subprocess, "run", failure)
    with pytest.raises(subprocess.CalledProcessError):
        jump_build.build(tmp_path / "failed.so")
    assert not (tmp_path / "failed.so").exists()
    assert not (tmp_path / "failed.json").exists()
    assert not list(tmp_path.glob(".jump-build-*"))


def test_measurement_and_schema_cache_reset_restore_and_parameter_invalidation(tmp_path, job):
    batch = jump_backend.make_env(job, output=tmp_path)
    env = batch.envs[0]
    p = env.plant
    p.invalidate_measurements()
    with patch.object(p.geometry, "measure", wraps=p.geometry.measure) as shape:
        from native_stand.simulation.jump_unified_learning import packet

        first = packet(p, 0.001)
        p.readback()
        packet(p, 0.001)
        assert shape.call_count == 1
        p.physics_steps += 1
        packet(p, 0.001)
        assert shape.call_count == 2
        p.invalidate_measurements()
        packet(p, 0.001)
        assert shape.call_count == 3
    # Repeated reads are identity-reused. A parameter change keeps the exact
    # frozen layout but refreshes Critic values, and only in this environment.
    batch.invalidate_observations([0])
    saved = batch.runtime_state()
    obs = env.observations()
    assert env.observations() is obs
    schema = env.schema
    batch.invalidate_observations([0])
    equal(obs, env.observations())
    assert env.schema is schema and len(schema.fields) == 235
    original_mass = p.model.body_mass[p.root]
    p.model.body_mass[p.root] *= 1.01
    batch.invalidate_observations([0], parameters_changed=True)
    after = env.observations()
    assert not torch.equal(after["student_critic"], obs["student_critic"])
    assert after["student_critic"].shape[-1] == 6370
    assert json_digest(env.schema.contract()) == json_digest(job.assets["jump_critic_schema"])
    assert first.phase == 9
    # Restore uses the original model parameters: they are immutable job state,
    # not an undocumented component of the physical integration checkpoint.
    p.model.body_mass[p.root] = original_mass
    batch.invalidate_observations([0], parameters_changed=True)
    batch.load_runtime_state(saved)
    equal(batch.observations(), obs)
    batch.close()


def test_packet_math_matches_independent_full_readback(tmp_path, job):
    from scipy.spatial.transform import Rotation

    from native_stand.simulation.jump_unified_learning import packet

    batch = jump_backend.make_env(job, output=tmp_path)
    p = batch.envs[0].plant
    for sign in (-1, 0, 1):
        p.data.qvel[:] = sign * np.linspace(-0.1, 0.1, p.model.nv)
        mujoco.mj_forward(p.model, p.data)
        p.invalidate_measurements()
        actual = packet(p, 0.001)
        readback = p.readback()
        shape = p.geometry.measure(
            {n: float(p.data.qpos[p.model.jnt_qposadr[j]]) for n, j in p.refs.items()}
        )
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(p.model, p.data, mujoco.mjtObj.mjOBJ_BODY, p.root, velocity, 0)
        upright = (
            p.data.xmat[p.root].reshape(3, 3) @ Rotation.from_euler("x", -np.pi / 2).as_matrix()
        )
        equal(np.asarray(actual.leg_length_m), shape.length)
        equal(np.asarray(actual.spring_length_m), shape.spring)
        equal(np.asarray(actual.COM_velocity_m_s), p.COM_velocity())
        equal(np.asarray(actual.omega_rad_s), upright.T @ velocity[:3])
        equal(np.asarray(actual.tire_clearance_m), readback["clearance_m"])
        equal(np.asarray(actual.adopted_target), array(p.execution.c.states[0].adopted))
    batch.close()


def test_normal_short_fault_terminal_observation_and_other_environment_isolation(tmp_path, job):
    job = replace(job, runtime=replace(job.runtime, envs=2))
    batch = script_physics(jump_backend.make_env(job, output=tmp_path), fault_at=2)
    untouched = batch.envs[1].runtime_state()
    batch.reset([0])
    equal(batch.envs[1].runtime_state(), untouched)
    outcome = batch.step(torch.zeros(2, 6))
    assert outcome["extras"]["actual_physics_steps"] == [2, 2]
    assert not outcome["extras"]["transition_valid"].any()
    assert outcome["terminated"].all()
    assert all(row["jump_control_s"] == pytest.approx(0.002) for row in batch.completed)
    assert all(row["reason"] == "engineering_fault" for row in batch.completed)
    terminal = outcome["terminal_observations"]
    assert set(terminal.keys()) == set(outcome["observations"].keys())
    assert terminal["student_actor"].shape == (2, 37)
    assert terminal["student_critic"].shape == (2, 6370)
    assert not torch.equal(terminal["student_critic"], outcome["observations"]["student_critic"])
    script_physics(batch)
    normal = batch.step(torch.zeros(2, 6))
    assert normal["extras"]["actual_physics_steps"] == [10, 10]
    assert normal["extras"]["transition_valid"].all()
    assert normal["terminal_observations"] is normal["observations"]
    assert normal["extras"]["actual_C_owner"].tolist() == [2, 2]
    assert not normal["extras"]["control_owner"].any()
    batch.close()


def test_ring_boundaries_partial_reset_and_exact_vs_fresh_restore(tmp_path, job):
    job = replace(job, runtime=replace(job.runtime, envs=2))
    batch = script_physics(jump_backend.make_env(job, output=tmp_path / "initial"))
    for _ in range(14):
        result = batch.step(torch.zeros(2, 6))
        assert not result["dones"].any()
    assert batch.envs[0].plant.execution.tick == 140
    saved = batch.runtime_state()
    path = tmp_path / "runtime.pt"
    torch.save(saved, path)
    result = batch.step(torch.zeros(2, 6))
    continued = batch.runtime_state()
    batch.load_runtime_state(torch.load(path, weights_only=False))
    repeated = batch.step(torch.zeros(2, 6))
    equal(result, repeated)
    equal(continued, batch.runtime_state())
    other = batch.envs[1].runtime_state()
    batch.reset([0])
    equal(other, batch.envs[1].runtime_state())
    assert batch.envs[0].plant.execution.tick == 0
    assert batch.envs[1].plant.execution.tick == 150
    fresh = script_physics(
        jump_backend.make_env(
            replace(job, assets=dict(job.assets, jump_restore_mode="fresh_reset")),
            output=tmp_path / "fresh",
        )
    )
    fresh.load_runtime_state(saved)
    assert fresh.envs[0].reward.elapsed_s == 0
    assert fresh.envs[0].request["request_id"] != saved["environments"][0]["request"]["request_id"]
    assert len(fresh.envs[0].course.pending) == 1
    assert fresh.completed == saved["completed"]
    batch.close()
    fresh.close()


def test_policy_continuation_both_Adam_std_RNG_and_Critic_equal(tmp_path, job):
    from native_stand.students.resume import restore_rng, rng_state

    job = small_job(job)
    batch = script_physics(jump_backend.make_env(job, output=tmp_path / "training"))
    torch.manual_seed(11012)
    handle = build_policy(job, batch.observations())

    def update(handle):
        ppo = handle["ppo"]
        with torch.inference_mode():
            for _ in range(job.runtime.steps_per_env):
                obs = batch.observations()
                action = ppo.act(obs)
                out = batch.step(action)
                ppo.process_env_step(obs, out["rewards"], out["dones"], {})
            ppo.compute_returns(batch.observations())
        ppo.update()

    update(handle)
    state = handle["capture"]({"updates": 1}, execution_state=batch.runtime_state())
    state["fixture_rng"] = rng_state()
    path = tmp_path / "training_state.pt"
    torch.save(state, path)
    update(handle)
    continuous = handle["capture"]({"updates": 2}, execution_state=batch.runtime_state())
    continuous_rng = rng_state()
    saved = torch.load(path, weights_only=False)
    batch.load_runtime_state(saved["execution_state"])
    other = build_policy(job, batch.observations())
    other["restore"](saved)
    restore_rng(saved["fixture_rng"])
    update(other)
    resumed = other["capture"]({"updates": 2}, execution_state=batch.runtime_state())
    equal(continuous, resumed)
    equal(continuous_rng, rng_state())
    assert resumed["optimizer"]["actor"]["state"] and resumed["optimizer"]["critic"]["state"]
    batch.close()


def test_cuda_and_unimplemented_backend_remain_rejected(job):
    from native_stand.simulation.dispatch import require_supported

    with pytest.raises(ValueError):
        jump_backend.validate_job(replace(job, runtime=replace(job.runtime, device="cuda:0")))
    with pytest.raises(ValueError):
        require_supported(replace(job, backend="isaac"), "train")


def test_shared_measurement_initialization_reset_and_actual_C_call_counts(tmp_path, job):
    from native_stand.learning.ppo.performance import Measurement

    job = replace(job, runtime=replace(job.runtime, envs=2))
    timer = Measurement(True, "cpu")
    with timer.section("environment_construction", synchronize=False):
        batch = jump_backend.make_env(job, output=tmp_path, measurement=timer)
    assert timer.counters["jump_C_compilations"] == 1
    assert timer.counters["jump_environment_resets:full"] == 2
    assert timer.parents["jump_C_build"] == "environment_construction"
    assert all(e.plant.execution.c.measurement is timer for e in batch.envs)
    script_physics(batch)
    timer.reset()
    for _ in range(12):
        batch.step(torch.zeros(2, 6))
    assert timer.counters["jump_script_or_physics_ticks"] == 240
    assert timer.counters["jump_reward_packets"] == 240
    assert timer.counters["jump_C_FSM_calls"] == 48
    assert timer.counters["jump_C_PD_calls"] == 48
    assert timer.counters["jump_C_encode_calls"] == 24
    assert timer.counters["jump_C_publish_calls"] == 24
    assert timer.counters.get("jump_C_diagnostic_snapshots", 0) == 0
    assert timer.parents["jump_observation_pack"] == "jump_batch_step"
    assert set(timer.clocks.values()) == {"cpu_wall"}
    other = batch.envs[1].runtime_state()
    batch.reset([0])
    equal(other, batch.envs[1].runtime_state())
    assert timer.counters["jump_environment_resets:partial"] == 1
    batch.close()


def test_balance_candidates_count_early_failure_and_keep_short_unknown():
    from dataclasses import asdict

    from native_stand.simulation.jump_adapter import consider
    from native_stand.tasks.jump.adaptive_course import CourseConfig

    request = dict(kind="ground_active_jump", plant_id="p", DR_id="nominal", level=0, generation=0)
    good = dict(
        request=request, success=False, fault=False, balance_seconds=1.0, balance_good_seconds=1.0
    )
    failed = dict(good, fault=True, balance_seconds=0.002, balance_good_seconds=0.002)
    unknown = dict(good, balance_seconds=0.002, balance_good_seconds=0.002)
    index = dict(candidates_by_objective={"balance": {}, "landing_only": {}, "active_jump": {}})
    window = dict(selection=asdict(CourseConfig()), rows=[good] * 32 + [failed] * 2)
    reference = dict(update=100, checkpoint="fixture", sha256="fixture")
    assert not consider(index, window, reference)["candidates_by_objective"]["balance"]
    window["rows"] = [unknown] * 32
    assert not consider(index, window, reference)["candidates_by_objective"]["balance"]
    window["rows"] = [good] * 32
    assert consider(index, window, reference)["candidates_by_objective"]["balance"]


def test_pending_candidate_publication_is_idempotent_and_protects_checkpoint(tmp_path):
    import copy
    from dataclasses import asdict

    from native_stand.artifacts.model_refs import ModelIndex
    from native_stand.learning.ppo.performance import Measurement
    from native_stand.simulation.jump_adapter import JumpAdapter
    from native_stand.tasks.jump.adaptive_course import CourseConfig

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"scripted CPU checkpoint identity")
    reference = dict(checkpoint=str(checkpoint), sha256=file_sha256(checkpoint), update=100)
    row = dict(
        request=dict(kind="landing_only", plant_id="p", DR_id="nominal", level=0, generation=0),
        success=True,
        fault=False,
        balance_seconds=1.0,
        balance_good_seconds=1.0,
    )
    pending = dict(end_update=100, rows=[row] * 32, selection=asdict(CourseConfig()))
    index = ModelIndex(tmp_path / "models.json")
    context = SimpleNamespace(
        refs=index, run_dir=tmp_path, store=SimpleNamespace(update=lambda **k: None)
    )
    adapter = JumpAdapter(SimpleNamespace())
    adapter.measurement = Measurement(True, "cpu")
    adapter.candidates = dict(
        candidates_by_objective={"balance": {}, "landing_only": {}, "active_jump": {}}
    )
    adapter.pending = copy.deepcopy(pending)
    adapter.after_save(context, reference)
    first = index.candidate_index(tmp_path.name)
    assert str(checkpoint) in index.consumer_paths()
    assert first["candidates_by_objective"]["landing_only"]
    assert not first["candidates_by_objective"]["active_jump"]
    assert adapter.pending is None
    # A crash after index publication and before saving the cleared pending is
    # resumed with the original pending window; no second publication occurs.
    adapter.pending = copy.deepcopy(pending)
    adapter.after_save(context, reference)
    equal(first, index.candidate_index(tmp_path.name))


def test_MuJoCo_mass_matrix_old_and_new_API_with_actual_native_readback(monkeypatch):
    from native_stand.simulation.jump_support_model import full_mass

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="hinge"/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = np.empty((1, 1))
    full_mass(model, data, out)
    assert out[0, 0] == pytest.approx(0.004)
    calls = []
    monkeypatch.setattr(mujoco, "mj_fullM", lambda *args: calls.append(args))
    old = SimpleNamespace(qM=np.array([0.004]))
    new = SimpleNamespace(M=np.array([0.004]))
    full_mass(model, old, out)
    full_mass(model, new, out)
    assert calls[0][1] is out and calls[0][2] is old.qM
    assert calls[1][1] is new and calls[1][2] is out
