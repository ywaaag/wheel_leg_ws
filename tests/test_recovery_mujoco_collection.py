"""CPU checks for the formal original-XML Recovery collection boundary."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from native_stand.artifacts.io import file_sha256
from native_stand.experiments.config import load_job_config
from native_stand.simulation.recovery_mujoco import RecoveryMuJoCoCollection, execution_factories

FROZEN_TEST_SCOPE = "PORTABLE"


def test_recipe_has_disjoint_real_dr_conditions_and_rise_first_selection():
    job = load_job_config(Path(__file__).parents[1] / "configs/jobs/recovery2100_mujoco_rise.json")
    assert job.supervision.selection_metrics == ("rise_success_rate", "continuous_10s_rate")
    cases = job.assets["recovery_mujoco"]["cases"]
    seeds = [{c["seed"] for c in cases[role]} for role in ("train", "development", "holdout")]
    assert not seeds[0] & seeds[1] and not seeds[0] & seeds[2] and not seeds[1] & seeds[2]
    for role in cases:
        assert len(cases[role]) == job.runtime.envs
        assert any(c["DR_strength"] > 0 for c in cases[role])
    assert job.control.physics_dt_s == 0.0005
    assert job.supervision.initial_student is not None
    assert "Actor-only" in job.experiment_variables["initialization_boundary"]


def test_execution_factory_verifies_sources_before_loading_and_uses_relative_package(tmp_path):
    sources = {}
    for name in (
        "execution",
        "lifecycle",
        "legacy_lifecycle",
        "preentry_lifecycle",
        "settle_lifecycle",
    ):
        path = tmp_path / (name + ".py")
        path.write_text(
            "class Execution: pass\n" if name == "execution" else "class Lifecycle: pass\n"
        )
        sources[name] = dict(path=str(path), sha256=file_sha256(path))
    ex, life = execution_factories(sources)
    assert ex.__module__.endswith(".execution") and life.__module__.endswith(".settle_lifecycle")
    # A cached module must not allow changed external source bytes.
    Path(sources["execution"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        execution_factories(sources)


def test_partial_engine_failure_preserves_exact_executed_substeps(monkeypatch):
    env = RecoveryMuJoCoCollection.__new__(RecoveryMuJoCoCollection)
    env.num_envs = 1
    data = SimpleNamespace(
        qpos=np.zeros(6),
        qvel=np.zeros(6),
        ctrl=np.zeros(6),
        qfrc_applied=np.zeros(6),
        qfrc_actuator=np.zeros(6),
    )
    env.data = [data]
    env.models = [SimpleNamespace()]
    env.refs = dict(
        qpos=np.arange(6),
        dofs=np.arange(6),
        actuators=np.arange(6),
        gears=np.ones(6),
        joints=list(range(6)),
    )
    env.catalog = dict(
        mechanics=dict(
            damping={str(i): 0.0 for i in range(6)},
            friction_loss_Nm={str(i): 0.0 for i in range(6)},
        )
    )
    submitted = []
    env.execution = SimpleNamespace(
        submit=lambda action: submitted.append(action.clone()),
        step=lambda q, v: (torch.zeros(1, 6), torch.zeros(1, 6)),
    )
    calls = []

    def step(m, d):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("scripted integrator failure")

    env.mj = SimpleNamespace(
        mj_name2id=None,
        mj_id2name=lambda m, t, j: str(j),
        mjtObj=SimpleNamespace(mjOBJ_JOINT=0),
        mj_step=step,
    )
    env.observations = lambda: None
    action = torch.tensor([[0.2, 0.3, 4.0, 0.2, 0.3, 5.0]])
    with pytest.raises(RuntimeError, match="integrator failure") as caught:
        env.step(action)
    assert caught.value.actual_physics_steps.tolist() == [1]
    torch.testing.assert_close(submitted[0], action)


def test_deployment_binds_teacher_and_execution_without_copying_physics_binary():
    from native_stand.artifacts.deployment import job_dependencies

    job = load_job_config(Path(__file__).parents[1] / "configs/jobs/recovery2100_mujoco_rise.json")
    dependencies = job_dependencies(job)
    roles = {r["role"] for r in dependencies}
    assert {
        "teacher_checkpoint",
        "V4_contract",
        "V4_tensor",
        "execution_execution",
        "execution_settle_lifecycle",
    } <= roles
    assert all(not r["path"].endswith(".mjb") for r in dependencies)


def test_actual_C_network_and_shared_decoder_match_physical_actor(tmp_path):
    from native_stand.artifacts.c_actor import export_recovery_actor
    from native_stand.learning.ppo.recovery_cubic import PhysicalActor

    contract = dict(
        center_rad=[-0.13, -0.15, 0.13, -0.15], leg_lower_rad=[-1.5] * 4, leg_upper_rad=[1.5] * 4
    )
    with torch.random.fork_rng():
        torch.manual_seed(73)
        actor = PhysicalActor(contract, hidden_dims=(16, 8, 4)).eval().requires_grad_(False)
        x = torch.randn(23, 31) * 0.2
    x[:, 24] = 0.68
    with torch.no_grad():
        y = actor.physical_mean(x).numpy()
    result = export_recovery_actor(actor, tmp_path, x.numpy(), y)
    assert result["mean_max_abs_error"] < 1e-5
    assert result["leg_max_abs_error_rad"] < 5e-5
    assert result["wheel_max_abs_error_rad_s"] < 1e-3
    assert result["scratch_array_bytes"] == 2 * 31 * 4
    assert result["hardware"] == "OFF"
