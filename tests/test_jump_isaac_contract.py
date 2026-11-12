"""No engine imports: actual C/V4 source, GPU Critic layout and policy state."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from fixtures.jump_cpu_io import mapped_job
from fixtures.jump_isaac_schema import schema
from test_jump_cpu_reuse import no_physics_or_cuda  # noqa: F401
from test_jump_unified import observation

from native_stand.artifacts.checkpoint import load_actor_source
from native_stand.artifacts.io import json_digest
from native_stand.learning.policies.factory import build_policy
from native_stand.simulation.dispatch import execution_service, require_supported
from native_stand.simulation.jump_isaac_bridge import InverseFeedback
from native_stand.tasks.jump.isaac_contract import TRAINING

FROZEN_TEST_SCOPE = "HOST"
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("WHEEL_LEG_DATA_ROOT", "/home/kellen/wheel_leg_training_data"))


def test_actual_C_V4_layout_matches_frozen_Isaac_schema_and_has_no_solver_padding(tmp_path):
    job = mapped_job(ROOT, DATA, name="jump_isaac_raw37_wheel48")
    actual = schema(job, tmp_path)
    assert json_digest(actual) == json_digest(job.assets["jump_critic_schema"])
    assert actual["width"] == 5591 and len(actual["fields"]) == 226
    assert all(not f["name"].startswith(("physics.", "model.")) for f in actual["fields"])
    assert any(f["name"] == "response.queue" for f in actual["fields"])
    assert execution_service(job)["module"] == "native_stand.simulation.jump_backend"
    with pytest.raises(ValueError, match="独立评估"):
        require_supported(job, "evaluate")


def test_GPU_policy_initialization_uses_same_distilled_Actor_without_BC_optimizer():
    job = mapped_job(ROOT, DATA, name="jump_isaac_raw37_wheel48")
    obs = observation(1)
    obs["student_critic"] = torch.zeros(1, 5591)
    source = load_actor_source(job.initialization.checkpoint)
    policy = build_policy(job, obs, device="cpu", source=source)
    for key, value in source.payload["actor"].items():
        assert torch.equal(policy["ppo"].actor.state_dict()[key], value)
    assert not policy["ppo"].optimizer.actor.state and not policy["ppo"].optimizer.critic.state
    saved = policy["capture"]({"updates": 0})
    assert saved["version"] == TRAINING and saved["critic_schema"]["width"] == 5591
    policy["restore"](saved)
    bad = dict(saved, version="jump_unified_raw37_wheel48_v1")
    with pytest.raises(ValueError):
        policy["restore"](bad)


def test_inverse_feedback_uses_measured_state_and_no_contact_signal():
    from native_stand.artifacts.io import read_json
    from native_stand.simulation.jump_shared_c import State

    job = mapped_job(ROOT, DATA)
    refs = job.assets["jump_bindings"]
    description = read_json(refs["description"]["path"])
    mechanics = read_json(refs["mechanics"]["path"])["mechanics"]
    inverse = InverseFeedback(
        description,
        read_json(refs["spring_geometry"]["path"])["springs"],
        mechanics,
        read_json(refs["V4"]["path"]),
    )
    state = State()
    state.adopted_sequence = 1
    q = np.array([inverse.initial.q[n] for n in mechanics["joint_names"]])
    joint = np.stack((q, np.zeros_like(q), np.zeros_like(q)))
    root = np.array(
        [0, 0, inverse.initial.root_support_height_m, 2**-0.5, 0, 0, 2**-0.5, 0, 0, 0, 0, 0, 0]
    )
    actual = inverse.input(
        tick=10,
        state=state,
        root=root,
        joint=joint,
        acceleration=np.zeros(6),
        motor=np.zeros(6),
        NET=np.zeros(6),
        command=[0, 0, 0.34],
        request=False,
    )
    assert actual.support_mask == 3
    assert np.isfinite(actual.support_J).all()
    assert np.linalg.norm(actual.accel) == pytest.approx(9.81)
    assert actual.leg_length[:] == pytest.approx([0.28, 0.28], abs=1e-8)
