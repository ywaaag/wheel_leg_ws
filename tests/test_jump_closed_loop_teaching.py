"""Supervised input/action boundaries; no engine or PPO."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from native_stand.learning.distillation.jump_labels import motion_teacher_observation
from native_stand.learning.distillation.routing import ActionLabels, Teacher, label
from native_stand.simulation.jump_teaching import collection_commands

FROZEN_TEST_SCOPE = "PORTABLE"


def test_motion_projection_keeps_actual_physical_previous_and_replaces_only_flags():
    x = torch.randn(8, 37)
    saved = x.clone()
    y = motion_teacher_observation(x, "jump37_to_motion31_v1")
    assert torch.equal(y[:, :25], x[:, :25])
    assert torch.equal(y[:, 25:], y.new_tensor([1, 0, 0, 1, 0, 0]).expand(8, -1))
    assert torch.equal(x, saved)
    with pytest.raises(ValueError):
        motion_teacher_observation(x[:, :31], "jump37_to_motion31_v1")


def test_routing_projects_teacher_only_and_preserves_raw_student_labels():
    x = torch.randn(4, 37)
    x[:, 24] = 0.68
    x[:, 25:31] = x.new_tensor([0, 0, 1, 0, 0, 1])
    seen = []

    def mean(obs):
        seen.append(obs.clone())
        return obs.new_full((len(obs), 6), 0.5)

    teacher = Teacher(
        "motion",
        SimpleNamespace(mean=mean),
        ActionLabels("affine", (0.5, 0.5, 10, 0.5, 0.5, 10), (0,) * 6),
        0,
        {"safe": True},
        True,
        "jump37_to_motion31_v1",
    )
    labels, owner, valid = label(
        [teacher],
        x,
        {"safe": [True, False, True, False]},
        ActionLabels("affine", (0.5, 0.5, 10, 0.5, 0.5, 10), (0,) * 6),
    )
    assert seen[0].shape == (2, 31)
    assert torch.equal(labels[valid], torch.full((2, 6), 0.5))
    assert torch.isnan(labels[~valid]).all()
    assert owner.tolist() == [0, -1, 0, -1]


def test_collection_is_supervised_only_with_small_explicit_commands():
    job = SimpleNamespace(
        algorithm="distillation",
        supervision=object(),
        backend="isaac",
        runtime=SimpleNamespace(envs=6),
        assets={
            "jump_compute": "cuda_firmware_warp_inverse_v1",
            "jump_balance_collection": {
                "version": "jump_balance_student_collection_v1",
                "commands": [[0, 0, 0.34], [0.2, 0, 0.34], [0, -0.3, 0.34]],
            },
        },
    )
    rows = collection_commands(job)
    assert rows.shape == (6, 3)
    np.testing.assert_array_equal(rows[:3], rows[3:])
    job.algorithm = "ppo"
    with pytest.raises(ValueError, match="supervised-only"):
        collection_commands(job)


def test_declared_executor_transform_matches_torque_including_saturation():
    from native_stand.learning.distillation.jump_labels import match_jump_executor

    generator = torch.Generator().manual_seed(181)
    q = torch.randn(100, 4, generator=generator) * 0.1
    dq = torch.randn(100, 4, generator=generator) * 5
    original = torch.randn(100, 6, generator=generator) * 0.1
    spring = torch.randn(100, 4, generator=generator) * 10
    conditions = {"jump_spring_model_nm_" + str(i): spring[:, i] for i in range(4)}
    corrected = match_jump_executor(
        original,
        conditions,
        torch.ones(100, dtype=torch.bool),
        "motion_no_C_spring_to_jump_C_spring_v1",
    )
    before = (40 * (original[:, [0, 1, 3, 4]] - q) - 2 * dq).clamp(-40, 40)
    after = (40 * (corrected[:, [0, 1, 3, 4]] - q) - 2 * dq + spring).clamp(-40, 40)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-6)
    assert torch.equal(original[:, [2, 5]], corrected[:, [2, 5]])
    with pytest.raises(ValueError, match="no clipping"):
        match_jump_executor(
            original + 100,
            conditions,
            torch.ones(100, dtype=torch.bool),
            "motion_no_C_spring_to_jump_C_spring_v1",
        )


def test_supervision_request_does_not_reserve_or_settle_ppo_course():
    from native_stand.simulation.jump_teaching import teaching_request
    from native_stand.tasks.jump.adaptive_course import AdaptiveCourse

    course = AdaptiveCourse(seed=8, plant_id="test", DR_id="nominal")
    before = course.state_dict()
    request = teaching_request()
    next_request = teaching_request(request)
    assert request["kind"] == "landing_only" and request["drop_m"] == 0
    assert next_request["request_id"] == 2
    assert course.state_dict() == before


def test_inactive_zero_actor_is_uncovered_even_if_physical_contact_is_safe():
    from native_stand.learning.distillation.jump_labels import motion_teacher_eligible

    x = torch.zeros(2, 37)
    x[0, 24] = 0.68
    x[0, 25:31] = x.new_tensor([0, 0, 1, 0, 0, 1])
    assert motion_teacher_eligible(x, "jump37_to_motion31_v1").tolist() == [True, False]


def test_bc_inference_contract_normalizes_default_activation(tmp_path):
    from native_stand.learning.inference import InferenceSpec, build_actor, load_model
    from native_stand.artifacts.io import file_sha256

    record = dict(
        kind="direct",
        input_dim=37,
        hidden_dims=[8, 4],
        output_dim=6,
        input_contract={"version": "test"},
        output_contract={"mean_activation": "linear"},
    )
    spec = InferenceSpec.from_record(record)
    actor = build_actor(spec)
    path = tmp_path / "student.pt"
    torch.save({"actor": actor.state_dict(), "inference_spec": record}, path)
    loaded = load_model(path, spec, sha256=file_sha256(path))
    x = torch.randn(5, 37)
    torch.testing.assert_close(loaded.mean(x), actor({"student_actor": x}))
    record["activation"] = "ReLU"
    torch.save({"actor": actor.state_dict(), "inference_spec": record}, path)
    with pytest.raises(ValueError, match="activation"):
        load_model(path, spec, sha256=file_sha256(path))
