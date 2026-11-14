"""Physical support demonstration contract: shared tensor/numpy semantics."""

import copy

import numpy as np
import pytest
import torch
from test_motion_wide_distillation import contracts

from native_stand.learning.distillation.routing import ActionLabels
from native_stand.simulation.motion_support_probe import transform_for
from native_stand.tasks.motion.support_teacher import VERSION, numpy_shift, tensor_shift, validate

FROZEN_TEST_SCOPE = "PORTABLE"


def shift():
    return dict(
        version=VERSION,
        height_m=[0.2, 0.24, 0.4],
        offset_rad=[[0.18, 0.18, -0.18, 0.18], [0, 0, 0, 0], [0, 0, 0, 0]],
        evidence_sha256="a" * 64,
    )


def test_teacher_shadow_labels_match_actual_collection_without_input_mutation(tmp_path):
    old, student = contracts(tmp_path)
    # Fixture old center originally covers .3-.4; extend its boundary table explicitly.
    for c in (old, student):
        c["height_command_range_m"][0] = 0.2
        c["support_offset_table"]["height_m"][0] = 0.2
    heights = torch.linspace(0.2, 0.4, 41)
    x = torch.zeros(41, 31)
    x[:, 24] = heights * 2
    saved = x.clone()
    raw = torch.full((41, 6), 0.7)
    original = ActionLabels("motion_contract", contract=old)
    adapted = ActionLabels("motion_support_teacher", contract=old, support_shift=shift())
    source = original.physical(raw, x)
    physical = adapted.physical(raw, x)
    for i, h in enumerate(heights):
        case = dict(leg_scale_rad=0.15)
        expected = transform_for(case, support_shift=shift())(
            raw[i].numpy(), float(h), source[i].numpy()
        )
        np.testing.assert_allclose(physical[i].numpy(), expected, atol=1e-6)
    student_codec = ActionLabels("motion_contract", contract=student)
    labels = student_codec.labels(physical, x)
    torch.testing.assert_close(student_codec.physical(labels, x), physical)
    torch.testing.assert_close(x, saved)
    assert torch.equal(physical[:, [2, 5]], source[:, [2, 5]])
    np.testing.assert_allclose(
        tensor_shift(heights, shift()).numpy(),
        np.array([numpy_shift(float(h), shift()) for h in heights]),
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("height_m", [0.2, 0.2, 0.4]),
        ("offset_rad", [[0.9] * 4, [0] * 4, [0] * 4]),
        ("evidence_sha256", "missing"),
    ],
)
def test_reject_malformed_support_contract(field, value):
    c = copy.deepcopy(shift())
    c[field] = value
    with pytest.raises(ValueError):
        validate(c)
    with pytest.raises(ValueError):
        ActionLabels("motion_support_teacher", support_shift=c)


def test_shift_cannot_silently_change_student_decoder():
    with pytest.raises(ValueError):
        ActionLabels("motion_contract", support_shift=shift())
    with pytest.raises(ValueError):
        tensor_shift(torch.tensor([0.19]), shift())


def test_public_teacher_routing_records_shift_identity_and_rejects_student_inverse(tmp_path):
    from types import SimpleNamespace

    from native_stand.learning.distillation.routing import Teacher, label

    old, wide = contracts(tmp_path)
    for c in (old, wide):
        c["height_command_range_m"][0] = 0.2
        c["support_offset_table"]["height_m"][0] = 0.2
    x = torch.zeros(4, 31)
    x[:, 24] = torch.tensor([0.4, 0.44, 0.48, 0.68])
    codec = ActionLabels("motion_support_teacher", contract=old, support_shift=shift())
    model = SimpleNamespace(mean=lambda obs: torch.zeros(len(obs), 6))
    teacher = Teacher("low", model, codec, 0, {})
    target = ActionLabels("motion_contract", contract=wide)
    converted, owners, valid = label([teacher], x, {}, target, student_kind="tanh")
    torch.testing.assert_close(target.physical(converted, x), codec.physical(torch.zeros(4, 6), x))
    assert owners.tolist() == [0] * 4 and valid.all()
    with pytest.raises(ValueError, match="inverse"):
        codec.labels(torch.zeros(4, 6), x)


def test_supervision_preflight_rejects_shift_before_owner_creates_run(tmp_path):
    from dataclasses import replace

    from test_public_migration import supervision_job

    from native_stand.experiments.supervision import preflight

    job = supervision_job(tmp_path)
    invalid = shift()
    invalid["offset_rad"][0][0] = 2
    t = replace(
        job.supervision.teachers[0],
        labels=dict(kind="motion_support_teacher", support_shift=invalid),
    )
    job = replace(job, supervision=replace(job.supervision, teachers=(t,)))
    with pytest.raises(ValueError, match="bounded"):
        preflight(job)
    assert not (tmp_path / "runs").exists()
