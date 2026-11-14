"""Wide Motion contract routing and actual shared C execution; no physics."""

import copy

import numpy as np
import pytest
import torch
from test_motion_production_integration import make_job

from native_stand.learning.distillation.routing import ActionLabels
from native_stand.simulation.control_core import ControlCore, build
from native_stand.tasks.motion.action_contract import ActionDecoder
from native_stand.tasks.motion.spec import WIDE_OUTPUT_VERSION

FROZEN_TEST_SCOPE = "PORTABLE"


def contracts(tmp_path):
    _, old = make_job(tmp_path, retention=False)
    wide = copy.deepcopy(old)
    wide.update(action_version=WIDE_OUTPUT_VERSION, leg_halfwidth_rad=0.5)
    return old, wide


def test_physical_labels_preserve_teacher_targets_at_all_heights(tmp_path):
    old, wide = contracts(tmp_path)
    x = torch.zeros(41, 31)
    x[:, 24] = torch.linspace(0.3, 0.4, 41) * 2
    raw = torch.linspace(-1, 1, 246).reshape(41, 6)
    teacher, student = (
        ActionLabels("motion_contract", contract=old),
        ActionLabels("motion_contract", contract=wide),
    )
    physical = teacher.physical(raw, x)
    labels = student.labels(physical, x)
    torch.testing.assert_close(
        labels[:, [0, 1, 3, 4]], raw[:, [0, 1, 3, 4]] * 0.3, atol=2e-7, rtol=1e-6
    )
    torch.testing.assert_close(student.physical(labels, x), physical)
    outside = physical.clone()
    outside[:, 0] += 2
    with pytest.raises(ValueError, match="outside"):
        student.labels(outside, x)
    forged = copy.deepcopy(old)
    forged["leg_halfwidth_rad"] = 0.5
    with pytest.raises(ValueError):
        ActionDecoder.from_contract(forged)


def test_actual_C_wide_torch_match_and_old_kind_rejects_forged_scale(tmp_path):
    old, wide = contracts(tmp_path)
    binding = build(tmp_path / "core.so")
    core = ControlCore(binding, wide)
    decoder = ActionDecoder.from_contract(wide)
    for h in np.linspace(0.3, 0.4, 17):
        raw = np.array([-1.4, -0.2, 0.3, 0.8, 1.3, -0.7], np.float32)
        wanted = decoder.decode(torch.from_numpy(raw[None]), torch.tensor([h]))["requested_target"][
            0
        ].numpy()
        np.testing.assert_allclose(core.decode(raw, h), wanted, atol=2e-6)
    legacy = ControlCore(binding, old)
    legacy.config.leg_scale = 0.5
    with pytest.raises(ValueError):
        legacy.decode(np.zeros(6, np.float32), 0.34)


def test_BC_candidate_carries_Student_contract_and_loads_normal_inference(tmp_path):
    from dataclasses import replace

    from native_stand.artifacts.checkpoint import load_actor_source
    from native_stand.artifacts.io import file_sha256, write_npz
    from native_stand.learning.distillation.loop import run_distillation
    from native_stand.learning.distillation.splits import EpisodeSplit
    from native_stand.learning.inference import build_actor, load_model, spec_for_job

    job, old = make_job(tmp_path, retention=False)
    wide = copy.deepcopy(old)
    wide.update(action_version=WIDE_OUTPUT_VERSION, leg_halfwidth_rad=0.5)
    job = replace(
        job,
        model=replace(job.model, action_version=WIDE_OUTPUT_VERSION),
        action=replace(job.action, version=WIDE_OUTPUT_VERSION, leg_scale_rad=0.5),
    )
    spec = spec_for_job(job, wide)
    teacher = tmp_path / "teacher.pt"
    torch.save({"actor": build_actor(spec).state_dict()}, teacher)
    states = np.zeros((30, 31), np.float32)
    states[:, 24] = 0.68
    labels = np.zeros((30, 6), np.float32)
    provenance = dict(
        teacher_sha256=file_sha256(teacher),
        input_version=job.model.interface,
        action_version=WIDE_OUTPUT_VERSION,
        action_units="policy_output",
        action_dim=6,
    )
    data = dict(
        states=states,
        episodes=np.repeat(np.arange(6), 5).tolist(),
        teacher_actions=labels,
        label_provenance=provenance,
    )
    dataset = tmp_path / "data.npz"
    write_npz(dataset, dict(states=states, episodes=np.array(data["episodes"])))
    result = run_distillation(
        job,
        dataset_path=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "fit",
        rounds=1,
        prepared=dict(
            data=data, split=EpisodeSplit((0, 1, 2), (3,), (4, 5)), student_spec=spec.record()
        ),
    )
    path = result["candidate"]["path"]
    payload = load_actor_source(path).payload
    assert payload["output_contract"] == wide
    assert payload["execution_timing"] == job.assets.get("delayed_execution")
    model = load_model(path, spec, sha256=file_sha256(path))
    assert model.mean(torch.from_numpy(states)).shape == (30, 6)
    latest = torch.load(result["latest_state"]["path"], weights_only=False)
    assert latest["output_contract"] == wide and latest["round"] == 1 and "optimizer" in latest


def test_wide_contract_requires_explicit_BC_scope_until_PPO_source_guard_is_adapted(tmp_path):
    from dataclasses import replace

    from native_stand.experiments.config import ConfigError
    from native_stand.tasks.motion.spec import build_task

    job, _ = make_job(tmp_path, retention=False)
    job = replace(
        job,
        action=replace(job.action, version=WIDE_OUTPUT_VERSION, leg_scale_rad=0.5),
        model=replace(job.model, action_version=WIDE_OUTPUT_VERSION),
    )
    with pytest.raises(ConfigError, match="PPO"):
        build_task(job)
    assert build_task(replace(job, algorithm="distillation")).action_dim == 6


def test_changed_loaded_center_preserves_physical_labels_without_rewriting_inputs(tmp_path):
    old, wide = contracts(tmp_path)
    for row in wide["support_offset_table"]["offset_rad"]:
        row[0] += 0.07
        row[1] -= 0.04
    x = torch.zeros(10, 31)
    x[:, 24] = torch.linspace(0.3, 0.4, 10) * 2
    x[:, 16:22] = 0.2
    saved = x.clone()
    raw = torch.full((10, 6), 0.6)
    a, b = (
        ActionLabels("motion_contract", contract=old),
        ActionLabels("motion_contract", contract=wide),
    )
    physical = a.physical(raw, x)
    converted = b.labels(physical, x)
    torch.testing.assert_close(
        converted[:, 0], torch.full((10,), 0.6 * 0.3 - 0.07 / 0.5), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        converted[:, 1], torch.full((10,), 0.6 * 0.3 + 0.04 / 0.5), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(b.physical(converted, x), physical)
    torch.testing.assert_close(x, saved)
