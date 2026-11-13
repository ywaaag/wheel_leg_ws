"""Recovery physical objective through the shared flow, including round resume."""

import dataclasses

import numpy as np
import pytest
import torch
from test_public_migration import supervision_job

from native_stand.artifacts.io import file_sha256, read_json, write_npz
from native_stand.experiments.supervision import PhaseConfig, TeacherConfig
from native_stand.experiments.supervision_flow import execute
from native_stand.learning.distillation.objective import per_row_loss, sampling_weights
from native_stand.learning.inference import InferenceSpec, build_actor

FROZEN_TEST_SCOPE = "PORTABLE"
OBJECTIVE = dict(
    kind="physical_huber",
    scales=[0.05, 0.05, 10.0, 0.05, 0.05, 10.0],
    phase_weights={str(i): 3.0 if i == 3 else 1.0 for i in range(7)},
    updates_per_round=2,
)


def test_physical_units_equal_penalty_and_episode_phase_mass():
    target = torch.zeros(2, 6)
    prediction = target.clone()
    prediction[0, 0] = 0.05
    prediction[1, 2] = 10.0
    torch.testing.assert_close(
        per_row_loss(prediction, target, OBJECTIVE), torch.full((2,), 1 / 12)
    )
    weights = sampling_weights(
        [0] * 101 + [1] * 2, [0] + [3] * 100 + [0, 3], OBJECTIVE["phase_weights"]
    )
    assert float(weights[:101].sum()) == pytest.approx(0.5)
    assert float(weights[1:101].sum()) == pytest.approx(0.375)
    with pytest.raises(ValueError, match="Unknown"):
        sampling_weights([0], [10], OBJECTIVE["phase_weights"])


def test_teacher_collection_fixed_updates_physical_labels_unknown_and_resume(tmp_path):
    base = supervision_job(tmp_path)
    contract = dict(
        action_version="recovery_cubic_physical_Gaussian_v2",
        center_rad=[0.0] * 4,
        leg_lower_rad=[-1.5] * 4,
        leg_upper_rad=[1.5] * 4,
    )
    teacher_spec = InferenceSpec("recovery_physical", 31, (16,), 6, {"version": "real31"}, contract)
    student_spec = dataclasses.replace(teacher_spec, hidden_dims=(8,))
    source = tmp_path / "teacher.pt"
    torch.save(dict(actor=build_actor(teacher_spec).state_dict(), contract=contract), source)
    recipe = dataclasses.replace(
        base.supervision,
        teachers=(
            TeacherConfig(
                "2100",
                str(source),
                file_sha256(source),
                teacher_spec.record(),
                {"kind": "physical"},
            ),
        ),
        student=student_spec.record(),
        student_labels={"kind": "physical"},
        objective=OBJECTIVE,
        dataset=base.supervision.dataset,
        phases=(
            PhaseConfig("collect", steps=2, policy="2100"),
            PhaseConfig("label"),
            PhaseConfig("fit", rounds=2),
        ),
    )
    job = dataclasses.replace(base, supervision=recipe)
    seen = []

    def collect(job, ref, spec, steps, directory, charge):
        assert ref["sha256"] == file_sha256(source) and spec["hidden_dims"] == (16,)
        path = directory / "real.npz"
        write_npz(
            path,
            dict(
                states=np.zeros((100, 31), np.float32),
                episodes=np.arange(100),
                condition_recovery_phase=np.arange(100) % 7,
            ),
        )
        seen.append(1)
        return dict(path=str(path), sha256=file_sha256(path), collection_id="actual")

    # The seed source is real in production; this fixture supplies phase evidence explicitly.
    with np.load(recipe.dataset["path"]) as raw:
        write_npz(
            tmp_path / "seed.npz",
            dict(
                states=raw["states"],
                episodes=raw["episodes"],
                condition_recovery_phase=np.arange(100) % 7,
            ),
        )
    recipe = dataclasses.replace(
        recipe,
        dataset=dict(path=str(tmp_path / "seed.npz"), sha256=file_sha256(tmp_path / "seed.npz")),
    )
    job = dataclasses.replace(job, supervision=recipe)
    charged = []
    args = dict(collector=collect, on_round_complete=charged.append, on_round_saved=lambda r: None)
    state, done = execute(job, tmp_path / "flow", stop_requested=lambda: len(charged) == 1, **args)
    assert not done
    state, done = execute(job, tmp_path / "flow", stop_requested=lambda: False, **args)
    assert done and len(seen) == 1
    assert [r["optimizer_steps"] for r in charged] == [2, 2]
    saved = torch.load(state["latest"]["path"], weights_only=False)
    assert saved["objective"] == OBJECTIVE and saved["round"] == 2
    with np.load(tmp_path / "flow/phase_001_label/labels.npz") as raw:
        assert set(raw["label_quality"]) == {"UNKNOWN"}
        assert len(raw["condition_recovery_phase"]) == 200
    assert read_json(tmp_path / "flow/supervision_state.json")["phase_cursor"] == 3
