"""Readback and real-observation export boundaries without physical integration."""

import numpy as np
import torch
from test_public_migration import model_file

from native_stand.artifacts.checkpoint import load_actor_source
from native_stand.artifacts.deployment import export_package, verify_package
from native_stand.artifacts.io import file_sha256, write_npz
from native_stand.tasks.recovery.distillation_metrics import summarize

FROZEN_TEST_SCOPE = "PORTABLE"


def test_incomplete_episode_unknown_and_reset_cannot_hide_first_failure():
    data = dict(
        physical_episode_keys=np.array(["a", "a", "b", "c"]),
        episodes=np.array([0, 0, 0, 1]),
        terminated=np.array([0, 1, 0, 1], bool),
        truncated=np.zeros(4, bool),
        condition_entered=np.array([1, 1, 1, 1]),
        condition_dual_contact=np.array([1, 0, 1, 1]),
        condition_continuous_hold_s=np.array([0.1, 0, 9.99, 10.0]),
        condition_HF_peak=np.array([0.1, 0.2, 0.1, 0.1]),
        condition_validation_cohort=np.array([0, 0, 1, 0]),
        condition_initial_Swing=np.ones(4),
    )
    report = summarize(data)
    assert report["completed"] == 1 and report["UNKNOWN"] == 1 and report["first_episodes"] == 2
    assert report["continuous_10s_rate"] == 0 and report["settling_contact_loss_rate"] == 1


def test_export_uses_supplied_actual_rows_and_rejects_rebound_source(tmp_path):
    import pytest

    spec, path = model_file(tmp_path)
    data = tmp_path / "observed.npz"
    inputs = np.arange(5 * 31, dtype=np.float32).reshape(5, 31) / 100
    write_npz(data, dict(states=inputs))
    golden = dict(path=str(data), sha256=file_sha256(data), indices=[1, 3])
    output = tmp_path / "deployment"
    export_package(load_actor_source(path), spec, output, execution={}, golden=golden)
    record = verify_package(output)
    assert record["golden_source"] == golden
    with np.load(output / "golden.npz") as raw:
        np.testing.assert_array_equal(raw["inputs"], inputs[[1, 3]])
    with pytest.raises(ValueError, match="golden source differs"):
        export_package(
            load_actor_source(path), spec, output, execution={}, golden=dict(golden, indices=[0])
        )


def test_sha_bound_teacher_contract_can_move_but_cannot_change(tmp_path):
    import dataclasses

    import pytest

    from native_stand.learning.inference import InferenceSpec, build_actor, from_source

    contract = dict(
        action_version="recovery_cubic_physical_Gaussian_v2",
        center_rad=[0.0] * 4,
        leg_lower_rad=[-1.5] * 4,
        leg_upper_rad=[1.5] * 4,
        center_source=dict(path="/original/contract.json", sha256="a" * 64),
    )
    spec = InferenceSpec("recovery_physical", 31, (8,), 6, {"version": "31"}, contract)
    source = tmp_path / "teacher.pt"
    torch.save(dict(actor=build_actor(spec).state_dict(), contract=contract), source)
    moved = dataclasses.replace(
        spec,
        output_contract={
            **contract,
            "center_source": dict(path="/moved/contract.json", sha256="a" * 64),
        },
    )
    from_source(load_actor_source(source), moved)
    changed = dataclasses.replace(
        moved,
        output_contract={
            **moved.output_contract,
            "center_source": dict(path="/moved/contract.json", sha256="b" * 64),
        },
    )
    with pytest.raises(ValueError, match="output contract differs"):
        from_source(load_actor_source(source), changed)


def test_formal_flow_compares_separate_closed_loops_and_final_does_not_reselect(
    tmp_path, monkeypatch
):
    import dataclasses

    from test_public_migration import supervision_job

    from native_stand.artifacts.io import read_json
    from native_stand.experiments.supervision import PhaseConfig
    from native_stand.experiments.supervision_flow import execute
    from native_stand.simulation import collection

    base = supervision_job(tmp_path)
    recipe = dataclasses.replace(
        base.supervision,
        phases=(
            PhaseConfig("label"),
            PhaseConfig("fit", rounds=1),
            PhaseConfig("compare", steps=4),
            PhaseConfig("compare", steps=4, final=True),
        ),
    )

    class IO:
        num_envs = 2

        def __init__(self):
            self.tick = 0
            self.seeds = []
            self.closed = False

        def observations(self):
            return {"student_actor": torch.zeros(2, 31)}

        def reset_collection(self, *, fresh=False, seed_offset=0):
            self.tick = 0
            self.seeds.append(seed_offset)

        def step(self, action):
            self.tick += 1
            return dict(
                terminated=torch.full((2,), self.tick == 2),
                truncated=torch.zeros(2, dtype=torch.bool),
                extras=dict(
                    actual_physics_steps=[10, 10],
                    control_owner=torch.zeros(2, dtype=torch.long),
                    collection_conditions=dict(
                        entered=torch.ones(2),
                        dual_contact=torch.ones(2),
                        continuous_hold_s=torch.full((2,), 10.0),
                        HF_peak=torch.zeros(2),
                        initial_Swing=torch.ones(2),
                        validation_cohort=torch.zeros(2),
                    ),
                ),
            )

        def close(self):
            self.closed = True

    env = IO()
    monkeypatch.setattr(collection, "make_environment", lambda job: env)
    charged = []
    state, done = execute(
        dataclasses.replace(base, supervision=recipe),
        tmp_path / "flow",
        stop_requested=lambda: False,
        on_round_complete=lambda row: None,
        on_round_saved=lambda row: None,
        on_physical_steps=charged.append,
    )
    assert done and env.closed and sum(charged) == 320
    assert env.seeds == [50000, 50000, 100000, 100000]
    dev = read_json(tmp_path / "flow/phase_002_compare/closed_loop.json")
    final = read_json(tmp_path / "flow/phase_003_compare/closed_loop.json")
    assert dev["partition"] == "development" and final["partition"] == "final_holdout"
    assert final["reports"]["student"]["continuous_10s_rate"] == 1
    assert state["closed_loop_candidate"] == state["selected"] and state["no_improvement"] == 0
    assert len(state["collections"]) == 1  # independent comparisons never feed BC


def test_mapped_asset_module_uses_its_actual_package_root(tmp_path):
    import pytest

    from native_stand.simulation.recovery_isaac.bindings import asset_module_root

    package = tmp_path / "agent_world"
    assets = package / "assets"
    assets.mkdir(parents=True)
    for folder in (package, assets):
        (folder / "__init__.py").write_text("")
    module = assets / "robot.py"
    module.write_text("ACTUAL=1\n")
    template = dict(
        asset_module_path=str(module),
        asset_module_sha256=file_sha256(module),
        asset_module_root="/stale/source",
    )
    assert asset_module_root(template) == tmp_path
    module.write_text("CHANGED=1\n")
    with pytest.raises(ValueError, match="identity differs"):
        asset_module_root(template)
