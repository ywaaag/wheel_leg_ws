"""New bounded Motion contract, exact terrain math; no engine/GPU launched."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from native_stand.artifacts.branch_initialization import require_allocated
from native_stand.artifacts.io import file_sha256, read_json
from native_stand.experiments.config import load_job_config
from native_stand.simulation.isaac.motion_backend import _resolve_source, validate_job
from native_stand.simulation.terrain_surface import TriangleRoads
from native_stand.tasks.motion.action_contract import ActionDecoder
from native_stand.tasks.motion.maneuver_program import extended_tensor_program
from native_stand.tasks.motion.maneuver_targets import mode_masks, roll_target

FROZEN_TEST_SCOPE = "HOST"
ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "configs/jobs/2026_11_12__motion_u3000_bank_height020_roughH1_smoke37_r6.json"


def test_actual_source_branch_only_extends_decoder_and_keeps_complete_training_state():
    from test_motion_spin_branches import equal_tree

    from native_stand.artifacts.checkpoint import load_complete_state

    job = load_job_config(JOB)
    before = load_complete_state(job.initialization.checkpoint)
    after = _resolve_source(job, None)
    from native_stand.simulation.isaac.environment import control_router
    from native_stand.simulation.isaac.motion_backend import _check_source_progress

    assert _check_source_progress(job, after.payload, None) == 0
    assert control_router(job, after.payload, None).student_decoder.center.height_range_m == (
        0.2,
        0.4,
    )
    for key in (
        "actor",
        "critic",
        "optimizer",
        "std",
        "training_rng",
        "source_anchor_actor",
        "source_anchor_runtime_state",
    ):
        equal_tree(before.payload[key], after.payload[key])
    assert after.payload["model_online_updates"] == 5097
    assert after.payload["course_updates"] == 0
    assert after.payload["warmup_updates_done"] == 50
    assert after.payload["action_version"] == "motion_multiheight020_support_center_015_80_v2"
    assert after.payload["output_contract_extension_receipt"]["legacy_max_difference_rad"] <= 1e-6
    assert validate_job(job)
    grant = require_allocated(job)
    assert grant["smoke_updates"] == 37 and grant["formal_updates"] == 0
    assert job.assets["historical_online_updates"] == 5111 + 27


def test_no_gpu_refuses_before_any_worker_or_run_is_created(tmp_path, monkeypatch):
    import subprocess

    from native_stand.simulation.isaac.motion_backend import preflight

    job = load_job_config(
        ROOT / "configs/jobs/2026_11_12__motion_u3000_bank_height020_H1_cloud10000.json"
    )

    def unavailable(*args, **kwargs):
        raise OSError(8, "Exec format error", "nvidia-smi")

    monkeypatch.setattr(subprocess, "check_output", unavailable)
    with pytest.raises(RuntimeError, match="无卡模式"):
        preflight(job)
    assert not list(tmp_path.iterdir())


def test_worker_record_task_and_physical_decoder_share_full_height_table():
    from native_stand.experiments.config import job_from_record
    from native_stand.experiments.worker import build_task

    job = job_from_record(load_job_config(JOB).as_run_record())
    task = build_task(job)
    decoder = ActionDecoder.from_contract(read_json(job.assets["output_contract"]))
    heights = torch.linspace(0.2, 0.4, 101)
    torch.testing.assert_close(task.center.centers(heights), decoder.center.centers(heights))
    failed = load_job_config(
        ROOT / "configs/jobs/2026_11_12__motion_u3000_bank_height020_roughH1_smoke64.json"
    )
    with pytest.raises(ValueError, match="覆盖声明高度范围"):
        validate_job(failed)


@pytest.mark.parametrize("change", ["long", "overspend", "wrong_output", "prepared"])
def test_allocation_does_not_allow_long_training_or_unbound_configuration(change):
    job = load_job_config(JOB)
    if change == "long":
        job = replace(job, limits=replace(job.limits, formal_updates=1))
    elif change == "overspend":
        job = replace(job, limits=replace(job.limits, smoke_updates=65))
    elif change == "wrong_output":
        job = replace(job, assets={**job.assets, "output_contract_sha256": "a" * 64})
    else:
        job = replace(job, assets={**job.assets, "preparation_only": True})
    with pytest.raises(ValueError):
        require_allocated(job)


def test_old_prepared_branches_still_have_no_executable_budget():
    for name in ("motion48_u3000_prepared", "spin48_u3000_prepared"):
        with pytest.raises(ValueError, match="尚未分配"):
            require_allocated(load_job_config(ROOT / "configs/jobs" / f"{name}.json"))


def test_new_cloud_user_grant_is_separate_from_the_original_shared_remainder():
    job = load_job_config(
        ROOT / "configs/jobs/2026_11_12__motion_u3000_bank_height020_H1_cloud10000.json"
    )
    grant = require_allocated(job)
    assert grant["formal_updates"] == 10000 and grant["smoke_updates"] == 0
    assert job.assets["historical_online_updates"] == 5167
    assert job.assets["original_online_limit"] == 15167
    with pytest.raises(ValueError):
        require_allocated(replace(job, limits=replace(job.limits, formal_updates=10001)))
    with pytest.raises(ValueError):
        require_allocated(
            replace(job, assets={**job.assets, "new_independent_authorization": False})
        )


def test_worker_owned_kit_teardown_waits_until_budget_and_state_are_terminal(tmp_path, monkeypatch):
    import atexit

    from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter

    callbacks = []
    calls = []
    state = {"status": "RUNNING"}
    monkeypatch.setattr(atexit, "register", lambda f: callbacks.append(f))
    adapter = MotionProductionAdapter()
    adapter.job = NS(assets={"application_lifecycle": {"owner": "worker_atexit"}})
    adapter.context = NS(run_dir=tmp_path, updates_used=0, store=NS(read=lambda: state))
    adapter.app = NS(close=lambda **kw: calls.append(kw))
    adapter.env = NS(close=lambda: calls.append("env"))
    adapter.close()
    assert calls == ["env"] and len(callbacks) == 1
    state["status"] = "COMPLETED"
    callbacks[0]()
    assert calls == ["env", {"exit_code": 0}]


def test_new_centers_preserve_old_function_and_accept_full_low_range():
    job = load_job_config(JOB)
    new = ActionDecoder.from_contract(read_json(job.assets["output_contract"]))
    oldjob = load_job_config(ROOT / "configs/jobs/motion48_u3000_prepared.json")
    old = ActionDecoder.from_contract(read_json(oldjob.assets["output_contract"]))
    h = torch.linspace(0.30, 0.40, 101, dtype=torch.float64)
    a = torch.linspace(-1.1, 1.1, 606, dtype=torch.float64).reshape(101, 6)
    torch.testing.assert_close(
        old.decode(a, h)["requested_target"],
        new.decode(a, h)["requested_target"],
        atol=1e-6,
        rtol=0,
    )
    low = new.decode(torch.zeros(201, 6), torch.linspace(0.20, 0.40, 201, dtype=torch.float64))[
        "requested_target"
    ]
    assert torch.isfinite(low).all()
    torch.testing.assert_close(low[:, 0], -low[:, 3], atol=1e-7, rtol=0)
    torch.testing.assert_close(low[:, 1], low[:, 4], atol=1e-7, rtol=0)
    with pytest.raises(ValueError):
        new.decode(torch.zeros(1, 6), torch.tensor([0.199]))


def test_new_center_binding_and_geometry_are_checked_before_full_initialization(tmp_path):
    from native_stand.artifacts.io import write_json
    from native_stand.tasks.motion.height_extension import extend_source

    job = load_job_config(JOB)
    oldjob = load_job_config(ROOT / "configs/jobs/motion48_u3000_prepared.json")
    source = {"output_contract": read_json(oldjob.assets["output_contract"])}
    assert extend_source(source, job)["output_contract_extension_receipt"]["center_binding_sha256"]
    binding = read_json(job.assets["support_center_policy_binding"]["path"])
    binding["policy_sha256"] = "f" * 64
    path = tmp_path / "wrong_binding.json"
    write_json(path, binding)
    wrong = replace(
        job,
        assets={
            **job.assets,
            "support_center_policy_binding": {"path": str(path), "sha256": file_sha256(path)},
        },
    )
    with pytest.raises(ValueError, match="新合同身份"):
        extend_source(source, wrong)


def test_low_height_command_finishes_before_movement_and_never_jumps_reset():
    ages = torch.tensor([0.0, 1.0, 3.0, 7.9, 8.0, 8.9, 9.0, 10.0, 12.0])
    goals = torch.full_like(ages, 0.2)
    xy = torch.tensor([0.25, 0.0]).expand(len(ages), -1)
    c, r = extended_tensor_program(ages, xy, goals)
    assert c[0, 2] == pytest.approx(0.34)
    assert (c[:, 0][ages < 9] == 0).all()
    assert c[-1, 0] == pytest.approx(0.25) and c[-1, 2] == pytest.approx(0.20)
    assert r.abs().max() <= 0.0200001
    assert r[ages >= 8].abs().max() == 0


def test_height_and_turn_program_numpy_and_training_tensors_use_same_physical_commands():
    from native_stand.tasks.motion.maneuver_program import extended_numpy_program

    for command in ([0.25, 0.0, 0.20], [2.0, -0.5, 0.34], [-2.0, 0.5, 0.4]):
        ages = torch.linspace(0, 30, 301, dtype=torch.float64)
        xy = torch.tensor(command[:2], dtype=torch.float64).expand(len(ages), -1)
        goals = torch.full_like(ages, command[2])
        got, _ = extended_tensor_program(ages, xy, goals)
        expected = torch.as_tensor(
            __import__("numpy").array([extended_numpy_program(float(t), command) for t in ages])
        )
        torch.testing.assert_close(got, expected, atol=1e-12, rtol=0)


def test_low_height_critic_phase_flags_follow_actual_rate_limited_command():
    from native_stand.tasks.motion.maneuver_program import extended_phase_flags

    age = torch.tensor([4.0, 8.5, 9.5, 10.5], dtype=torch.float64)
    goals = torch.full_like(age, 0.2)
    quiet, ramp = extended_phase_flags(age, goals)
    assert quiet.tolist() == [False, True, False, False]
    assert ramp.tolist() == [False, False, True, False]


def test_symmetry_only_flat_nonturn_and_inward_bank_signs():
    command = torch.tensor([[0, 0, 0.34], [2, 0, 0.34], [2, 0.5, 0.34], [0, 0, 0.34], [2, 0, 0.34]])
    mode = mode_masks(command, is_flat=torch.tensor([True, True, True, False, False]))
    assert mode["symmetry_required"].tolist() == [True, True, False, False, False]
    assert roll_target(command)[2] < 0


def surface():
    # Source x longitudinal; exact triangle, not bilinear interpolation.
    h = torch.tensor([[0.0, 1.0, 0.0], [2.0, 4.0, 2.0], [0.0, 1.0, 0.0]])
    heights = torch.stack((torch.zeros_like(h), h))
    return TriangleRoads(
        heights,
        torch.zeros(2, 3),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        torch.tensor([0, 1]),
        cell=1.0,
        length=2.0,
        width=2.0,
        repetitions=3,
        spawn_x=0.0,
    )


def test_actual_triangle_query_rotation_gradient_repeat_and_no_flat_fallback():
    p = surface()
    xy = torch.tensor([[123.0, 456.0], [0.75, 0.25]])
    z, g = p.sample(xy)
    assert z[0] == 0 and z[1] == pytest.approx(0.75)
    torch.testing.assert_close(g[1], torch.tensor([-2.0, 1.0]))
    repeated = xy.clone()
    repeated[1, 1] += 2
    assert torch.equal(p.sample(repeated)[0], z)
    with pytest.raises(RuntimeError, match="no flat fallback"):
        p.sample(torch.tensor([[0.0, 0.0], [0.0, 6.1]]))


def test_road_exit_has_explicit_invalid_terminal_height_without_extrapolated_surface():
    from native_stand.simulation.ground_surface import from_env

    p = surface()
    pos = torch.tensor([[0.0, 0.0, 0.34], [3.0, 0.25, 0.34]])
    env = NS(
        motion_surface=p,
        motion_road_exit=torch.zeros(2, dtype=torch.bool),
        robot=NS(data=NS(root_pos_w=NS(torch=pos))),
    )
    got = from_env(env)
    assert env.motion_road_exit.tolist() == [False, True]
    assert env.motion_terrain_height_valid.tolist() == [True, False]
    assert got[1] == 0  # terminal sentinel, not asserted actual terrain height
    assert p.contains(torch.tensor([[0.0, 0.0], [0.0, 0.05]]), margin=0.1).tolist() == [True, False]


def test_height_rate_consumes_actual_collision_surface_not_absolute_root_z():
    from native_stand.simulation.ground_surface import from_env, velocity_from_env

    p = surface()
    pos = torch.tensor([[0.0, 0.0, 0.34], [0.75, 0.25, 1.09]])
    env = NS(
        motion_surface=p,
        robot=NS(
            data=NS(
                root_pos_w=NS(torch=pos),
                root_lin_vel_w=NS(torch=torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])),
            )
        ),
    )
    torch.testing.assert_close(from_env(env), torch.full((2,), 0.34), atol=1e-7, rtol=0)
    assert velocity_from_env(env)[1] == pytest.approx(0.3)


def test_terrain_sources_sha_and_isolated_cohorts():
    from native_stand.tasks.motion.course_strategy import resolve

    job = load_job_config(JOB)
    loaded = _resolve_source(job, None)
    stage = resolve(job.curriculum.version).stage_config(
        0, 128, state=loaded.payload["motion_course_state"]
    )
    assert sum(stage["slot_counts"]) == 128
    actual = set(
        x for x, n in zip(stage["terrain_families"], stage["slot_counts"], strict=False) if n
    )
    assert actual == {"flat", *job.assets["terrain_scene"]["families"]}
    for row in job.assets["terrain_scene"]["assets"]:
        for key in ("usd", "heights"):
            assert file_sha256(row[key]["path"]) == row[key]["sha256"]


def test_engine_failure_is_durable_before_application_shutdown_and_original_error_survives(
    tmp_path,
):
    from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter

    calls = []
    adapter = MotionProductionAdapter()
    adapter.context = NS(run_dir=tmp_path, updates_used=0)
    adapter.env = NS(close=lambda: calls.append("env"), motion_timing=NS(tick=7))
    adapter.app = NS(close=lambda: calls.append("app"))
    with pytest.raises(RuntimeError, match="physical I/O failure"):
        try:
            raise RuntimeError("physical I/O failure")
        finally:
            adapter.close()
    record = read_json(tmp_path / "engine_failure.json")
    assert "physical I/O failure" in record["traceback"]
    assert record["physical_tick"] == 7 and record["updates_used"] == 0
    assert record["reported_before_application_shutdown"]
    assert calls == ["env", "app"]
    adapter.close()
    assert calls == ["env", "app"]


def test_maneuver_height_wrapper_does_not_consume_not_yet_appended_transfer_reward():
    from native_stand.simulation.isaac.height import subclass

    # Physical I/O boundary: inner base reward precedes fullchain transfer fields.
    class PhysicalIO:
        def _get_observations(self):
            return {"student_critic": torch.arange(140).float()[None]}

        def _get_dones(self):
            self.student_reward_result = {"fields": ("height",), "density": torch.ones(1, 1)}
            return (False, False)

    cls = subclass(PhysicalIO, {}, reward_owner="maneuver")
    wrapped = cls.__new__(cls)
    wrapped.height_adapter = NS()
    wrapped.motion_quality_ready = True
    assert wrapped._get_dones() == (False, False)
    assert wrapped.student_reward_result["fields"] == ("height",)
    assert torch.equal(
        wrapped._get_observations()["student_critic"], torch.arange(140).float()[None]
    )
