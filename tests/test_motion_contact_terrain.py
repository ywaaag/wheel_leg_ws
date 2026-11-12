"""Real domain math/native I/O boundary tests, no engine or real PPO rollout."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from native_stand.artifacts.io import read_json
from native_stand.simulation.isaac.contact_materials import ContactMaterials
from native_stand.simulation.terrain_layout import tile_offsets, validate
from native_stand.simulation.terrain_surface import TriangleRoads
from native_stand.tasks.motion.contact_randomization import CRITIC_VERSION, append_context, sample

FROZEN_TEST_SCOPE = "PORTABLE"
ROOT = Path(__file__).resolve().parents[1]
JOB = (
    ROOT / "configs/jobs/2026_11_12__motion_u3000_bank_height020_H1_cloud10000_contact_grid_v2.json"
)


def contact_config():
    return read_json(JOB)["assets"]["wheel_contact_dr"]


class MaterialIO:
    """Scripted native material boundary, not a simulated contact solver."""

    def __init__(self, n=4):
        self.mat = torch.zeros(n, 5, 3)
        self.mat[..., :2] = 0.5
        self.sd = torch.zeros(n, 5, 2)
        self.mode = torch.zeros(n, 5, 2, dtype=torch.uint8)
        self.calls = []

    def get_material_properties(self):
        return self.mat

    def get_compliant_material_properties(self):
        return self.sd, self.mode

    def set_compliant_material_properties(self, data, combination, indices):
        assert data.shape == (len(self.mat), 5, 4)
        assert combination.shape == (len(self.mat), 5, 3)
        self.calls.append(("compliant", indices.clone()))
        self.mat[indices.long(), :, :2] = data[indices.long(), :, :2]
        self.sd[indices.long()] = data[indices.long(), :, 2:]
        self.mode[indices.long()] = combination[indices.long(), :, 1:]

    def set_material_properties(self, data, indices):
        self.calls.append(("ordinary", indices.clone()))
        self.mat[indices.long()] = data[indices.long()]
        self.sd[indices.long(), :, 0] = 0


def contact_fixture():
    io = MaterialIO()
    io.link_paths = [["/World/envs/env_0/Robot/left_wheel", "/World/envs/env_0/Robot/right_wheel"]]

    class WheelIO:
        count, max_shapes = 4, 1

        def __init__(self, column):
            self.column = column

        def get_material_properties(self):
            return io.mat[:, self.column : self.column + 1]

        def get_compliant_material_properties(self):
            return io.sd[:, self.column : self.column + 1], io.mode[
                :, self.column : self.column + 1
            ]

        def set_compliant_material_properties(self, data, mode, ids):
            assert data.shape == (4, 1, 4) and mode.shape == (4, 3)
            io.sd[ids.long(), self.column] = data[ids.long(), 0, 2:]
            io.mat[ids.long(), self.column, :2] = data[ids.long(), 0, :2]
            io.mode[ids.long(), self.column] = mode[ids.long(), 1:]

        def set_material_properties(self, data, ids):
            io.mat[ids.long(), self.column] = data[ids.long(), 0]
            io.sd[ids.long(), self.column, 0] = 0

    views = {"left_wheel": WheelIO(1), "right_wheel": WheelIO(4)}
    sim_view = NS(create_rigid_body_view=lambda pattern: views[pattern.rsplit("/", 1)[-1]])
    dr = NS(
        n=4,
        device="cpu",
        columns=[1, 4],
        env=NS(robot=NS(root_view=io, _physics_sim_view=sim_view)),
        profiles=torch.tensor([-1, 0, 0, 10]),
        amplitude=torch.ones(4, 1),
        episodes=torch.ones(4, dtype=torch.long),
        restitution=torch.full((4, 2), 0.2),
        last_readback={},
    )
    calls = []

    def noise(counter, ids, width, stream):
        calls.append((ids.clone(), stream))
        return torch.zeros(len(ids), width)

    contact = ContactMaterials(dr, contact_config(), noise, lambda x: x)
    return contact, dr, io, calls


def test_actual_native_reset_order_rigid_and_partial_are_isolated():
    contact, dr, io, calls = contact_fixture()
    # Exactly0.5 mixture boundary is rigid; force below it to exercise soft.
    contact.noise = lambda c, ids, w, s: torch.full((len(ids), w), -0.5)
    contact.reset(torch.arange(4))
    assert (contact.actual[[1, 2]] > 0).all() and not contact.actual[[0, 3]].any()
    assert not dr.restitution[[1, 2]].any()
    assert io.mat[0, 1, 2] == 0
    saved = contact.actual.clone()
    old = io.sd.clone()
    dr.profiles[1] = -1
    # The old material reset runs first. It may leave a nonzero damping cache.
    io.set_material_properties(io.mat.clone(), torch.tensor([1]))
    contact.reset(torch.tensor([1]))
    assert not contact.actual[1].any() and not io.sd[1].any()
    torch.testing.assert_close(contact.actual[[0, 2, 3]], saved[[0, 2, 3]])
    torch.testing.assert_close(io.sd[[0, 2, 3]], old[[0, 2, 3]])
    assert not contact.last_readback["contact_force_response_validated"]


def test_nominal_and_strength_zero_never_sample_contact_or_policy_rng():
    contact, dr, io, calls = contact_fixture()
    dr.profiles[:] = -1
    before = torch.get_rng_state()
    contact.reset(torch.arange(4))
    dr.profiles[:] = 0
    dr.amplitude[:] = 0
    contact.reset(torch.arange(4))
    assert calls == [] and not contact.actual.any()
    assert torch.equal(before, torch.get_rng_state())


def test_readback_mismatch_is_rejected_and_nonwheel_material_is_untouched():
    contact, dr, io, _ = contact_fixture()
    io.mat[0, 0, 2] = 0.3
    contact.reset(torch.tensor([0]))
    assert io.mat[0, 0, 2] == pytest.approx(0.3)
    contact.noise = lambda c, ids, w, s: torch.full((len(ids), w), -0.5)
    for view in contact.wheels:
        view.set_compliant_material_properties = lambda *args: None
    with pytest.raises(RuntimeError, match="readback"):
        contact.reset(torch.tensor([1]))


def test_native_warp_frontend_uses_explicit_float_uint8_and_indices(monkeypatch):
    import warp as wp

    from native_stand.simulation.isaac.contact_materials import physics_tensor, write_native

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    wp.init()
    assert not wp.get_cuda_devices()
    original = torch.zeros(2, 3, 3)
    calls = []

    class WarpBoundary:
        def get_material_properties(self):
            return wp.from_torch(original, dtype=wp.float32)

        def set_compliant_material_properties(self, data, mode, ids):
            assert isinstance(data, wp.array) and data.dtype == wp.float32
            assert isinstance(mode, wp.array) and mode.dtype == wp.uint8
            assert isinstance(ids, wp.array) and ids.dtype == wp.int32
            calls.append((physics_tensor(data).clone(), physics_tensor(ids).clone()))

    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    modes = torch.zeros(2, 3, 3, dtype=torch.uint8)
    write_native(
        WarpBoundary(),
        "set_compliant_material_properties",
        (values, modes, torch.tensor([1], dtype=torch.int32)),
    )
    torch.testing.assert_close(calls[0][0], values)
    assert calls[0][1].tolist() == [1]


def test_contact_strength_causal_context_and_invalid_ranges():
    c = contact_config()
    u = torch.tensor([[-1.0, -1.0, 1.0, -1.0, 1.0]])
    value = sample(u, torch.ones(1), c)
    torch.testing.assert_close(value, torch.tensor([[1e5, 1e6, 500.0, 3000.0]]))
    assert not sample(u, torch.zeros(1), c).any()
    got = append_context(torch.ones(1, 877), value, c)
    assert got.shape == (1, 881)
    torch.testing.assert_close(got[:, 877:], torch.tensor([[0.1, 1.0, 1 / 6, 1.0]]))
    for wrong in ([-1.0, 100.0], [100.0, 1.0], [100.0, float("inf")]):
        with pytest.raises(ValueError):
            sample(u, torch.ones(1), {**c, "stiffness_n_m": wrong})


def test_native_rigid_infinite_stiffness_is_finite_disabled_context_not_missing_soft_write():
    contact, dr, io, _ = contact_fixture()
    getter = io.get_compliant_material_properties

    def native_rigid():
        sd, modes = getter()
        sd = sd.clone()
        # Script the actual native PhysX rigid encoding found by local smoke.
        sd[..., 0] = torch.where(sd[..., 0] == 0, float("inf"), sd[..., 0])
        return sd, modes

    io.get_compliant_material_properties = native_rigid
    contact.noise = lambda c, ids, w, s: torch.full((len(ids), w), -0.5)
    contact.reset(torch.arange(4))
    assert torch.isfinite(contact.actual).all()
    assert not contact.actual[[0, 3]].any() and (contact.actual[[1, 2]] > 0).all()
    assert contact.last_readback["native_rigid_positive_infinite_stiffness_rows"] == 2
    for view in contact.wheels:
        view.set_compliant_material_properties = lambda *args: None
    io.sd[1] = 0
    with pytest.raises(RuntimeError, match="readback differs"):
        contact.reset(torch.tensor([1]))


def test_native_force_spring_contract_rejects_acceleration_or_double_soft_ground():
    from pxr import Sdf, Usd, UsdShade

    from native_stand.simulation.isaac.contact_materials import check_scene_materials

    stage = Usd.Stage.CreateInMemory()
    ground = UsdShade.Material.Define(stage, "/World/ground").GetPrim()
    wheel = UsdShade.Material.Define(stage, "/World/recovery_dr_wheel_material").GetPrim()
    env = NS(sim=NS(stage=stage), cfg=NS(ground_material_path="/World/ground"))
    check_scene_materials(env)
    a = wheel.CreateAttribute(
        "physxMaterial:compliantContactAccelerationSpring", Sdf.ValueTypeNames.Bool
    )
    a.Set(True)
    with pytest.raises(RuntimeError, match="acceleration"):
        check_scene_materials(env)
    a.Set(False)
    ground.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(
        100.0
    )
    with pytest.raises(RuntimeError, match="rigid ground"):
        check_scene_materials(env)


def test_full_first_layer_and_dual_adam_migration_preserve_old_function(tmp_path):
    from test_motion_spin_branches import equal_tree, fixture

    from native_stand.artifacts.checkpoint import load_complete_state, save_full
    from native_stand.learning.ppo.assembly import build_critic
    from native_stand.simulation.isaac.motion_backend import _resolve_source
    from native_stand.tasks.motion.branch_state import confirmation_execution_identity

    job, _, _ = fixture(tmp_path)
    old = _resolve_source(job, None).payload
    cfg = dict(job.assets["branch_course"])
    newjob = replace(
        job,
        model=replace(job.model, critic_input=881, critic_privileged_version=CRITIC_VERSION),
        observation=replace(job.observation, critic_dim=881),
        assets={
            **job.assets,
            "wheel_contact_dr": contact_config(),
            "contact_critic_extension": {
                "from": "motion48_height_bank_planned_causal877_v2",
                "to": CRITIC_VERSION,
            },
            "branch_course": cfg,
        },
    )
    cfg["confirmation_execution_sha256"] = confirmation_execution_identity(newjob)
    newjob = replace(newjob, curriculum=replace(newjob.curriculum, stages=(cfg,)))
    new = _resolve_source(newjob, None).payload
    for name in ("actor", "std", "training_rng", "source_anchor_actor", "source_policy_retention"):
        equal_tree(old[name], new[name])
    equal_tree(old["optimizer"]["actor"], new["optimizer"]["actor"])
    assert new["warmup_updates_done"] == 50
    before, after = build_critic(job), build_critic(newjob)
    before.load_state_dict(old["critic"])
    after.load_state_dict(new["critic"])
    x = torch.randn(5, 881)
    torch.testing.assert_close(before({"student_critic": x[:, :877]}), after({"student_critic": x}))
    save_full(tmp_path / "new.pt", new)
    loaded = load_complete_state(tmp_path / "new.pt")
    equal_tree(new, loaded.payload)
    assert loaded.payload["critic"]["mlp.0.weight"].shape[1] == 881


def test_normal_and_pre_reset_terminal_use_same881_actual_contact_columns():
    from native_stand.simulation.isaac.maneuver_planned import environment_class

    class Boundary:
        def _get_observations(self):
            return {"student_critic": torch.zeros(2, 875)}

        def _get_dones(self):
            return torch.tensor([False, False]), torch.tensor([True, False])

    env = environment_class(Boundary)()
    env.planned_value_ready = True
    env.target_xy = torch.tensor([[0.0, 0.0], [3.6, 0.0]])
    env.motion_timing = NS(
        dr=NS(
            contact=NS(
                actual=torch.tensor([[1e5, 2e5, 500.0, 1000.0], [0.0, 0.0, 0.0, 0.0]]),
                config=contact_config(),
            )
        )
    )
    normal = env._get_observations()["student_critic"]
    env._get_dones()
    assert normal.shape == (2, 881)
    torch.testing.assert_close(normal, env.student_terminal_critic)
    assert env.student_terminal_critic[1, -4:].sum() == 0


def test_actual881_adapter_capture_restore_keeps_Adam_warmup_and_extension(tmp_path, monkeypatch):
    from tensordict import TensorDict
    from test_motion_spin_branches import equal_tree, fixture
    from test_phase3_fixes import _production_context

    from native_stand.artifacts.checkpoint import load_complete_state
    from native_stand.learning.ppo.source_anchor import KEY
    from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter
    from native_stand.students.resume import rng_state
    from native_stand.tasks.motion.branch_state import confirmation_execution_identity
    from native_stand.tasks.motion.maneuver_statistics import RESPONSE_VERSION, episode_class

    job, _, _ = fixture(tmp_path)
    cfg = dict(job.assets["branch_course"])
    job = replace(
        job,
        model=replace(job.model, critic_input=881, critic_privileged_version=CRITIC_VERSION),
        observation=replace(job.observation, critic_dim=881),
        assets={
            **job.assets,
            "wheel_contact_dr": contact_config(),
            "contact_critic_extension": {
                "from": "motion48_height_bank_planned_causal877_v2",
                "to": CRITIC_VERSION,
            },
            "branch_course": cfg,
        },
    )
    cfg["confirmation_execution_sha256"] = confirmation_execution_identity(job)
    job = replace(job, curriculum=replace(job.curriculum, stages=(cfg,)))
    context = _production_context(tmp_path / "run", job, smoke=1)

    def physics_boundary(adapter):
        adapter.env = NS(
            num_envs=4,
            device="cpu",
            stage_spec=adapter.spec["curriculum"]["stages"][0],
            identified_dr=NS(episodes=torch.ones(4, dtype=torch.long)),
            transient_events=NS(summary=lambda: {}),
        )
        adapter.env.onpolicy_episodes = episode_class(object, version=RESPONSE_VERSION)(adapter.env)
        adapter.obs = TensorDict(
            dict(
                student_actor=torch.zeros(4, 31),
                student_critic=torch.zeros(4, 881),
                recovery_actor_valid=torch.ones(4, 1, dtype=torch.bool),
                **{KEY: torch.full((4, 1), -1, dtype=torch.long)},
            ),
            batch_size=[4],
        )
        adapter.modules = {"rng_state": rng_state}

    monkeypatch.setattr(MotionProductionAdapter, "_open_environment", physics_boundary)
    monkeypatch.setattr(MotionProductionAdapter, "_write_restoration", lambda self: None)
    adapter = MotionProductionAdapter()
    try:
        adapter.build(context, "smoke", None)
        assert adapter.algorithm.actor_update_enabled
        saved = adapter.capture(context, {})
        assert saved["warmup_updates_done"] == 50
        assert saved["contact_critic_extension_receipt"]["to_dim"] == 881
        ref = context.save(saved, update=0, stage="smoke")
        restored = MotionProductionAdapter()
        restored.build(context, "smoke", load_complete_state(ref["checkpoint"]))
        equal_tree(saved["optimizer"], restored.algorithm.optimizer.state_dict())
        equal_tree(saved["critic"], restored.algorithm.critic.state_dict())
        equal_tree(saved["actor"], restored.algorithm.actor.state_dict())
        assert (
            restored.algorithm.source_anchor.state_dict()
            == adapter.algorithm.source_anchor.state_dict()
        )
        assert restored.state == adapter.state
    finally:
        context.metrics.close()


def test_source_KL_mask_excludes_real_rough_and_compliant_rows():
    from native_stand.learning.ppo.source_anchor import KEY
    from native_stand.recovery.controlled_ppo import KEY as CONTROL
    from native_stand.simulation.isaac.motion_backend import annotate_retention

    n = 5
    env = NS(
        device="cpu",
        stage_spec={"identified_profiles": [-1] * n, "modal_modes": [-1] * n},
        command_index=torch.arange(n),
        command=torch.tensor([[3.6, 0.0, 0.34]] * n),
        target_xy=torch.tensor([[3.6, 0.0]] * n),
        transient_events=NS(enabled=torch.zeros(n, dtype=torch.bool)),
        rise_hold=NS(episode=NS(steps=torch.full((n,), 300))),
        standing_snapshot=torch.zeros(n, 2),
        recovery_snapshot=torch.full((n, 12), 10.0),
        motion_flat_terrain=torch.tensor([True, False, True, True, True]),
        motion_timing=NS(
            dr=NS(
                contact=NS(
                    actual=torch.tensor(
                        [
                            [0.0, 0, 0, 0],
                            [0.0, 0, 0, 0],
                            [1e5, 1e5, 500, 500],
                            [0.0, 0, 0, 0],
                            [0.0, 0, 0, 0],
                        ]
                    )
                )
            )
        ),
    )
    env.transient_events.enabled[3] = True
    value = {CONTROL: torch.tensor([[True], [True], [True], [True], [False]])}
    job = NS(
        assets={
            "source_policy_retention": {
                "version": "motion48_per_command_nominal_source_KL_v4",
                "commands": [[3.6, 0.0]],
            },
            "terrain_scene": {"actual": True},
        }
    )
    annotate_retention(value, env, job)
    assert value[KEY].flatten().tolist() == [0, -1, -1, -1, -1]


def test_two_dimensional_query_periodicity_gradient_and_finite_extent():
    h = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]] * 2)
    s = TriangleRoads(
        h,
        torch.zeros(1, 3),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.ones(1, dtype=torch.long),
        cell=1.0,
        length=2.0,
        width=2.0,
        repetitions=75,
        lateral_repetitions=5,
        spawn_x=12.0,
    )
    xy = torch.tensor([[[-0.2, 0.3], [1.8, 0.3], [-0.2, 2.3]]])
    z, g = s.sample(xy)
    torch.testing.assert_close(z[:, 0], z[:, 1])
    torch.testing.assert_close(z[:, 0], z[:, 2])
    torch.testing.assert_close(g[:, 0], g[:, 1])
    assert s.contains(torch.tensor([[0.0, 120.0]])).item()
    assert not s.contains(torch.tensor([[5.1, 0.0]])).item()


def test_new_layout_covers_real_full_horizon_while_old_layout_is_rejected():
    from native_stand.tasks.motion.maneuver_program import extended_tensor_program

    cfg = read_json(JOB)["assets"]["terrain_scene"]
    assert len(tile_offsets(cfg)) == 75
    assert validate(cfg)
    old = {**cfg, "repetitions": 5}
    with pytest.raises(ValueError, match="episode horizon"):
        validate(old)
    ages = torch.arange(3001, dtype=torch.float64) * 0.01
    target = torch.tensor([[3.6, 0.0]], dtype=torch.float64).expand(len(ages), -1)
    command, _ = extended_tensor_program(ages, target, torch.full_like(ages, 0.34))
    distance = command[:, 0].cumsum(0) * 0.01
    assert distance.max() < cfg["length_m"] * cfg["repetitions"] - cfg["spawn_x_m"] - 0.12


def test_derived_USD_points_match_query_and_keep_H1_interior():
    from pxr import Usd, UsdGeom, UsdPhysics

    cfg = read_json(JOB)["assets"]["terrain_scene"]
    for row in cfg["assets"]:
        h = np.load(row["heights"]["path"], allow_pickle=False)["height_m"]
        assert np.array_equal(h[0], h[-1]) and np.array_equal(h[:, 0], h[:, -1])
        receipt = read_json(Path(row["usd"]["path"]).with_name("derivation.json"))
        assert receipt["unchanged_interior_fraction"] == 1
        stage = Usd.Stage.Open(row["usd"]["path"])
        mesh = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
        assert mesh.HasAPI(UsdPhysics.CollisionAPI)
        points = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
        ix = np.rint(points[:, 0] / 0.02).astype(int)
        iy = np.rint((points[:, 1] + 1.5) / 0.02).astype(int)
        np.testing.assert_array_equal(points[:, 2], h[iy, ix])


def test_formal_terrain_scene_authors_real_shared_collision_instances():
    from pxr import Usd, UsdPhysics, UsdShade

    from native_stand.simulation.isaac.terrain_scene import setup

    stage = Usd.Stage.CreateInMemory()
    UsdShade.Material.Define(stage, "/World/material")
    for i in range(2):
        UsdPhysics.CollisionGroup.Define(stage, f"/World/collisions/env_{i}")
    plane = UsdPhysics.CollisionGroup.Define(stage, "/World/plane_0")
    plane.CreateFilteredGroupsRel().AddTarget("/World/collisions/env_0")
    env = NS(
        device="cpu",
        num_envs=2,
        sim=NS(stage=stage),
        cfg=NS(ground_material_path="/World/material"),
        stage_spec={"sampling_slots": [0, 1], "terrain_families": ["cobblestones_H1", "flat"]},
        scene=NS(env_origins=torch.zeros(2, 3)),
        DR_static_ground_angle=torch.zeros(2),
        DR_static_ground_normal=torch.tensor([[0.0, 0.0, 1.0]] * 2),
    )
    setup(env, read_json(JOB)["assets"]["terrain_scene"])
    tiles = [p for p in stage.Traverse() if p.IsInstance()]
    assert len(tiles) == 75
    assert len(stage.GetPrototypes()) == 1
    mesh = stage.GetPrimAtPath("/World/MotionRoads/road_0/tile_74/Surface")
    assert mesh.HasAPI(UsdPhysics.CollisionAPI)
    binding = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial("physics")[0]
    assert str(binding.GetPath()) == "/World/material"
    assert env.motion_terrain_readback["extent_m"] == [150.0, 9.0]
    assert env.motion_terrain_readback["stage_mesh_count"] == 75
    assert env.motion_surface.contains(torch.tensor([[0.0, 110.0], [0.0, 0.0]])).all()


def test_runtime_failure_report_uses_preserved_pre_reset_causes(tmp_path):
    from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter

    a = MotionProductionAdapter()
    snapshot = torch.zeros(2, 32)
    snapshot[1, 4] = 1
    snapshot[1, 12] = 0.001
    snapshot[1, 14] = -0.006
    a.context = NS(run_dir=tmp_path, updates_used=3)
    a.env = NS(
        recovery_snapshot=snapshot,
        motion_timing=NS(tick=2210),
        student_fault_components={
            "penetration": torch.tensor([False, True]),
            "closure": torch.zeros(2, dtype=torch.bool),
        },
        student_fault_joint_data={"q": torch.tensor([[0.0, 0.0], [0.1, 0.2]])},
        motion_terrain_ids=torch.tensor([0, 2]),
        cfg=NS(closure_fault_m=0.005),
        recovery_config={"initialization": {"severe_penetration_m": 0.005}},
    )
    a.record_runtime_fault()
    saved = read_json(tmp_path / "runtime_fault_detail.json")
    assert saved["rows"] == [1] and saved["components"]["penetration"] == [True]
    assert saved["pre_reset_joints"]["q"][0] == pytest.approx([0.1, 0.2])
    assert saved["updates_used"] == 3 and saved["terrain_ids"] == [2]
