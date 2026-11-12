"""Static H1 USD collision and matching height queries, before sim.reset()."""

import torch

from native_stand.artifacts.io import file_sha256
from native_stand.simulation.terrain_surface import TriangleRoads


def align_reset_height(env, ids):
    """Translate a NEW rough reset to actual floor, without changing joints/velocities."""
    data = env.robot.data
    selected = ids[~env.motion_flat_terrain[ids]]
    if len(selected):
        xy = data.root_pos_w.torch[:, :2]
        if not bool(env.motion_surface.contains(xy)[selected].all()):
            raise RuntimeError("Rough reset is outside actual collision terrain")
        ground, _ = env.motion_surface.sample(xy, check=False)
        pose = torch.cat((data.root_pos_w.torch[selected], data.root_quat_w.torch[selected]), -1)
        pose[:, 2] += ground[selected] - env.scene.env_origins[selected, 2]
        env.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=selected.to(torch.int32))


def setup(env, config):
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    from native_stand.simulation.terrain_layout import tile_offsets, validate

    validate(config)
    stage = env.sim.stage
    slots = torch.tensor(env.stage_spec["sampling_slots"], device=env.device)
    families = config["families"]
    rows = env.stage_spec["terrain_families"]
    ids = torch.tensor(
        [0 if x == "flat" else families.index(x) + 1 for x in rows], device=env.device
    )[slots]
    arrays = []
    for row in config["assets"]:
        for key in ("usd", "heights"):
            ref = row[key]
            if file_sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("terrain source SHA changed")
        with np.load(row["heights"]["path"], allow_pickle=False) as data:
            h = data["height_m"]
            if h.shape != (
                round(config["width_m"] / config["cell_m"]) + 1,
                round(config["length_m"] / config["cell_m"]) + 1,
            ):
                raise ValueError(
                    "Authored triangle query dimensions differ from terrain declaration"
                )
            if config.get("lateral_repetitions", 1) > 1 and (
                not np.array_equal(h[0], h[-1]) or not np.array_equal(h[:, 0], h[:, -1])
            ):
                raise ValueError("Tiled collision assets are not seam matched")
            arrays.append(torch.as_tensor(h, device=env.device, dtype=torch.float32))
    heights = torch.stack([torch.zeros_like(arrays[0]), *arrays])
    # Rough lanes have no additional slope plane; their relief is the terrain.
    rough = ids > 0
    env.DR_static_ground_angle[rough] = 0
    env.DR_static_ground_normal[rough] = ids.new_tensor([0, 0, 1]).float()
    env.scene.env_origins[rough, 2] = 0
    groups = [
        UsdPhysics.CollisionGroup.Get(stage, f"/World/collisions/env_{i}")
        for i in range(env.num_envs)
    ]
    plane_groups = [
        UsdPhysics.CollisionGroup(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.CollisionGroup) and "plane_" in str(prim.GetPath())
    ]
    for i, family in enumerate(ids.cpu().tolist()):
        if family == 0:
            continue
        # Remove both directions of plane membership; keep env self-collision.
        group_path = groups[i].GetPath()
        for plane_group in plane_groups:
            plane_group.GetFilteredGroupsRel().RemoveTarget(group_path)
            groups[i].GetFilteredGroupsRel().RemoveTarget(plane_group.GetPath())
        road_path = f"/World/MotionRoads/road_{i}"
        root = UsdGeom.Xform.Define(stage, road_path)
        origin = env.scene.env_origins[i].cpu().tolist()
        root.AddTranslateOp().Set(Gf.Vec3d(origin[0], origin[1] - config["spawn_x_m"], origin[2]))
        root.AddRotateZOp().Set(90.0)
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(
            UsdShade.Material.Get(stage, env.cfg.ground_material_path),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        for k, offset in enumerate(tile_offsets(config)):
            path = road_path + f"/tile_{k}"
            tile = UsdGeom.Xform.Define(stage, path)
            tile.GetPrim().GetReferences().AddReference(config["assets"][family - 1]["usd"]["path"])
            tile.AddTranslateOp(opSuffix="repeat").Set(Gf.Vec3d(*offset))
            # The verified USD already authors triangle CollisionAPI. Share its
            # prototype; do not expand/copy every mesh in Python.
            tile.GetPrim().SetInstanceable(True)
        collision = UsdPhysics.CollisionGroup.Define(stage, road_path + "/collision_group")
        collection = Usd.CollectionAPI.Apply(collision.GetPrim(), "colliders")
        collection.CreateExpansionRuleAttr("expandPrims")
        collection.CreateIncludesRel().AddTarget(Sdf.Path(road_path))
        collision.CreateFilteredGroupsRel().AddTarget(group_path)
        groups[i].CreateFilteredGroupsRel().AddTarget(collision.GetPath())
    env.motion_surface = TriangleRoads(
        heights,
        env.scene.env_origins,
        env.DR_static_ground_normal,
        ids,
        cell=config["cell_m"],
        length=config["length_m"],
        width=config["width_m"],
        repetitions=config["repetitions"],
        spawn_x=config["spawn_x_m"],
        lateral_repetitions=config.get("lateral_repetitions", 1),
    )
    env.motion_flat_terrain = ids == 0
    env.motion_terrain_ids = ids
    env.motion_road_exit = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.motion_terrain_height_valid = torch.ones_like(env.motion_road_exit)
    env.motion_terrain_config = config
    env.motion_terrain_readback = dict(
        families=families,
        rows=ids.cpu().tolist(),
        duplicate_plane_disabled=True,
        static_triangle_mesh=True,
        collision_approximation="none",
        height_query="same NPZ triangle diagonal; asset x mapped to world+Y",
        extent_m=[
            config["length_m"] * config["repetitions"],
            config["width_m"] * config.get("lateral_repetitions", 1),
        ],
        tile_instances_per_rough_env=len(tile_offsets(config)),
        instance_prototypes_shared=True,
        stage_mesh_count=sum(
            p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith("/World/MotionRoads")
            for p in stage.Traverse(Usd.TraverseInstanceProxies())
        ),
    )
