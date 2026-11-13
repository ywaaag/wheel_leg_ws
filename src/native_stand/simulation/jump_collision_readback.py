"""Startup collision evidence and geometric guards, separate from task observations.

Cooking count limits are necessary evidence, never a GPU narrowphase certificate.
Runtime-only imports stay inside the startup reader, after AppLauncher.
"""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from native_stand.artifacts.io import file_sha256, json_digest


def convex_mesh_readback(mesh):
    """Serialize the public cooked representation, including actual polygon data."""
    vertices = np.asarray([list(v) for v in mesh.vertices], dtype=float)
    indices = [int(i) for i in mesh.indices]
    polygons = [
        dict(
            index_base=int(p.index_base),
            num_vertices=int(p.num_vertices),
            plane=[float(x) for x in p.plane],
        )
        for p in mesh.polygons
    ]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("Invalid cooked convex vertices")
    if not np.isfinite(vertices).all() or not polygons:
        raise ValueError("Nonfinite or empty cooked hull")
    for polygon in polygons:
        start, count = polygon["index_base"], polygon["num_vertices"]
        if start < 0 or count < 3 or start + count > len(indices):
            raise ValueError("Invalid cooked polygon index range")
        if not np.isfinite(polygon["plane"]).all():
            raise ValueError("Nonfinite cooked polygon plane")
    if not indices or min(indices) < 0 or max(indices) >= len(vertices):
        raise ValueError("Invalid cooked vertex index")
    counts = dict(
        vertices=len(vertices),
        polygons=len(polygons),
        maximum_polygon_vertices=max(p["num_vertices"] for p in polygons),
    )
    result = dict(vertices=vertices.tolist(), indices=indices, polygons=polygons, counts=counts)
    result["geometry_sha256"] = json_digest(result)
    result["within_documented_GPU_count_limits"] = (
        counts["vertices"] <= 64
        and counts["polygons"] <= 64
        and counts["maximum_polygon_vertices"] <= 32
    )
    result["buildGPUData"] = "NOT_EXPOSED_BY_PUBLIC_REPRESENTATION_API"
    result["narrowphase_device"] = "UNVERIFIED"
    return result


def convex_clearance(body_positions, body_quaternions, local_vertices, inverse_rotate):
    """Lowest cooked vertex above the flat z=0 floor, using link-local vertices.

    One constant vertex tensor per wheel is shared across clones. Only rotations
    and support reductions run per physical tick; no cooking or CPU transfer.
    """
    down = torch.zeros_like(body_positions)
    down[..., 2] = -1
    local_down = inverse_rotate(body_quaternions, down)
    return torch.stack(
        [
            body_positions[:, wheel, 2] - (local_down[:, wheel] @ vertices.T).amax(-1)
            for wheel, vertices in enumerate(local_vertices)
        ],
        -1,
    )


def cooked_cylinder_envelope(mesh, radius, half, axis):
    """Measure the actual mesh rather than reuse an authored polygon error bound."""
    vertices = np.asarray(mesh["vertices"])
    planes = np.asarray([p["plane"] for p in mesh["polygons"]])
    normals = planes[:, :3]
    norms = np.linalg.norm(normals, axis=-1)
    if (norms <= 0).any():
        raise ValueError("Degenerate cooked hull plane")
    radial_axes = [i for i in range(3) if i != axis]
    cylinder_support = half * abs(normals[:, axis]) + radius * np.linalg.norm(
        normals[:, radial_axes], axis=-1
    )
    return dict(
        maximum_radial_excess_m=float(
            np.linalg.norm(vertices[:, radial_axes], axis=-1).max() - radius
        ),
        maximum_axial_excess_m=float(abs(vertices[:, axis]).max() - half),
        maximum_cylinder_outside_hull_plane_m=float(
            ((cylinder_support + planes[:, 3]) / norms).max()
        ),
    )


def guarded_clearance(measured):
    """Both analytic and cooked envelopes must pass the unchanged 1mm guard."""
    analytic = measured["clearance"]
    cooked = measured.get("collision_clearance")
    if cooked is None:
        return analytic
    if torch.is_tensor(analytic):
        return torch.minimum(analytic, cooked)
    return np.minimum(analytic, cooked)


def _host(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def capture_collision_readback(scene):
    """Read this initialized engine once; never copy evidence from a previous run."""
    import carb
    import omni.physx
    from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdPhysics, UsdUtils

    stage = scene.sim.stage
    cooking = omni.physx.get_physx_cooking_interface()
    stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    settings = carb.settings.get_settings()
    summary = dict(
        version="jump_collision_startup_readback_v1",
        engine_tick=scene.actual_ticks,
        reset_epoch=scene.reset_epoch,
        collisionApproximateCylinders=settings.get("/physics/collisionApproximateCylinders"),
        cooking_statistics="NOT_EXPOSED_BY_PUBLIC_COOKING_API; inspect this startup log",
        buildGPUData="NOT_EXPOSED_BY_PUBLIC_REPRESENTATION_API",
        narrowphase_device="UNVERIFIED; CUDA dynamics and count limits are insufficient",
        wheels=[],
        geometry_scope="first-clone cooked prototypes; every clone authored geometry verified",
        source_asset=scene.job_asset,
        runtime_versions={
            name: version(name) for name in ("isaacsim", "isaaclab", "isaaclab-physx")
        },
        source_files_sha256={
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("jump_collision_readback.py", "jump_isaac_scene.py", "jump_isaac.py")
        },
    )
    local_hulls = []
    custom_key = "physxCollisionCustomGeometry"
    # This is the sensor's public rigid-body view, whose prim_paths fixes row order.
    view = scene.sensor.body_physx_view
    actual = {
        "contact_offsets": _host(view.get_contact_offsets()),
        "rest_offsets": _host(view.get_rest_offsets()),
        "material_properties": _host(view.get_material_properties()),
    }
    simulation_view = scene.sensor._physics_sim_view
    floor = simulation_view.create_rigid_body_view("/World/ground")
    summary["ground_actual_shape_parameters"] = dict(
        body_paths=floor.prim_paths,
        contact_offsets=_host(floor.get_contact_offsets()).tolist(),
        rest_offsets=_host(floor.get_rest_offsets()).tolist(),
        material_properties=_host(floor.get_material_properties()).tolist(),
    )
    for wheel, side in enumerate(("left", "right")):
        body_path = f"/World/envs/env_0/Robot/{side}_wheel"
        path = body_path + "/collisions/collision_0/cylinder"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError("Missing actual wheel collider: " + path)
        body = stage.GetPrimAtPath(body_path)
        frame = UsdGeom.XformCache()
        relative = (
            frame.GetLocalToWorldTransform(prim) * frame.GetLocalToWorldTransform(body).GetInverse()
        )
        row = dict(
            path=path,
            USD_type=prim.GetTypeName(),
            collision_enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
            physxCollisionCustomGeometry=prim.GetCustomDataByKey(custom_key),
            local_transform_row_vector_convention=[list(v) for v in relative],
            cooked_status="NOT_A_MESH; native cylinder route requires runtime evidence",
        )
        if prim.HasMetadata(custom_key):
            row["physxCollisionCustomGeometry"] = prim.GetMetadata(custom_key)
        row["native_colliders"] = []

        def collider_readback(collider, row=row):
            row["native_colliders"].append(
                dict(
                    path=str(PhysicsSchemaTools.intToSdfPath(collider.path_id)),
                    result=str(collider.result),
                    local_position=list(collider.local_pos),
                    local_quaternion_xyzw=list(collider.local_rot),
                    AABB_min=list(collider.aabb_local_min),
                    AABB_max=list(collider.aabb_local_max),
                )
            )

        omni.physx.get_physx_property_query_interface().query_prim(
            stage_id=stage_id,
            prim_id=PhysicsSchemaTools.sdfPathToInt(Sdf.Path(body_path)),
            timeout_ms=10000,
            collider_fn=collider_readback,
        )
        native = [c for c in row["native_colliders"] if c["path"] == path]
        if len(native) != 1 or not native[0]["result"].endswith(".VALID"):
            raise RuntimeError("Native wheel collision transform unavailable: " + path)
        rb = scene.robot.root_view
        body_id = scene.wheel_ids[wheel]
        row.update(
            mass_kg=_host(rb.get_masses())[0, body_id].tolist(),
            inertia=_host(rb.get_inertias())[0, body_id].tolist(),
            COM=_host(rb.get_coms())[0, body_id].tolist(),
        )
        rows = [i for i, p in enumerate(view.prim_paths) if p.endswith(f"/{side}_wheel")]
        row["actual_shape_parameters"] = {k: values[rows].tolist() for k, values in actual.items()}
        row["actual_shape_parameter_rows"] = rows
        row["actual_shape_parameter_layout"] = (
            "rigid-body view rows x padded max_shapes; native_colliders gives actual shapes"
        )
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            authored = dict(
                points=[list(p) for p in mesh.GetPointsAttr().Get()],
                counts=list(mesh.GetFaceVertexCountsAttr().Get()),
                indices=list(mesh.GetFaceVertexIndicesAttr().Get()),
            )
            row["authored_geometry_sha256"] = json_digest(authored)
            for clone in range(scene.num_envs):
                candidate = UsdGeom.Mesh(
                    stage.GetPrimAtPath(path.replace("env_0/", f"env_{clone}/"))
                )
                if (
                    [list(p) for p in candidate.GetPointsAttr().Get()] != authored["points"]
                    or list(candidate.GetFaceVertexCountsAttr().Get()) != authored["counts"]
                    or list(candidate.GetFaceVertexIndicesAttr().Get()) != authored["indices"]
                ):
                    raise ValueError("Clone wheel collision geometry differs")
            result = {}

            def cooked(status, meshes, result=result):
                result["status"] = str(status)
                result["meshes"] = [convex_mesh_readback(m) for m in meshes]

            cooking.request_convex_collision_representation(
                stage_id, PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path)), False, cooked
            )
            if (
                not result.get("status", "").endswith("RESULT_VALID")
                or len(result.get("meshes", [])) != 1
            ):
                raise RuntimeError(
                    "Actual cooked wheel hull unavailable: " + str(result.get("status"))
                )
            row["cooked_status"] = result["status"]
            row["cooked_mesh"] = result["meshes"][0]
            source_shape = next(
                g
                for g in scene.geometry["collision_shapes"]
                if g["body"] == side + "_wheel" and g["kind"] == "Cylinder"
            )
            row["cooked_cylinder_envelope"] = cooked_cylinder_envelope(
                row["cooked_mesh"],
                source_shape["radius"],
                source_shape["half_height"],
                source_shape["axis"],
            )
            envelope = row["cooked_cylinder_envelope"]
            if (
                envelope["maximum_radial_excess_m"] > 0.000331
                or envelope["maximum_axial_excess_m"] > 2e-6
                or envelope["maximum_cylinder_outside_hull_plane_m"] > 2e-6
            ):
                raise RuntimeError(
                    "Actual cooked wheel envelope differs from declared cylinder approximation"
                )
            vertices = [
                list(relative.Transform(Gf.Vec3d(*p))) for p in row["cooked_mesh"]["vertices"]
            ]
            authored_frame_vertices = torch.tensor(
                vertices, dtype=torch.float32, device=scene.device
            )
            points = torch.tensor(
                row["cooked_mesh"]["vertices"], dtype=torch.float32, device=scene.device
            )
            q = points.new_tensor(native[0]["local_quaternion_xyzw"]).expand(len(points), -1)
            native_vertices = scene.apply(q, points) + points.new_tensor(
                native[0]["local_position"]
            )
            if not torch.allclose(native_vertices, authored_frame_vertices, atol=2e-6, rtol=0):
                raise RuntimeError("Native/authored wheel collision transform differs")
            local_hulls.append(native_vertices)
        summary["wheels"].append(row)
    # Do not silently use an authored hull when the actual cooked readback failed.
    if local_hulls and len(local_hulls) != 2:
        raise RuntimeError("Incomplete cooked wheel coverage")
    return summary, local_hulls
