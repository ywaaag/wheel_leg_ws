"""CPU-only derived H1 assets and successor job; never creates/reserves a run."""

from __future__ import annotations

import copy
from pathlib import Path


def prepare(root):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    from native_stand.artifacts.io import file_sha256, read_json, write_json
    from native_stand.experiments.config import job_from_record, load_job_config
    from native_stand.simulation.terrain_layout import periodic_edges
    from native_stand.tasks.motion.branch_state import confirmation_execution_identity
    from native_stand.tasks.motion.contact_randomization import CRITIC_VERSION, VERSION

    original = root / "configs/jobs/2026_11_12__motion_u3000_bank_height020_H1_cloud10000.json"
    job = load_job_config(original)
    record = copy.deepcopy(job.as_run_record())
    record["name"] += "_contact_grid_v2"
    record["assets"]["resource_mode"] = "exclusive"
    record["assets"]["wheel_contact_dr"] = dict(
        version=VERSION,
        sampling="reset_only",
        profiles=[0],
        compliant_probability=0.5,
        stiffness_n_m=[100000.0, 1000000.0],
        damping_ns_m=[500.0, 3000.0],
        units="native force spring N/m and Ns/m per contact; not acceleration spring",
        basis="engineering hypothesis, not real wheel-ground identification",
        nominal="rigid; no compliant-contact sampling or policy RNG consumption",
        enable_scope="half of combined DR rows, other cohorts retain rigid contact",
        ground="rigid; sole compliant wheel material determines normal spring-damper",
    )
    record["assets"]["contact_critic_extension"] = dict(
        **{"from": "motion48_height_bank_planned_causal877_v2", "to": CRITIC_VERSION}
    )
    record["model"].update(critic_input=881, critic_privileged_version=CRITIC_VERSION)
    record["observation"]["critic_dim"] = 881
    terrain = record["assets"]["terrain_scene"]
    terrain.update(
        version="motion_H1_periodic_grid_scene_v2",
        repetitions=25,
        lateral_repetitions=3,
        spawn_x_m=12.1,
        seam_contract="periodic_C0_edges_v1",
        horizon_guard=dict(episode_s=30.0, forward_speed_mps=3.6, distance_factor=1.2),
        critic_context="881; vertical root height over actual local collision triangle; actual wheel K/C",
    )
    derived = root / "assets/terrains/motion_H1_periodic_v2"
    rows = []
    for asset in terrain["assets"]:
        for ref in (asset["usd"], asset["heights"]):
            if file_sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("Original H1 source changed")
        directory = derived / asset["family"]
        directory.mkdir(parents=True, exist_ok=True)
        with np.load(asset["heights"]["path"], allow_pickle=False) as data:
            # Collision USD points are float32. Preserve that physical surface
            # exactly in the interior, rather than claim float64 NPZ identity.
            old = data["height_m"].astype(np.float32)
            h = periodic_edges(old, seam_cells=10)
            arrays = {k: data[k] for k in data.files}
        arrays["height_m"] = h
        heights = directory / "heightfield.npz"
        np.savez_compressed(heights, **arrays)
        usd = directory / "terrain.usdc"
        stage = Usd.Stage.Open(asset["usd"]["path"])
        stage.GetRootLayer().Export(str(usd))
        stage = Usd.Stage.Open(str(usd))
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        if len(meshes) != 1:
            raise ValueError("Expected one authored H1 collision mesh")
        mesh = UsdGeom.Mesh(meshes[0])
        points = np.array(mesh.GetPointsAttr().Get())
        ix = np.rint(points[:, 0] / terrain["cell_m"]).astype(int)
        iy = np.rint((points[:, 1] + terrain["width_m"] / 2) / terrain["cell_m"]).astype(int)
        points[:, 2] = h[iy, ix]
        mesh.GetPointsAttr().Set(points)
        mesh.GetExtentAttr().Set(
            [Gf.Vec3f(*map(float, points.min(0))), Gf.Vec3f(*map(float, points.max(0)))]
        )
        stage.GetRootLayer().Save()
        row = dict(
            family=asset["family"],
            usd=dict(path=str(usd), sha256=file_sha256(usd)),
            heights=dict(path=str(heights), sha256=file_sha256(heights)),
        )
        rows.append(row)
        write_json(
            directory / "derivation.json",
            dict(
                source=asset,
                output=row,
                seam_strip_each_side_m=0.2,
                unchanged_interior_fraction=float(np.mean(h[10:-10] == old[10:-10])),
                interior_comparison="source physical USD float32 precision",
                maximum_change_m=float(np.max(np.abs(h - old))),
                physics_steps=0,
            ),
        )
    terrain["assets"] = rows
    course = record["assets"]["branch_course"]
    course["confirmation_execution_sha256"] = confirmation_execution_identity(
        job_from_record(record)
    )
    record["curriculum"]["stages"] = [course]
    # Carry exactly the same unspent10000 authorization into one successor
    # declaration. Old files and failed-run consumption are never rewritten.
    branch = read_json(job.assets["branch_initialization"]["path"])
    grant = read_json(branch["new_authorization"]["path"])
    allocation = read_json(branch["allocation"]["path"])
    grant.update(
        run_id=record["name"],
        supersedes_unstarted_job=dict(path=str(original), sha256=file_sha256(original)),
        scope_extension="user: H1 area tiling, ground-relative height, wheel contact compliance/damping DR; same10000, no extra budget",
    )
    suffix = "motion_contact_grid10000"
    grant_path = root / f"configs/branches/{suffix}_authorization.json"
    write_json(grant_path, grant)
    branch["new_authorization"] = dict(path=str(grant_path), sha256=file_sha256(grant_path))
    for entry in allocation["allocations"]:
        entry.update(
            run_id=record["name"], execution_sha256=course["confirmation_execution_sha256"]
        )
    allocation["supersedes_unstarted_job"] = grant["supersedes_unstarted_job"]
    allocation_path = root / f"configs/branches/{suffix}_allocation.json"
    write_json(allocation_path, allocation)
    branch["allocation"] = dict(path=str(allocation_path), sha256=file_sha256(allocation_path))
    branch_path = root / f"configs/branches/{suffix}_initialization.json"
    write_json(branch_path, branch)
    record["assets"]["branch_initialization"] = dict(
        path=str(branch_path), sha256=file_sha256(branch_path)
    )
    output = root / f"configs/jobs/{record['name']}.json"
    record.pop("config_path", None)
    record.pop("config_sha256", None)
    write_json(output, record)
    print(output)


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
