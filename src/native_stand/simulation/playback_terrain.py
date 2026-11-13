"""Single finite manual-test atlas, authored from SHA-bound H1 collision grids.

USD collision points/faces are checked against the NPZ height grid/topology.
The stale NPZ vertices copy is not used. MuJoCo uses its own
heightfield triangulation; telemetry queries the actual collision via mj_ray,
never a fictitious plane or an independent interpolation. The engine difference
is exposed rather than claiming PhysX triangle/contact equivalence.
"""

import xml.etree.ElementTree as ET

import numpy as np

from native_stand.artifacts.io import file_sha256

FAMILIES = ("steep_waves_H1", "cobblestones_H1", "broken_rubble_H1", "pitted_road_H1")


class Atlas:
    def __init__(self, grids, cell):
        self.cell = cell
        self.x0, self.y0 = -8.0, -3.0
        self.width, self.length = 16.0, 15.0
        self.z = np.zeros((round(self.length / cell) + 1, round(self.width / cell) + 1), np.float32)
        self.spawns = {"flat": [0.0, 0.0]}
        self.bounds = {}
        for family, center, h in zip(FAMILIES, (-6.0, -2.0, 2.0, 6.0), grids, strict=True):
            # Authored asset x -> world +Y; asset y -> world -X.
            x, y = center - 1.5, 3.0
            ix, iy = round((x - self.x0) / cell), round((y - self.y0) / cell)
            rotated = h.T[:, ::-1]
            self.z[iy : iy + rotated.shape[0], ix : ix + rotated.shape[1]] = rotated
            self.spawns[family] = [center, y + 0.1]
            self.bounds[family] = (x, x + 3.0, y, y + 6.0)
        self.model = self.data = None

    def author(self, xml, world, scene):
        asset = ET.SubElement(xml, "asset")
        lo, hi = float(self.z.min()), float(self.z.max())
        self.low, self.relief = lo, max(hi - lo, 0.001)
        ET.SubElement(
            asset,
            "hfield",
            name="H1_atlas",
            nrow=str(self.z.shape[0]),
            ncol=str(self.z.shape[1]),
            size=f"8 7.5 {self.relief} .1",
        )
        ET.SubElement(
            world,
            "geom",
            name="ground",
            type="hfield",
            hfield="H1_atlas",
            pos=f"0 4.5 {lo}",
            contype="2",
            conaffinity="1",
            friction=f"{scene.friction} .005 .0001",
            solref=".008 1",
            rgba=".36 .44 .36 1",
        )
        for family, (x0, x1, y0, _y1) in self.bounds.items():
            ET.SubElement(
                world,
                "site",
                name=family,
                pos=f"{(x0 + x1) / 2} {y0} .08",
                type="box",
                size="1.5 .015 .015",
                rgba=".9 .6 .1 1",
            )
        ET.SubElement(world, "light", pos="0 0 8", dir="0 0 -1", directional="true")

    def install(self, model):
        import mujoco

        model.hfield_data[:] = ((self.z - self.low) / self.relief).ravel()
        model.geom_group[model.geom_bodyid != 0] = 1
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)

    def height(self, xy):
        import mujoco

        xy = np.asarray(xy)
        self.check_bounds(xy)
        if self.model is None:
            raise ValueError("Collision height query requires installed MuJoCo model")
        heights = []
        group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        for point in xy.reshape(-1, 2):
            geomid = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                np.r_[point, 2.0],
                np.array([0.0, 0.0, -1.0]),
                group,
                True,
                -1,
                geomid,
            )
            if distance < 0 or geomid[0] != mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
            ):
                raise ValueError("H1碰撞高度不可用")
            heights.append(2.0 - distance)
        return np.array(heights).reshape(xy.shape[:-1])

    def check_bounds(self, xy):
        xy = np.asarray(xy)
        if (
            not np.isfinite(xy).all()
            or (xy[..., 0] < self.x0).any()
            or (xy[..., 0] > self.x0 + self.width).any()
            or (xy[..., 1] < self.y0).any()
            or (xy[..., 1] > self.y0 + self.length).any()
        ):
            raise ValueError("离开有限H1碰撞地图；没有平面外推，请重置")

    def region(self, xy):
        try:
            self.check_bounds(xy)
        except ValueError:
            return "outside_collision_map"
        for name, (x0, x1, y0, y1) in self.bounds.items():
            if x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1:
                return name
        return "flat"


def load_atlas(config):
    if (
        tuple(config["families"]) != FAMILIES
        or config["cell_m"] != 0.02
        or config["length_m"] != 6.0
        or config["width_m"] != 3.0
    ):
        raise ValueError("H1地图要求明确的四种6×3m、2cm碰撞网格")
    grids = []
    for family, row in zip(FAMILIES, config["assets"], strict=True):
        if row["family"] != family:
            raise ValueError("H1地形顺序不一致")
        for key in ("heights", "usd"):
            ref = row[key]
            if file_sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("H1地形SHA不一致: " + family)
        with np.load(row["heights"]["path"], allow_pickle=False) as d:
            h = d["height_m"]
            # The periodic derivative's NPZ vertices_m is historical. Actual
            # authored USD collision points, not that stale copy, are authority.
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = Usd.Stage.Open(row["usd"]["path"])
            meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
            if len(meshes) != 1 or not meshes[0].GetPrim().HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError("H1需要单一真实USD碰撞网格")
            mesh = meshes[0]
            vertices = np.asarray(mesh.GetPointsAttr().Get())
            triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            if not np.array_equal(triangles, d["triangles"]):
                raise ValueError("USD与NPZ碰撞拓扑不一致")
            x, y = d["x_m"], d["y_m"]
            xx, yy = np.meshgrid(x, y)
            if (
                h.shape != (151, 301)
                or not np.isfinite(h).all()
                or not np.allclose(np.diff(x), 0.02, atol=1e-6)
                or not np.allclose(np.diff(y), 0.02, atol=1e-6)
                or not np.allclose(vertices, np.stack((xx, yy, h), -1).reshape(-1, 3), atol=1e-6)
            ):
                raise ValueError("H1高度与碰撞顶点不一致")
            a = (np.arange(150)[:, None] * 301 + np.arange(300)[None]).ravel()
            faces = np.stack(
                (np.stack((a, a + 1, a + 301), -1), np.stack((a + 1, a + 302, a + 301), -1)), 1
            ).reshape(-1, 3)
            if not np.array_equal(triangles, faces):
                raise ValueError("H1碰撞三角拓扑不支持")
            grids.append(h.copy())
    return Atlas(grids, config["cell_m"])
