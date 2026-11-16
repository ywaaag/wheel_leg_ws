"""Full authored closed-chain kinematics. All geometry is explicit input data.

IK changes reference coordinates only. Never project an integrated robot onto
this solution. CAD forward=-Z, up=Y; model left is geometric right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

CHANNELS = (
    "left_hip_pitch",
    "left_active_link_pitch",
    "left_wheel_spin",
    "right_hip_pitch",
    "right_active_link_pitch",
    "right_wheel_spin",
)
LEGS = (0, 1, 3, 4)
SIDES = ("left", "right")


def transform(position, rotation):
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = position
    return out


def points(pose, local):
    return np.asarray(local) @ pose[:3, :3].T + pose[:3, 3]


@dataclass
class Shape:
    q: dict
    poses: dict
    length: np.ndarray
    swing: np.ndarray
    spring: np.ndarray
    knee: np.ndarray
    closure_m: float
    COM_local_m: np.ndarray
    root_support_height_m: float
    wheel_bottom_local_m: np.ndarray
    nonwheel_ground_clearance_m: float
    stop_margin_m: float


class ClosedChain:
    def __init__(self, description, springs, *, stop_margin_m=0.002):
        self.description = description
        self.joints = description["joints"]
        self.loops = description["loops"]
        self.springs = springs
        self.margin = float(stop_margin_m)
        self.by_name = {j["name"]: j for j in self.joints}
        self.leg_names = [
            j["name"]
            for j in self.joints
            if j["name"].startswith(("left_", "right_")) and "wheel" not in j["name"]
        ]
        self.passive = [n for n in self.leg_names if n not in CHANNELS]
        self.low = np.array([self.by_name[n]["limits_rad"][0] for n in self.leg_names])
        self.high = np.array([self.by_name[n]["limits_rad"][1] for n in self.leg_names])
        self._leg_chain = None

    def leg_lengths(self, q):
        """Same wheel-centre FK as measure, without unrelated shape/COM work.

        Preserve the original matrix multiplication order. A single SciPy
        rotation conversion handles the independent ancestor joints in batch.
        The full authored FK/IK/shape routines remain the reference consumers.
        """
        if self._leg_chain is None:
            by_child = {j["child"]: j for j in self.joints}
            needed = set()
            for side in SIDES:
                child, visited = side + "_wheel", set()
                while child != "robot":
                    if child in visited or child not in by_child:
                        raise ValueError("Disconnected authored joint tree")
                    visited.add(child)
                    joint = by_child[child]
                    needed.add(joint["name"])
                    child = joint["parent"]
            pending = [j for j in self.joints if j["name"] in needed]
            known, chain = {"robot"}, []
            while pending:
                later = []
                for joint in pending:
                    if joint["parent"] not in known:
                        later.append(joint)
                        continue
                    chain.append(
                        (joint, np.asarray(joint["frame0"]), np.asarray(joint["inverse_frame1"]))
                    )
                    known.add(joint["child"])
                if len(later) == len(pending):
                    raise ValueError("Disconnected authored joint tree")
                pending = later
            self._leg_chain = chain
            self._leg_hips = [
                points(np.eye(4), self.by_name[side + "_hip_pitch"]["frame0_local_position"])
                for side in SIDES
            ]
        vectors = np.zeros((len(self._leg_chain), 3))
        for i, (joint, _, _) in enumerate(self._leg_chain):
            vectors[i, joint["axis"]] = q.get(joint["name"], 0.0)
        rotations = Rotation.from_rotvec(vectors).as_matrix()
        poses = {"robot": np.eye(4)}
        for (joint, frame0, frame1), rotation in zip(self._leg_chain, rotations, strict=True):
            poses[joint["child"]] = (
                poses[joint["parent"]] @ frame0 @ transform([0, 0, 0], rotation) @ frame1
            )
        lengths = []
        for side, hip in zip(SIDES, self._leg_hips, strict=True):
            vector = poses[side + "_wheel"][:3, 3] - hip
            lengths.append(np.hypot(vector[1], vector[2]))
        return np.asarray(lengths)

    def forward(self, q):
        poses = {"robot": np.eye(4)}
        pending = list(self.joints)
        while pending:
            later = []
            for j in pending:
                if j["parent"] not in poses:
                    later.append(j)
                    continue
                axis = np.zeros(3)
                axis[j["axis"]] = q.get(j["name"], 0.0)
                poses[j["child"]] = (
                    poses[j["parent"]]
                    @ np.asarray(j["frame0"])
                    @ transform([0, 0, 0], Rotation.from_rotvec(axis).as_matrix())
                    @ np.asarray(j["inverse_frame1"])
                )
            if len(later) == len(pending):
                raise ValueError("Disconnected authored joint tree")
            pending = later
        return poses

    def closure(self, poses):
        return np.concatenate(
            [
                points(poses[j["parent"]], j["anchor0"]) - points(poses[j["child"]], j["anchor1"])
                for j in self.loops
            ]
        )

    def measure(self, q):
        poses = self.forward(q)
        lengths, swings, spring_lengths, knees = [], [], [], []
        for side, spring in zip(SIDES, self.springs, strict=False):
            hip = points(poses["robot"], self.by_name[side + "_hip_pitch"]["frame0_local_position"])
            vector = poses[side + "_wheel"][:3, 3] - hip
            lengths.append(np.hypot(vector[1], vector[2]))
            swings.append(np.arctan2(-vector[2], -vector[1]))
            ends = [
                points(poses[a["body"].rsplit("/", 1)[-1]], a["local_m"]) for a in spring["anchors"]
            ]
            spring_lengths.append(np.linalg.norm(ends[1] - ends[0]))
            knees.append(q.get(side + "_knee_pitch", 0.0))
        # Support functions of the actual USD collision primitives, including
        # rotated wheel cylinders and all 24 guide wheels. CAD Y is upright Z.
        bottoms = {}
        for g in self.description["collision_shapes"]:
            p = poses[g["body"]] @ np.asarray(g["body_frame"])
            center = p[1, 3]
            direction = p[1, :3]
            kind = g["kind"]
            if kind == "Sphere":
                extent = g["radius"]
            elif kind == "Cube":
                extent = np.abs(direction) @ np.asarray(g["half_size"])
            else:
                axial = abs(direction[g["axis"]])
                radial = g["radius"] * np.sqrt(max(0.0, 1 - axial**2))
                extent = g["half_height"] * axial + (g["radius"] if kind == "Capsule" else radial)
            bottoms.setdefault(g["body"], []).append(float(center - extent))
        wheel = np.array([min(bottoms[s + "_wheel"]) for s in SIDES])
        root_height = -float(wheel.min())
        other = min(
            b
            for name, bs in bottoms.items()
            if name not in ("left_wheel", "right_wheel")
            for b in bs
        )
        masses = self.description["bodies"]
        total = sum(b["mass_kg"] for b in masses)
        com = sum(b["mass_kg"] * points(poses[b["name"]], b["COM_local_m"]) for b in masses) / total
        spring = np.asarray(spring_lengths)
        return Shape(
            dict(q),
            poses,
            np.asarray(lengths),
            np.asarray(swings),
            spring,
            np.asarray(knees),
            float(np.linalg.norm(self.closure(poses).reshape(-1, 3), axis=1).max()),
            com,
            root_height,
            wheel,
            other + root_height,
            float(np.minimum(spring - 0.155, 0.215 - spring).min()),
        )

    def solve(self, length, swing=(0.0, 0.0), *, seed=None, active=None, check=True):
        """IK in exact loop tree, or FK by fixing four actual active angles.

        The branch is continuous from the seed; the legal spring branch contains
        q_knee=0. The disconnected post-compression branch is always rejected.
        """
        length = np.broadcast_to(np.asarray(length, float), (2,))
        swing = np.broadcast_to(np.asarray(swing, float), (2,))
        if not np.isfinite(np.r_[length, swing]).all() or (length <= 0).any():
            raise ValueError("Finite positive sagittal length and swing required")
        q = {j["name"]: 0.0 for j in self.joints}
        if seed is not None:
            q.update(seed)
        names = self.leg_names if active is None else self.passive
        if active is not None:
            a = np.asarray(active, float)
            if a.shape != (4,) or not np.isfinite(a).all():
                raise ValueError("Four finite active angles")
            q.update(zip([CHANNELS[i] for i in LEGS], a, strict=False))
        lo = np.array([self.by_name[n]["limits_rad"][0] for n in names])
        hi = np.array([self.by_name[n]["limits_rad"][1] for n in names])
        x0 = np.maximum(lo + 1e-9, np.minimum(hi - 1e-9, [q[n] for n in names]))

        def residual(x):
            q.update(zip(names, x, strict=False))
            poses = self.forward(q)
            out = [self.closure(poses)]
            if active is None:
                for i, side in enumerate(SIDES):
                    hip = points(
                        poses["robot"], self.by_name[side + "_hip_pitch"]["frame0_local_position"]
                    )
                    v = poses[side + "_wheel"][:3, 3] - hip
                    out.append(v[1:3] + length[i] * np.array([np.cos(swing[i]), np.sin(swing[i])]))
            return np.concatenate(out)

        fit = least_squares(
            residual, x0, bounds=(lo, hi), max_nfev=100, ftol=1e-12, xtol=1e-12, gtol=1e-12
        )
        residual(fit.x)
        result = self.measure(q)
        if result.closure_m > 2e-5:
            raise ValueError("Unclosed or unreachable branch")
        if (
            active is None
            and max(abs(result.length - length).max(), abs(result.swing - swing).max()) > 2e-4
        ):
            raise ValueError("IK target unreachable")
        if check:
            self.validate(result)
        return result

    def validate(self, shape):
        if shape.closure_m > 2e-5:
            raise ValueError("Loop closure")
        for n in self.leg_names:
            lo, hi = self.by_name[n]["limits_rad"]
            if not lo - 1e-8 <= shape.q[n] <= hi + 1e-8:
                raise ValueError("Authored joint limit")
        if not (
            -0.233001 <= shape.knee[0] <= 1.12987845 and -1.12987845 <= shape.knee[1] <= 0.233001
        ):
            raise ValueError("Disconnected or spring-compressed knee branch")
        if shape.stop_margin_m < self.margin:
            raise ValueError("Spring stop margin")
        if shape.nonwheel_ground_clearance_m < 0.01:
            raise ValueError("Chassis/guide/leg ground clearance")

    def jacobian(self, shape, step=1e-5):
        """Exact closure finite differences: [L0,theta0,L1,theta1] / active4."""
        a = np.array([shape.q[CHANNELS[i]] for i in LEGS])
        J = np.zeros((4, 4))
        S = np.zeros((2, 4))
        for j in range(4):
            d = np.eye(4)[j] * step
            p = self.solve(shape.length, shape.swing, seed=shape.q, active=a + d, check=False)
            m = self.solve(shape.length, shape.swing, seed=shape.q, active=a - d, check=False)
            J[:, j] = (
                np.array([p.length[0], p.swing[0], p.length[1], p.swing[1]])
                - np.array([m.length[0], m.swing[0], m.length[1], m.swing[1]])
            ) / (2 * step)
            S[:, j] = (p.spring - m.spring) / (2 * step)
        if not np.isfinite(J).all() or np.linalg.cond(J) > 1e4:
            raise ValueError("Singular closed-chain shape Jacobian")
        return J, S


class ShapeTable:
    """Offline full-chain IK baked as piecewise cubics, zero swing first task.

    This is a reference lookup, not a dynamic state constraint or controller.
    Both sides retain their authored geometry. Every interpolated pose is checked.
    Firmware can evaluate the exported cubic coefficients without an IK solver.
    """

    def __init__(self, geometry, lengths, coordinates):
        from scipy.interpolate import PchipInterpolator

        self.g = geometry
        self.lengths = np.asarray(lengths, float)
        self.coordinates = np.asarray(coordinates, float)
        self.names = geometry.leg_names
        self.curve = PchipInterpolator(self.lengths, self.coordinates, axis=0, extrapolate=False)

    @classmethod
    def bake(cls, geometry, lower=0.18, upper=0.36, step=0.002):
        lengths = np.linspace(lower, upper, round((upper - lower) / step) + 1)
        rows = []
        seed = None
        for length in lengths:
            s = geometry.solve(length, seed=seed)
            seed = s.q
            rows.append([s.q[n] for n in geometry.leg_names])
        return cls(geometry, lengths, rows)

    def solve(self, length, swing=(0.0, 0.0), **unused):
        ls = np.broadcast_to(length, (2,))
        angle = np.broadcast_to(swing, (2,))
        if (
            np.any(abs(angle) > 1e-5)
            or np.any(ls < self.lengths[0])
            or np.any(ls > self.lengths[-1])
        ):
            raise ValueError("Outside frozen zero-swing reference table")
        q = {j["name"]: 0.0 for j in self.g.joints}
        for i, side in enumerate(SIDES):
            vals = self.curve(ls[i])
            q.update(
                {n: float(vals[k]) for k, n in enumerate(self.names) if n.startswith(side + "_")}
            )
        s = self.g.measure(q)
        self.g.validate(s)
        return s

    def measure(self, q):
        return self.g.measure(q)

    def contract(self):
        return dict(
            length_knots_m=self.lengths.tolist(),
            joint_names=self.names,
            coordinates_rad=self.coordinates.tolist(),
            coefficients_rad_per_m_power=self.curve.c.tolist(),
            polynomial="sum(c[k,interval,j]*(L-knot[interval])**(3-k),k=0..3)",
            swing_rad=0.0,
            extrapolation=False,
            kind="full_USD_closed_chain_IK_reference_lookup_not_dynamics",
        )
