"""Batched exact triangle query; asset loading belongs to the scene boundary."""

import torch


class TriangleRoads:
    def __init__(
        self,
        heights,
        origins,
        normals,
        family_ids,
        *,
        cell,
        length,
        width,
        repetitions,
        spawn_x,
        lateral_repetitions=1,
    ):
        self.heights, self.origins, self.normals = heights, origins, normals
        self.family_ids = family_ids
        self.cell, self.length, self.width = cell, length, width
        self.repetitions, self.spawn_x = repetitions, spawn_x
        self.lateral_repetitions = lateral_repetitions

    def contains(self, xy, *, margin=0.0):
        shape = (len(self.origins),) + (1,) * (xy.ndim - 2)
        origin = self.origins.reshape(*shape, 3)
        family = self.family_ids.reshape(shape)
        x = xy[..., 1] - origin[..., 1] + self.spawn_x
        y = -(xy[..., 0] - origin[..., 0])
        inside = (
            (x >= margin)
            & (x <= self.length * self.repetitions - margin)
            & (y.abs() <= self.width * self.lateral_repetitions / 2 - margin)
        )
        return (inside | (family == 0)) & torch.isfinite(xy).all(-1)

    def sample(self, xy, *, check=True):
        if xy.shape[0] != len(self.origins) or xy.shape[-1] != 2:
            raise ValueError("surface query must preserve environment rows")
        shape = (len(self.origins),) + (1,) * (xy.ndim - 2)
        origin = self.origins.reshape(*shape, 3)
        normal = self.normals.reshape(*shape, 3)
        family = self.family_ids.reshape(shape).expand(xy.shape[:-1])
        # Asset x is longitudinal; robot forward at the legal reset is world+Y.
        x = xy[..., 1] - origin[..., 1] + self.spawn_x
        y = -(xy[..., 0] - origin[..., 0])
        valid = self.contains(xy)
        if check and not bool((valid | (family == 0)).all()):
            raise RuntimeError("actual terrain query left its finite road; no flat fallback")
        # check=False is only for explicit terminal handling; never extrapolate
        # a fictitious terrain beyond its authored collision surface.
        x = x.nan_to_num().clamp(0, self.length * self.repetitions).remainder(self.length)
        half_width = self.width * self.lateral_repetitions / 2
        y = y.nan_to_num().clamp(-half_width, half_width)
        if self.lateral_repetitions > 1:
            y = (y + half_width).remainder(self.width) - self.width / 2
        gx = x / self.cell
        gy = (y + self.width / 2) / self.cell
        nx, ny = self.heights.shape[2], self.heights.shape[1]
        ix, iy = gx.long().clamp(0, nx - 2), gy.long().clamp(0, ny - 2)
        u, v = gx - ix, gy - iy
        a = self.heights[family, iy, ix]
        b = self.heights[family, iy, ix + 1]
        c = self.heights[family, iy + 1, ix]
        d = self.heights[family, iy + 1, ix + 1]
        first = u + v <= 1
        z = torch.where(
            first, a + u * (b - a) + v * (c - a), d + (1 - u) * (c - d) + (1 - v) * (b - d)
        )
        dx = torch.where(first, b - a, d - c) / self.cell
        dy = torch.where(first, c - a, d - b) / self.cell
        plane_z = (
            origin[..., 2] - ((xy - origin[..., :2]) * normal[..., :2]).sum(-1) / normal[..., 2]
        )
        plane_gradient = -normal[..., :2] / normal[..., 2:3]
        gradient = torch.stack((-dy, dx), -1)
        return (
            torch.where(family > 0, z + origin[..., 2], plane_z),
            torch.where((family > 0)[..., None], gradient, plane_gradient),
        )
