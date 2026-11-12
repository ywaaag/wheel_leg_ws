"""Vertical primitive support points for terrain-clearance diagnostics.

The ground must be queried at the support point's XY, never at the body origin.
This is a vertical-clearance proxy, not a replacement for solver collision tests.
Quaternion operations come from the running SDK, so its convention is preserved.
"""

import torch


def vertical_support_points(helper, positions, quaternions, quat_inverse, quat_apply):
    q = quaternions[:, helper.ids]
    up = q.new_tensor([0.0, 0.0, 1.0]).expand(*q.shape[:-1], 3)
    body_up = quat_inverse(q, up)
    normal = torch.einsum("nsi,sij->nsj", body_up, helper.rotation)
    axis_mask = torch.nn.functional.one_hot(helper.axis, 3).bool()[None]
    unit = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    radial = normal.masked_fill(axis_mask, 0.0)
    radial = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axial = -torch.sign(normal) * axis_mask * helper.height[None, :, None]
    cube = -torch.sign(normal) * helper.size[None, :, None]
    sphere = -unit * helper.radius[None, :, None]
    capsule = sphere + axial
    cylinder = -radial * helper.radius[None, :, None] + axial
    kind = helper.kind[None, :, None]
    local = torch.where(
        kind == 0, cube, torch.where(kind == 1, sphere, torch.where(kind == 2, capsule, cylinder))
    )
    body_point = helper.translation + torch.einsum("sij,nsj->nsi", helper.rotation, local)
    return positions[:, helper.ids] + quat_apply(q, body_point)
