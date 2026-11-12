"""Runtime readback for the exclusive six-joint effort control path."""

import torch

from native_stand.contract import JOINT_NAMES


def tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    from isaaclab.utils.warp import ProxyArray

    if isinstance(value, ProxyArray):
        return value.torch.detach().cpu().tolist()
    import warp as wp

    if isinstance(value, wp.array):
        return wp.to_torch(value).detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def check_control(env):
    data = env.robot.data
    ids = env.ids
    backend_stiff = torch.as_tensor(
        tensor(env.robot.root_view.get_dof_stiffnesses()), device=env.device
    )[:, ids]
    backend_damp = torch.as_tensor(
        tensor(env.robot.root_view.get_dof_dampings()), device=env.device
    )[:, ids]
    # The cache alone cannot detect a second limit still active in PhysX.
    backend_limits = torch.as_tensor(
        tensor(env.robot.root_view.get_dof_max_forces()), device=env.device
    )[:, ids]
    cache_limits = data.joint_effort_limits.torch[:, ids]
    groups = {}
    for name in ("leg_motors", "wheel_motors", "passive_linkage", "guide_wheels"):
        actuator = env.robot.actuators[name]
        groups[name] = {
            "joint_ids": actuator.joint_indices.tolist(),
            "stiffness": tensor(actuator.stiffness)[0],
            "damping": tensor(actuator.damping)[0],
            "effort_limits_nm": tensor(actuator.effort_limit)[0],
        }
    coverage = set(ids) == set(
        groups["leg_motors"]["joint_ids"] + groups["wheel_motors"]["joint_ids"]
    )
    gain_zero = (
        backend_stiff.abs().max() < 1e-8
        and backend_damp.abs().max() < 1e-8
        and data.joint_stiffness.torch[:, ids].abs().max() < 1e-8
        and data.joint_damping.torch[:, ids].abs().max() < 1e-8
        and all(
            max(abs(x) for x in groups[name]["stiffness"] + groups[name]["damping"]) < 1e-8
            for name in ("leg_motors", "wheel_motors")
        )
    )
    leg_solver = 67.0 if hasattr(env, "mcu_execution") else 40.0
    required = backend_limits.new_tensor([leg_solver, leg_solver, 5.5, leg_solver, leg_solver, 5.5])
    backend_compatible = bool((backend_limits >= required - 1e-6).all())
    cache_compatible = bool((cache_limits >= required - 1e-6).all())
    actuator_limits = (
        groups["leg_motors"]["effort_limits_nm"] == [leg_solver] * 4
        and groups["wheel_motors"]["effort_limits_nm"] == [5.5] * 2
    )
    limits = actuator_limits and backend_compatible and cache_compatible
    if not backend_compatible or not cache_compatible:
        raise RuntimeError(
            "Native backend/cache effort limits would clip the active generalized torque"
        )
    if not (coverage and bool(gain_zero) and limits):
        raise RuntimeError("Native effort-only control ownership or limits differ")
    return {
        "joint_names": list(env.robot.joint_names),
        "active_joint_ids": ids,
        "active_joint_names": list(JOINT_NAMES),
        "backend_stiffness": tensor(backend_stiff),
        "backend_damping": tensor(backend_damp),
        "joint_cache_stiffness": tensor(data.joint_stiffness.torch[:, ids]),
        "joint_cache_damping": tensor(data.joint_damping.torch[:, ids]),
        "backend_effort_limits_nm": tensor(backend_limits),
        "joint_cache_effort_limits_nm": tensor(cache_limits),
        "backend_effort_limits_compatible": backend_compatible,
        "joint_cache_effort_limits_compatible": cache_compatible,
        "actuators": groups,
        "coverage": coverage,
        "active_gain_zero": bool(gain_zero),
        "active_effort_limits_match": limits,
    }
