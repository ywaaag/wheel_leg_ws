"""World-Z clearance above the actual plane or injected triangle surface."""

from .isaac.observability import count, section

VERSION = "motion_vertical_ground_height_v2"


def vertical_height(position_world, plane_origin_world, plane_normal_world):
    return ((position_world - plane_origin_world) * plane_normal_world).sum(
        -1
    ) / plane_normal_world[..., 2]


def vertical_velocity(velocity_world, plane_normal_world):
    return (velocity_world * plane_normal_world).sum(-1) / plane_normal_world[..., 2]


def _surface_query(env, provider):
    """Reuse the same immutable position query until physics/reset advances."""
    enabled = getattr(env, "motion_surface_query_cache", False)
    clock = getattr(env, "_sim_step_counter", None)
    key = (id(provider), clock, getattr(env, "_motion_reset_epoch", 0))
    old = getattr(env, "_motion_surface_query", None)
    measurement = getattr(env, "measurement", None)
    if enabled and clock is not None and old is not None and old[0] == key:
        count(measurement, "motion.surface_cache", reason="hit")
        return old[1:]
    reason = (
        "disabled"
        if not enabled or clock is None
        else "first_use"
        if old is None
        else "surface_changed"
        if old[0][0] != key[0]
        else "reset_epoch_changed"
        if old[0][2] != key[2]
        else "physics_advanced"
    )
    count(measurement, "motion.surface_cache", reason=reason)
    xy = env.robot.data.root_pos_w.torch[..., :2]
    count(measurement, "motion.surface_query_rows", len(xy))
    with section(
        measurement,
        "motion.surface_query",
        parent=getattr(env, "_measurement_reset_parent", "rollout_seconds"),
    ):
        valid = provider.contains(xy)
        z, gradient = provider.sample(xy, check=False)
    if enabled and clock is not None:
        env._motion_surface_query = (key, valid, z, gradient)
    return valid, z, gradient


def from_env(env):
    provider = getattr(env, "motion_surface", None)
    if provider is None:
        return vertical_height(
            env.robot.data.root_pos_w.torch, env.scene.env_origins, env.DR_static_ground_normal
        )
    position = env.robot.data.root_pos_w.torch
    valid, z, _ = _surface_query(env, provider)
    env.motion_terrain_height_valid = valid
    if hasattr(env, "motion_road_exit"):
        env.motion_road_exit |= ~valid
    return __import__("torch").where(valid, position[..., 2] - z, 0.0)


def velocity_from_env(env):
    provider = getattr(env, "motion_surface", None)
    if provider is None:
        return vertical_velocity(env.robot.data.root_lin_vel_w.torch, env.DR_static_ground_normal)
    valid, _, gradient = _surface_query(env, provider)
    v = env.robot.data.root_lin_vel_w.torch
    return __import__("torch").where(valid, v[..., 2] - (gradient * v[..., :2]).sum(-1), 0.0)


def contract():
    return dict(
        version=VERSION,
        units="m",
        point="root-link",
        positive="world+Z",
        ground="injected actual triangle collision surface, otherwise assigned static plane",
        velocity="vz-dzdx*vx-dzdy*vy",
        actor_terrain_oracle=False,
    )
