"""Explicit command mutation shared by inference and diagnostic consumers."""

import torch


def set_evaluation_command(env, command):
    if env.curriculum_training or len(command) != 3:
        raise ValueError("评估命令只能用于独立推理环境")
    value = torch.as_tensor(command, device=env.device, dtype=env.target_xy.dtype)
    bounds = getattr(env, "motion_height_range_m", None)
    if (
        bounds is None
        or not bool(torch.isfinite(value).all())
        or not bounds[0] <= float(value[2]) <= bounds[1]
    ):
        raise ValueError("评估高度或命令超出实际输出合同")
    if not hasattr(env, "height_adapter"):
        raise ValueError("评估高度缺少正式高度适配器")
    height = value[2].expand(env.num_envs)
    if getattr(env, "maneuver_ready", False):
        changed = not torch.allclose(env.maneuver_goal_height, height, atol=1e-6, rtol=0)
        env.maneuver_goal_override = height.clone()
        env.maneuver_goal_height.copy_(height)
        # Preserve the closure-solved initial pose. The schedule changes height
        # before motion; .20 is a goal, never an invented .20 reset pose.
        env.height_adapter.override = None
    else:
        changed = not torch.allclose(env.height_adapter.target, height, atol=1e-6, rtol=0)
        env.height_adapter.override = height.clone()
    env.set_grouped_evaluation_targets(value[:2].expand(env.num_envs, -1))
    if changed:
        env.reset()
    return env._get_observations()
