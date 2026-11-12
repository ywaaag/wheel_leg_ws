"""Motion 31D输入编码与6D动作解码的唯一纯函数实现。

两组尺度必须分开：
- 输入（观测与previous）: 腿 1.5 rad、轮 100 rad/s，保持不变；
- 输出（动作包络）: 腿 ±0.15 rad、轮 ±40 rad/s，中心由命令高度几何加固定支撑偏移给出。

本模块不读文件、不建环境、不改PD。
"""

from __future__ import annotations

import math

import torch

INPUT_VERSION = "multiskill_mcu_v2"
FRAME_DIM = 22
TASK_DIM = 9
ACTOR_DIM = FRAME_DIM + TASK_DIM

# 模型数组顺序：模型left在前，对应几何右。
CHANNELS = (
    "left_hip_pitch",
    "left_active_link_pitch",
    "left_wheel_spin",
    "right_hip_pitch",
    "right_active_link_pitch",
    "right_wheel_spin",
)
GEOMETRIC_CHANNELS = (
    "right_hip",
    "right_active",
    "right_wheel",
    "left_hip",
    "left_active",
    "left_wheel",
)
LEGS = (0, 1, 3, 4)
WHEELS = (2, 5)

INPUT_LEG_SCALE_RAD = 1.5
INPUT_WHEEL_SCALE_RAD_S = 100.0
OUTPUT_LEG_HALFWIDTH_RAD = 0.15
OUTPUT_WHEEL_HALFWIDTH_RAD_S = 40.0

SKILLS = ("Motion", "Recovery", "Jump")
JUMPS = ("NONE", "SMALL", "BIG")


def _require(condition, message, checks=None):
    """Keep strict public validation; a control boundary may batch tensor checks."""
    if checks is None:
        if not bool(condition.all()):
            raise ValueError(message)
    else:
        checks.require(condition, message)


def encode_frame(
    angular_body: torch.Tensor,
    gravity_body: torch.Tensor,
    q: torch.Tensor,
    dq: torch.Tensor,
    previous_legs: torch.Tensor,
    previous_wheels: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """编码一帧22维观测。previous是实际采用的物理目标，不是原始动作。"""
    if q.shape[-1] != 6 or dq.shape != q.shape or reference.shape != q.shape:
        raise ValueError("需要左右共6通道的q/dq/reference")
    if previous_legs.shape[-1] != 4 or previous_wheels.shape[-1] != 2:
        raise ValueError("previous四腿两轮布局不符")
    if gravity_body.shape[-1] != 3 or angular_body.shape[-1] != 3:
        raise ValueError("角速度与重力向量必须是3维")
    frame = torch.cat(
        (
            angular_body * 0.25,
            gravity_body,
            (q[..., LEGS] - reference[..., LEGS]) / INPUT_LEG_SCALE_RAD,
            dq * 0.05,
            (previous_legs - reference[..., LEGS]) / INPUT_LEG_SCALE_RAD,
            previous_wheels / INPUT_WHEEL_SCALE_RAD_S,
        ),
        -1,
    )
    if frame.shape[-1] != FRAME_DIM:
        raise ValueError(f"帧维数应为{FRAME_DIM}，实际{frame.shape[-1]}")
    if not torch.isfinite(frame).all():
        raise ValueError("观测出现非有限值")
    return frame


def encode_task(
    command: torch.Tensor, *, skill: str = "Motion", jump: str = "NONE"
) -> torch.Tensor:
    """编码9维任务向量：三通道命令加技能与跳跃one-hot。"""
    if skill not in SKILLS or jump not in JUMPS or command.shape[-1] != 3:
        raise ValueError("需要明确技能、跳跃档位与三通道命令")
    one = command.new_zeros((*command.shape[:-1], 6))
    one[..., SKILLS.index(skill)] = 1
    one[..., 3 + JUMPS.index(jump)] = 1
    return torch.cat((command / command.new_tensor([4.0, 4.0, 0.5]), one), -1)


def encode_actor_input(frame: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
    """拼装31D Actor输入：22维当前帧加9维任务。"""
    if frame.shape[-1] != FRAME_DIM or task.shape[-1] != TASK_DIM:
        raise ValueError("31D输入布局不符")
    out = torch.cat((frame, task), -1)
    if not torch.isfinite(out).all():
        raise ValueError("Actor输入出现非有限值")
    return out


def kinematic_center(
    height_m: torch.Tensor,
    *,
    hip_above_root_m: float,
    wheel_radius_m: float,
    equivalent_upper_m: float,
    equivalent_lower_m: float,
    hip_zero_deg: float,
    active_zero_deg: float,
    support_offset_rad: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    height_range_m: tuple[float, float] = (0.30, 0.40),
    checks=None,
) -> torch.Tensor:
    """按命令高度算平地两连杆几何角，再加固定支撑偏移。

    偏移是输出中心偏移，不是编码器标零、实测弹簧矩或额外力矩前馈。
    """
    if height_m.ndim != 1 or not height_m.is_floating_point():
        raise ValueError("目标高度必须是一维浮点张量")
    _require(torch.isfinite(height_m), "目标高度出现非有限值", checks)
    low, high = height_range_m
    _require(
        (height_m >= low) & (height_m <= high),
        f"目标高度超出{low}至{high}m；拒绝外推或静默裁剪",
        checks,
    )
    length = height_m + hip_above_root_m - wheel_radius_m
    _require(
        (length > 0)
        & (length >= abs(equivalent_upper_m - equivalent_lower_m))
        & (length <= equivalent_upper_m + equivalent_lower_m),
        "该高度不满足声明的平面两连杆三角形范围",
        checks,
    )
    cosine = (length.square() + equivalent_upper_m**2 - equivalent_lower_m**2) / (
        2 * length * equivalent_upper_m
    )
    tolerance = 8 * torch.finfo(height_m.dtype).eps
    _require((cosine >= -1 - tolerance) & (cosine <= 1 + tolerance), "运动学角度无实数解", checks)
    beta = torch.acos(cosine.clamp(-1.0, 1.0))
    hip = beta - math.radians(hip_zero_deg)
    active = beta - math.radians(active_zero_deg)
    center = torch.stack((hip, active, -hip, active), -1)
    offset = torch.tensor(support_offset_rad, dtype=center.dtype, device=center.device)
    return center + offset


def decode_targets(
    raw_action: torch.Tensor,
    leg_center_rad: torch.Tensor,
    *,
    leg_limits_rad: torch.Tensor | None = None,
    leg_halfwidth_rad: float = OUTPUT_LEG_HALFWIDTH_RAD,
    wheel_halfwidth_rad_s: float = OUTPUT_WHEEL_HALFWIDTH_RAD_S,
    checks=None,
) -> dict[str, torch.Tensor]:
    """把网络原始输出解码为实际采用的四腿两轮物理目标。"""
    if raw_action.ndim != 2 or raw_action.shape[-1] != 6:
        raise ValueError("需要回合数×6的原始动作")
    if not raw_action.is_floating_point():
        raise ValueError("原始动作必须是有限浮点矩阵")
    _require(torch.isfinite(raw_action), "原始动作必须是有限浮点矩阵", checks)
    if leg_center_rad.shape != (raw_action.shape[0], 4):
        raise ValueError("支撑中心必须是回合数×四腿")
    center = torch.zeros_like(raw_action)
    center[:, LEGS] = leg_center_rad.to(raw_action)
    scale = raw_action.new_tensor(
        [
            leg_halfwidth_rad,
            leg_halfwidth_rad,
            wheel_halfwidth_rad_s,
            leg_halfwidth_rad,
            leg_halfwidth_rad,
            wheel_halfwidth_rad_s,
        ]
    )
    clipped = raw_action.clamp(-1.0, 1.0)
    requested = center + scale * clipped
    protected = requested.clone()
    mechanical_clipped = torch.zeros_like(raw_action, dtype=torch.bool)
    if leg_limits_rad is not None:
        if leg_limits_rad.shape != (raw_action.shape[0], 4, 2):
            raise ValueError("机械角度范围必须为回合数×四腿×上下限")
        limits = leg_limits_rad.to(raw_action)
        _require(limits[..., 0] <= limits[..., 1], "机械角度范围不合法", checks)
        _require(torch.isfinite(limits), "机械角度范围不合法", checks)
        legs = requested[:, LEGS].clamp(limits[..., 0], limits[..., 1])
        protected[:, LEGS] = legs
        mechanical_clipped[:, LEGS] = legs != requested[:, LEGS]
    return {
        "raw_action": raw_action,
        "clipped_action": clipped,
        "requested_target": requested,
        "protected_target": protected,
        "raw_clipped": clipped != raw_action,
        "mechanical_clipped": mechanical_clipped,
    }


def previous_from_adopted(adopted_target: torch.Tensor) -> torch.Tensor:
    """只由实际执行层调用：把采用目标换算为previous观测。"""
    if adopted_target.ndim != 2 or adopted_target.shape[-1] != 6:
        raise ValueError("实际采用目标必须是回合数×6")
    if not torch.isfinite(adopted_target).all():
        raise ValueError("实际采用目标出现非有限值")
    return torch.cat(
        (
            adopted_target[:, LEGS] / INPUT_LEG_SCALE_RAD,
            adopted_target[:, WHEELS] / INPUT_WHEEL_SCALE_RAD_S,
        ),
        -1,
    )


def normalized_std(leg_std_rad: float = 0.05, wheel_std_rad_s: float = 2.0) -> torch.Tensor:
    """把物理探索标准差换算为归一化动作空间标准差。"""
    if leg_std_rad <= 0 or wheel_std_rad_s <= 0:
        raise ValueError("物理探索标准差必须为正")
    leg = leg_std_rad / OUTPUT_LEG_HALFWIDTH_RAD
    wheel = wheel_std_rad_s / OUTPUT_WHEEL_HALFWIDTH_RAD_S
    return torch.tensor([leg, leg, wheel, leg, leg, wheel])
