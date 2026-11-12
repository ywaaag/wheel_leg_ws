"""Motion几何目标中心；沿用31D输入和实际采用目标，不注册训练路径。"""
import math

import torch

from .motion_centered_output import HeightEpisodeOutput, INPUT_VERSION, LEG_SCALE, WHEEL_SCALE

VERSION = 'motion31_kinematic_centered_015_40_v1'


class KinematicMotionOutput:
    """按命令高度算平地几何角；不添加承重偏移或改编码器零位。"""

    def __init__(self, declaration):
        if (declaration.get('action_version') != VERSION or
                declaration.get('input_version') != INPUT_VERSION or
                declaration.get('leg_halfwidth_rad') != LEG_SCALE or
                declaration.get('wheel_halfwidth_rad_s') != WHEEL_SCALE or
                declaration.get('height_command_range_m') != [.30, .40]):
            raise ValueError('需要明确的运动学中心／31D／腿.15rad／轮40rad/s声明')
        formula = declaration.get('center_formula', {})
        names = ('hip_above_root_m', 'wheel_radius_m', 'equivalent_upper_m',
                 'equivalent_lower_m', 'hip_zero_deg', 'active_zero_deg')
        try:
            values = tuple(float(formula[name]) for name in names)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('运动学长度和模型零位角字段错误') from exc
        if (formula.get('version') != 'flat_two_link_geometry_v1' or
                not all(math.isfinite(v) for v in values) or
                any(v <= 0 for v in values[1:4]) or
                formula.get('clearance_m') != 0 or
                declaration.get('support_target_offset_rad') != [0., 0., 0., 0.]):
            raise ValueError('需要有限几何参数、接触间隙0和明确的零支撑偏移')
        self._geometry = values
        # 全命令范围必须有三角形实数解；不能裁剪无解高度后继续控制。
        self.centers(torch.tensor([.30, .40], dtype=torch.float64))

    def centers(self, height_m):
        if (not isinstance(height_m, torch.Tensor) or height_m.ndim != 1 or
                not height_m.is_floating_point() or not torch.isfinite(height_m).all()):
            raise ValueError('目标高度必须是一维有限浮点张量，单位m')
        if ((height_m < .30) | (height_m > .40)).any():
            raise ValueError('目标高度超出.30至.40m，拒绝外推或静默裁剪')
        hip_z, radius, upper, lower, hip_zero, active_zero = self._geometry
        length = height_m + hip_z - radius
        if ((length <= 0) | (length < abs(upper-lower)) | (length > upper+lower)).any():
            raise ValueError('该高度不满足声明平面两杆的三角形范围')
        cosine = (length.square()+upper**2-lower**2)/(2*length*upper)
        # 仅容许本浮点精度的舍入误差；几何不可达已在上面拒绝。
        tolerance = 8*torch.finfo(height_m.dtype).eps
        if ((cosine < -1-tolerance) | (cosine > 1+tolerance)).any():
            raise ValueError('运动学角度无实数解')
        beta = torch.acos(cosine.clamp(-1, 1))
        hip = beta - math.radians(hip_zero)
        active = beta - math.radians(active_zero)
        return torch.stack((hip, active, -hip, active), -1)

    def bind_height(self, height_m):
        """回合开始前算一次；h是root离地高度命令，不是实时测高。"""
        center = self.centers(height_m)
        return HeightEpisodeOutput(height_m.detach().clone(), center.detach().clone())
