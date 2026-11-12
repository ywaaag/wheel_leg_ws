"""几何角加明确支撑目标偏移；输入零位与31D归一化保持原定义。"""
import math

import torch

from .motion_centered_output import HeightEpisodeOutput
from .motion_kinematic_output import KinematicMotionOutput, VERSION as GEOMETRIC_VERSION

VERSION = 'motion31_kinematic_support_centered_015_40_v1'


class SupportedKinematicMotionOutput:
    """网络0输出接近声明的支撑参考；偏移不是实测负载或额外力矩控制器。"""

    def __init__(self, declaration):
        if declaration.get('action_version') != VERSION:
            raise ValueError('需要明确的运动学加支撑偏移动作版本')
        try:
            offset = tuple(float(v) for v in declaration['support_target_offset_rad'])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('需要四腿固定支撑目标偏移，单位rad') from exc
        if (len(offset) != 4 or not all(math.isfinite(v) for v in offset) or
                abs(offset[0]+offset[2]) > 1e-12 or abs(offset[1]-offset[3]) > 1e-12):
            raise ValueError('支撑目标偏移必须有限，并保持左右几何镜像')
        geometric = dict(declaration, action_version=GEOMETRIC_VERSION,
                         support_target_offset_rad=[0., 0., 0., 0.])
        self._geometric = KinematicMotionOutput(geometric)
        self._offset = offset

    def centers(self, height_m):
        geometry = self._geometric.centers(height_m)
        return geometry + geometry.new_tensor(self._offset)

    def bind_height(self, height_m):
        center = self.centers(height_m)
        return HeightEpisodeOutput(height_m.detach().clone(), center.detach().clone())
