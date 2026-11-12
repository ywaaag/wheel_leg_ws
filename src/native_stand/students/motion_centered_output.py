"""显式Motion新输出原型：31D输入保持；本模块不注册或切换任何训练路径。"""
from dataclasses import dataclass

import torch

VERSION = 'motion31_height_centered_015_40_v1'
INPUT_VERSION = 'multiskill_mcu_v2'
LEGS = (0, 1, 3, 4)
WHEELS = (2, 5)
LEG_SCALE = .15
WHEEL_SCALE = 40.


def _finite_matrix(value, columns, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or
            value.shape[-1] != columns or not value.is_floating_point() or
            not torch.isfinite(value).all()):
        raise ValueError(name + '必须是有限浮点矩阵，末维为' + str(columns))


class CenteredMotionOutput:
    """输入为显式配置，不读文件、不修改网络或PD。拟合参考不是实体零位。"""
    def __init__(self, declaration):
        if (declaration.get('action_version') != VERSION or
                declaration.get('input_version') != INPUT_VERSION or
                declaration.get('leg_halfwidth_rad') != LEG_SCALE or
                declaration.get('wheel_halfwidth_rad_s') != WHEEL_SCALE):
            raise ValueError('需要明确的31D／腿.15rad／轮40rad/s新动作声明')
        formula = declaration.get('center_formula', {})
        try:
            values = torch.tensor([formula['reference_height_m'], formula['slope_rad_per_m'],
                                   formula['hip_at_reference_rad'], formula['active_at_reference_rad']],
                                  dtype=torch.float64)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError('高度线性公式字段错误') from exc
        if (formula.get('version') != 'common_linear_height_v1' or values.shape != (4,) or
                not torch.isfinite(values).all() or values[0] != .34 or values[1] >= 0 or
                declaration.get('height_command_range_m') != [.30, .40]):
            raise ValueError('需要参考高.34m、共同负斜率、两个参考角及.30至.40m范围')
        self._coefficients = values

    def centers(self, height_m):
        if (not isinstance(height_m, torch.Tensor) or height_m.ndim != 1 or
                not height_m.is_floating_point() or not torch.isfinite(height_m).all()):
            raise ValueError('目标高度必须是一维有限浮点张量，单位m')
        if ((height_m < .30) | (height_m > .40)).any():
            raise ValueError('目标高度超出.30至.40m，拒绝外推或静默裁剪')
        reference, slope, hip_bias, active_bias = self._coefficients.to(height_m)
        delta = slope*(height_m-reference)
        hip = hip_bias+delta
        active = active_bias+delta
        return torch.stack((hip, active, -hip, active), -1)

    def bind_height(self, height_m):
        """回合开始前绑定命令高度；不得传实时高度或在运动中重新绑定。"""
        center = self.centers(height_m)
        return HeightEpisodeOutput(height_m.detach().clone(), center.detach().clone())


@dataclass(frozen=True)
class HeightEpisodeOutput:
    height_m: torch.Tensor
    leg_center_rad: torch.Tensor

    def validate_actor_input(self, observation):
        _finite_matrix(observation, 31, 'Actor输入')
        if len(observation) != len(self.height_m):
            raise ValueError('Actor输入与绑定回合数量不同')
        if not torch.allclose(observation[:, 24]*.5, self.height_m.to(observation), atol=5e-7, rtol=0):
            raise ValueError('本回合高度命令已改变或obs24编码错误')
        expected = observation.new_tensor([1, 0, 0, 1, 0, 0])
        if not torch.equal(observation[:, 25:31], expected.expand(len(observation), -1)):
            raise ValueError('新动作仅用于Motion／NONE，不能给Recovery或Jump复用')

    def _center_scale(self, value):
        if len(value) != len(self.height_m):
            raise ValueError('动作数量与绑定回合数量不同')
        center = torch.zeros_like(value)
        center[:, LEGS] = self.leg_center_rad.to(value)
        scale = value.new_tensor([LEG_SCALE, LEG_SCALE, WHEEL_SCALE, LEG_SCALE, LEG_SCALE, WHEEL_SCALE])
        return center, scale

    def decode(self, raw_action, leg_limits_rad=None):
        _finite_matrix(raw_action, 6, 'raw action')
        center, scale = self._center_scale(raw_action)
        clipped = raw_action.clamp(-1, 1)
        requested = center+scale*clipped
        protected = requested.clone()
        mechanical_clipped = torch.zeros_like(raw_action, dtype=torch.bool)
        if leg_limits_rad is not None:
            if (leg_limits_rad.shape != (len(raw_action), 4, 2) or
                    not torch.isfinite(leg_limits_rad).all() or
                    (leg_limits_rad[..., 0] > leg_limits_rad[..., 1]).any()):
                raise ValueError('机械角度范围必须为回合数×四腿×上下限，单位rad')
            limits = leg_limits_rad.to(raw_action)
            legs = requested[:, LEGS].clamp(limits[..., 0], limits[..., 1])
            protected[:, LEGS] = legs
            mechanical_clipped[:, LEGS] = legs != requested[:, LEGS]
        return dict(raw_action=raw_action, clipped_action=clipped, requested_target=requested,
                    protected_target=protected, raw_clipped=clipped != raw_action,
                    mechanical_clipped=mechanical_clipped)

    def labels(self, teacher_physical_target):
        """越界标签保持原数值；不裁成±1后伪称准确模仿。"""
        _finite_matrix(teacher_physical_target, 6, '教师物理目标')
        center, scale = self._center_scale(teacher_physical_target)
        raw = (teacher_physical_target-center)/scale
        channel_valid = raw.abs() <= 1.+1e-6
        return dict(raw_label=raw, channel_representable=channel_valid,
                    exactly_representable=channel_valid.all(-1))

    @staticmethod
    def previous_from_adopted(adopted_target):
        """只能由实际采用层调用；允许来源控制器的目标超出新Actor包络。"""
        _finite_matrix(adopted_target, 6, '实际采用目标')
        return torch.cat((adopted_target[:, LEGS]/1.5, adopted_target[:, WHEELS]/100.), -1)

    @staticmethod
    def normalized_std(leg_std_rad=.05, wheel_std_rad_s=2.):
        values = torch.tensor([leg_std_rad, wheel_std_rad_s], dtype=torch.float64)
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError('物理探索标准差必须有限且为正')
        return torch.tensor([leg_std_rad/LEG_SCALE]*2+[wheel_std_rad_s/WHEEL_SCALE]+
                            [leg_std_rad/LEG_SCALE]*2+[wheel_std_rad_s/WHEEL_SCALE])
