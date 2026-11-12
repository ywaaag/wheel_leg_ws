"""支撑中心动作的物理奖励与读回；纯张量函数不读取运行文件。"""

from __future__ import annotations

import math

import torch

from .encoding import LEGS, WHEELS

VERSION = "motion_supported_physical_scale_corrections_v1"


def configuration():
    return {
        "version": VERSION,
        "linear_wide_mps": 0.8,
        "linear_narrow_mps": 0.10,
        "linear_weight": 4.0,
        "linear_tail_scale_mps": 0.30,
        "linear_tail_weight": -1.0,
        "yaw_wide_rad_s": 0.50,
        "yaw_narrow_rad_s": 0.15,
        "yaw_weight": 2.0,
        "yaw_tail_scale_rad_s": 0.50,
        "yaw_tail_weight": -0.5,
        "narrow_mix": 0.25,
        "effort_normalization": "per_env_effective_motor_caps",
        "motor_effort_excludes": ["NET", "passive_friction", "external_pulse"],
        "input_scales_unchanged": [1.5, 100.0],
        "output_scales": [0.15, 40.0],
        "policy_dt_s": 0.01,
    }


def validate_configuration(value, *, wheel_scale_rad_s=None):
    expected = configuration()
    wheel = (
        value.get("output_scales", [None, None])[1]
        if wheel_scale_rad_s is None
        else wheel_scale_rad_s
    )
    if wheel not in (40.0, 80.0):
        raise ValueError("输出轮尺度必须是显式40或80")
    expected["output_scales"] = [0.15, wheel]
    if dict(value) != expected:
        raise ValueError("支撑中心物理尺度修正声明不一致")
    return value


def huber(value):
    absolute = value.abs()
    return torch.where(absolute <= 1, 0.5 * value.square(), absolute - 0.5)


def tracking_terms(error, *, yaw=False, config=None):
    c = configuration() if config is None else config
    prefix = "yaw" if yaw else "linear"
    unit = "rad_s" if yaw else "mps"
    wide, narrow = c[f"{prefix}_wide_{unit}"], c[f"{prefix}_narrow_{unit}"]
    mix = c["narrow_mix"]
    tracking = c[f"{prefix}_weight"] * (
        (1 - mix) / (1 + (error / wide).square()) + mix * torch.exp(-(error / narrow).square())
    )
    tail = c[f"{prefix}_tail_weight"] * huber(error / c[f"{prefix}_tail_scale_{unit}"])
    return tracking, tail


def clip_probability(mean, std):
    """未截断高斯落到[-1,1]外的概率；不抽样，不改变分布缓存。"""
    denominator = std * math.sqrt(2)
    return 0.5 * torch.erfc((1 - mean) / denominator) + 0.5 * torch.erfc((1 + mean) / denominator)


def boundary_loss(mean, std, *, allowed_probability=0.1, coefficient=0.01):
    probability = clip_probability(mean[..., LEGS], std[..., LEGS])
    excess = torch.relu((probability - allowed_probability) / (0.5 - allowed_probability))
    return coefficient * huber(excess).mean(), probability


def saturation(request, driver, motor, caps, *, tolerance_nm=1e-4):
    """区分请求限幅、驱动边界、实际电机能力占用；caps须为生效值。"""
    if request.shape != caps.shape or driver.shape != caps.shape or motor.shape != caps.shape:
        raise ValueError("力矩与每环境六通道caps必须同形")
    return {
        "request_over_cap": request.abs() > caps + tolerance_nm,
        "driver_at_cap": driver.abs() >= caps - tolerance_nm,
        "motor_at_cap": motor.abs() >= caps - tolerance_nm,
        "motor_capacity_fraction": motor.abs() / caps,
    }


def wheel_target_from_trace(values, *, encoding):
    """必须声明字段语义，不能从新spec猜旧桥字段的尺度。"""
    if encoding == "physical_target":
        return values[..., WHEELS]
    if encoding == "legacy_bridge_1p5_100":
        return values[..., WHEELS] * 100
    if encoding == "supported_raw_015_40":
        return values[..., WHEELS].clamp(-1, 1) * 40
    raise ValueError("必须明确物理目标、旧桥或新raw编码")


def replace_density(result, name, density):
    index = result["fields"].index(name)
    value = density * result["eligible"]
    old = result["dt"][:, index].clone()
    result["raw"][:, index] = value
    result["density"][:, index] = value
    result["dt"][:, index] = value * 0.01
    result["total"] += result["dt"][:, index] - old


def apply_runtime(env, previous_motor, history_valid, config):
    """在原物理reward组装后替换具名列，不重复叠加同一成本。"""
    result = env.student_reward_result
    if result.get("supported_scale_corrections"):
        raise ValueError("同一transition重复修正奖励")
    timing = env.motion_timing
    base = timing.base
    caps = timing.dr.caps
    motor = base.motor_actual
    effort = (motor / caps).square().mean(-1)
    replace_density(result, "effort", -0.02 * effort)
    replace_density(result, "transfer_effort", -0.08 * effort)
    delta = (motor[:, WHEELS] - previous_motor[:, WHEELS]) / caps[:, WHEELS]
    replace_density(result, "wheel_torque_change", -0.10 * huber(delta).mean(-1) * history_valid)
    excess = torch.relu(env.effort_pre[:, WHEELS].abs() / caps[:, WHEELS] - 0.8)
    mix = timing.config["smooth_mix"]
    request = ((1 - mix) * huber(excess / 2) + mix * huber(excess)).mean(-1)
    replace_density(result, "wheel_request_excess", -0.25 * request)
    for yaw, axis, state_column in ((False, 0, 7), (True, 1, 9)):
        active = env.target_xy[:, axis] != 0
        track, tail = tracking_terms(
            env.recovery_snapshot[:, state_column] - env.motion_action_command[:, axis],
            yaw=yaw,
            config=config,
        )
        for name, value in (
            ("yaw_tracking" if yaw else "linear_tracking", track),
            ("yaw_error" if yaw else "linear_error", tail),
        ):
            old_density = result["density"][:, result["fields"].index(name)]
            replace_density(result, name, torch.where(active, value, old_density))
    diagnostic = saturation(env.effort_pre, base.driver, motor, caps)
    env.recovery_snapshot[:, 17] = diagnostic["driver_at_cap"].any(-1).float()
    env.supported_saturation_post = diagnostic
    env.supported_requested_target_post = base.requested.clone()
    env.supported_adopted_target_post = base.adopted.clone()
    result["supported_scale_corrections"] = VERSION


def corrected_environment_class(base_class, config):
    """只生成新作业实例的子类；不修改冻结类或源码。"""
    validate_configuration(config)

    class CorrectedMotionEnv(base_class):
        def _get_dones(self):
            previous = self.fullchain_prev_motor.clone()
            valid = self.fullchain_history_valid.float().clone()
            result = super()._get_dones()
            apply_runtime(self, previous, valid, config)
            return result

    return CorrectedMotionEnv
