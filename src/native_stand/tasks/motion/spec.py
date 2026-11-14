"""Motion任务声明：把配置中的观测、动作、控制与奖励语义变成明确对象。

修改奖励或动作尺度时只改这里与配置，不再同时修多个准备器与校验器。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from native_stand.experiments.config import ConfigError, JobConfig

from . import encoding

SUPPORTED_OUTPUT_VERSION = "motion31_kinematic_support_centered_015_40_v1"
WIDE_OUTPUT_VERSION = "motion_multiheight020_support_center_050_80_v1"
SUPPORTED_LEG_SCALES = {WIDE_OUTPUT_VERSION: 0.5}
SUPPORTED_OUTPUT_VERSIONS = {
    WIDE_OUTPUT_VERSION: 80.0,
    SUPPORTED_OUTPUT_VERSION: 40.0,
    "motion31_kinematic_support_centered_015_80_v1": 80.0,
    "motion_multiheight_support_center_015_80_v1": 80.0,
    "motion_multiheight020_support_center_015_80_v2": 80.0,
}

CENTER_KEYS = (
    "hip_above_root_m",
    "wheel_radius_m",
    "equivalent_upper_m",
    "equivalent_lower_m",
    "hip_zero_deg",
    "active_zero_deg",
)


@dataclass(frozen=True)
class CenterSpec:
    """输出中心：平地几何加显式固定偏移或冻结分段支撑表。"""

    hip_above_root_m: float
    wheel_radius_m: float
    equivalent_upper_m: float
    equivalent_lower_m: float
    hip_zero_deg: float
    active_zero_deg: float
    support_offset_rad: tuple[float, float, float, float]
    height_range_m: tuple[float, float]
    height_command_m: float
    support_height_nodes_m: tuple[float, ...] = ()
    support_offset_nodes_rad: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        low, high = self.height_range_m
        if not low < high:
            raise ConfigError("center.height_range_m 上限必须大于下限")
        if not low <= self.height_command_m <= high:
            raise ConfigError("center.height_command_m 超出声明范围")
        offset = self.support_offset_rad
        if len(offset) != 4:
            raise ConfigError("center.support_offset_rad 必须是四腿常数")
        if abs(offset[0] + offset[2]) > 1e-12 or abs(offset[1] - offset[3]) > 1e-12:
            raise ConfigError("center.support_offset_rad 必须保持左右几何镜像")
        if self.support_height_nodes_m:
            nodes, values = self.support_height_nodes_m, self.support_offset_nodes_rad
            if len(nodes) < 2 or len(nodes) != len(values):
                raise ConfigError("支撑中心表需要至少两个同长度节点")
            if not all(math.isfinite(v) for v in nodes) or any(
                a >= b for a, b in zip(nodes, nodes[1:], strict=False)
            ):
                raise ConfigError("支撑中心高度节点必须有限且严格递增")
            if nodes[0] != low or nodes[-1] != high:
                raise ConfigError("支撑中心表必须覆盖声明高度范围，不允许外推")
            if any(len(row) != 2 or not all(math.isfinite(v) for v in row) for row in values):
                raise ConfigError("支撑中心表的每行必须为有限hip/active两项")
            if any(offset):
                raise ConfigError("分段支撑中心不能再叠固定支撑偏移")
        elif self.support_offset_nodes_rad:
            raise ConfigError("支撑中心偏移表缺少高度节点")

    def centers(self, height_m, *, checks=None):
        center = encoding.kinematic_center(
            height_m,
            hip_above_root_m=self.hip_above_root_m,
            wheel_radius_m=self.wheel_radius_m,
            equivalent_upper_m=self.equivalent_upper_m,
            equivalent_lower_m=self.equivalent_lower_m,
            hip_zero_deg=self.hip_zero_deg,
            active_zero_deg=self.active_zero_deg,
            support_offset_rad=self.support_offset_rad,
            height_range_m=self.height_range_m,
            checks=checks,
        )
        if not self.support_height_nodes_m:
            return center
        import torch

        nodes = height_m.new_tensor(self.support_height_nodes_m)
        values = height_m.new_tensor(self.support_offset_nodes_rad)
        index = torch.searchsorted(nodes, height_m.contiguous(), right=True).clamp(
            1, len(nodes) - 1
        )
        ratio = (height_m - nodes[index - 1]) / (nodes[index] - nodes[index - 1])
        offset = values[index - 1] + ratio[:, None] * (values[index] - values[index - 1])
        return center + torch.stack((offset[:, 0], offset[:, 1], -offset[:, 0], offset[:, 1]), -1)


@dataclass(frozen=True)
class MotionTask:
    """一次Motion作业的任务视图；不承担全局项目健康检查。"""

    job: JobConfig
    center: CenterSpec

    @property
    def actor_dim(self) -> int:
        return self.job.observation.actor_dim

    @property
    def critic_dim(self) -> int:
        return self.job.observation.critic_dim

    @property
    def action_dim(self) -> int:
        return self.job.action.dim

    def reward_weights(self) -> tuple[float, ...]:
        return self.job.reward.weights

    def evaluation_protocol(self) -> dict[str, Any]:
        evaluation = self.job.evaluation
        return {
            "version": evaluation.protocol_version,
            "envs": evaluation.envs,
            "scored_envs": evaluation.scored_envs,
            "seconds": evaluation.seconds,
            "seed": evaluation.seed,
            "conditions": [dict(row) for row in evaluation.conditions],
        }


def build_task(job: JobConfig) -> MotionTask:
    """校验并构造Motion任务视图。"""
    if job.task != "motion":
        raise ConfigError(f"该配置的 task={job.task}，不能作为Motion任务运行")
    if job.action.version not in SUPPORTED_OUTPUT_VERSIONS:
        raise ConfigError(
            f"action.version={job.action.version} 不是当前支撑中心输出；"
            f"需要显式声明 {SUPPORTED_OUTPUT_VERSION}"
        )
    if job.model.action_version not in (None, job.action.version):
        raise ConfigError("model.action_version 与 action.version 不一致")
    if (
        job.action.version == WIDE_OUTPUT_VERSION
        and job.algorithm == "ppo"
        and (job.backend != "isaac" or not job.assets.get("wide_actor_admission"))
    ):
        raise ConfigError("050/80 PPO requires explicit Isaac Actor admission")
    if job.action.dim != len(encoding.CHANNELS):
        raise ConfigError(f"action.dim 必须为 {len(encoding.CHANNELS)}")
    if tuple(job.action.order) != encoding.CHANNELS:
        raise ConfigError("action.order 必须保持模型left/right六通道顺序")
    if job.observation.actor_dim != encoding.ACTOR_DIM:
        raise ConfigError(f"observation.actor_dim 必须为 {encoding.ACTOR_DIM}（22帧+9任务）")
    if job.observation.leg_input_scale_rad != encoding.INPUT_LEG_SCALE_RAD:
        raise ConfigError("输入腿归一化必须保持1.5rad")
    if job.observation.wheel_input_scale_rad_s != encoding.INPUT_WHEEL_SCALE_RAD_S:
        raise ConfigError("输入轮归一化必须保持100rad/s")
    if job.action.leg_scale_rad != SUPPORTED_LEG_SCALES.get(job.action.version, 0.15):
        raise ConfigError("输出腿半宽与显式动作版本不一致")
    if job.action.wheel_scale_rad_s != SUPPORTED_OUTPUT_VERSIONS[job.action.version]:
        raise ConfigError("输出轮半宽与显式动作版本不一致")

    center_raw = dict(job.assets.get("center") or {})
    missing = [key for key in CENTER_KEYS if key not in center_raw]
    if missing:
        raise ConfigError(f"assets.center 缺少字段: {missing}")
    offset = tuple(float(v) for v in (center_raw.get("support_offset_rad") or (0.0,) * 4))
    table = center_raw.get("support_offset_table") or {}
    if bool(table) != (
        job.action.version
        in (
            "motion_multiheight_support_center_015_80_v1",
            "motion_multiheight020_support_center_015_80_v2",
            WIDE_OUTPUT_VERSION,
        )
    ):
        raise ConfigError("分段支撑中心必须绑定明确的multiheight动作版本")
    center = CenterSpec(
        hip_above_root_m=float(center_raw["hip_above_root_m"]),
        wheel_radius_m=float(center_raw["wheel_radius_m"]),
        equivalent_upper_m=float(center_raw["equivalent_upper_m"]),
        equivalent_lower_m=float(center_raw["equivalent_lower_m"]),
        hip_zero_deg=float(center_raw["hip_zero_deg"]),
        active_zero_deg=float(center_raw["active_zero_deg"]),
        support_offset_rad=offset,
        height_range_m=tuple(float(v) for v in (center_raw.get("height_range_m") or (0.30, 0.40))),
        height_command_m=float(center_raw.get("height_command_m", job.control.height_command_m)),
        support_height_nodes_m=tuple(float(v) for v in table.get("height_m", ())),
        support_offset_nodes_rad=tuple(
            tuple(float(v) for v in row) for row in table.get("offset_rad", ())
        ),
    )
    return MotionTask(job=job, center=center)


def semantics_record(task: MotionTask) -> Mapping[str, Any]:
    """写入run.json/source/的语义记录，供评估与部署核对。"""
    job = task.job
    record = {
        "task": "motion",
        "interface": job.model.interface,
        "input_frame_dim": encoding.FRAME_DIM,
        "input_actor_dim": encoding.ACTOR_DIM,
        "channels": list(encoding.CHANNELS),
        "geometric_channels": list(encoding.GEOMETRIC_CHANNELS),
        "input_scales": {
            "leg_rad": encoding.INPUT_LEG_SCALE_RAD,
            "wheel_rad_s": encoding.INPUT_WHEEL_SCALE_RAD_S,
        },
        "output_scales": {
            "leg_halfwidth_rad": job.action.leg_scale_rad,
            "wheel_halfwidth_rad_s": job.action.wheel_scale_rad_s,
        },
        "action_version": job.action.version,
        "support_center": dict(job.assets.get("center") or {}),
        "previous_semantics": "实际采用的物理目标；不是原始动作或未采用的预测",
        "periods_s": {
            "physics": job.control.physics_dt_s,
            "pd": job.control.pd_dt_s,
            "policy": job.control.policy_dt_s,
        },
        "effort_limits_nm": list(job.control.effort_limits_nm),
        "reward_version": job.reward.version,
        "reward_dt": job.reward.dt,
        "reward_weights": list(job.reward.weights),
    }
    if job.assets.get("delayed_execution") is not None:
        from .delayed_execution import validate_configuration

        record["execution_timing"] = dict(validate_configuration(job.assets["delayed_execution"]))
        record["wheel_identification_record_sha256"] = job.assets["wheel_identification_sha256"]
    return record
