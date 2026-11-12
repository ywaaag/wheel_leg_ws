"""日志零位分析所用的平面机构几何；不提供电控零位修正或控制输出。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarLegGeometry:
    upper_m: float
    lower_m: float
    hip_zero_deg: float
    active_zero_deg: float
    wheel_lateral_m: tuple[float, float]
    hip_up_m: float
    hip_back_m: float
    wheel_radius_m: float
    wheel_halfwidth_m: float


def wheel_centers(q_rad: np.ndarray, geometry: PlanarLegGeometry) -> np.ndarray:
    """四腿角顺序ch0/1/3/4；返回几何右／左两轮的模型坐标。"""
    q = np.asarray(q_rad, dtype=float)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all():
        raise ValueError("四腿关节角必须有限，顺序为ch0/1/3/4，单位rad")
    angles = np.stack((q[..., 0], q[..., 1], -q[..., 2], q[..., 3]), -1)
    angles = angles.reshape(*q.shape[:-1], 2, 2)
    angles += np.deg2rad([geometry.hip_zero_deg, geometry.active_zero_deg])
    beta, alpha = angles[..., 0], angles[..., 1]
    # 侧视平面：第一维向前，第二维向上；等效两上杆共享投影轴心。
    hip_end = geometry.upper_m * np.stack((-np.sin(beta), -np.cos(beta)), -1)
    active_end = geometry.upper_m * np.stack((np.sin(alpha), -np.cos(alpha)), -1)
    delta = active_end - hip_end
    distance = np.linalg.norm(delta, axis=-1)
    half_chord_sq = (distance / 2) ** 2
    if (distance <= 1e-10).any() or (half_chord_sq >= geometry.lower_m**2).any():
        raise ValueError("两下杆没有唯一的声明支路交点")
    lower = np.stack((delta[..., 1], -delta[..., 0]), -1) / distance[..., None]
    wheel = (hip_end + active_end) / 2
    wheel += np.sqrt(geometry.lower_m**2 - half_chord_sq)[..., None] * lower
    lateral = np.broadcast_to(geometry.wheel_lateral_m, wheel.shape[:-1])
    return np.stack(
        (lateral, wheel[..., 1] + geometry.hip_up_m, -wheel[..., 0] + geometry.hip_back_m), -1
    )


def support_heights(q_rad, gravity, geometry: PlanarLegGeometry) -> np.ndarray:
    """各轮单独与平面相切所需root高度，不把任一轮接触当成实测。"""
    g = np.asarray(gravity, dtype=float)
    q = np.asarray(q_rad, dtype=float)
    if g.shape != (*q.shape[:-1], 3) or not np.isfinite(g).all():
        raise ValueError("每组关节角需要同帧三维重力向量")
    norm = np.linalg.norm(g, axis=-1)
    if ((norm < 0.98) | (norm > 1.02)).any():
        raise ValueError("日志重力向量长度必须接近1")
    g = g / norm[..., None]
    # 圆柱轮轴为模型X；考虑侧倾时轮宽对最低点的贡献。
    extent = geometry.wheel_radius_m * np.sqrt(np.maximum(0.0, 1 - g[..., 0] ** 2))
    extent += geometry.wheel_halfwidth_m * np.abs(g[..., 0])
    return np.sum(wheel_centers(q, geometry) * g[..., None, :], -1) + extent[..., None]


def corrected_roll_gravity(gravity, correction_rad: float) -> np.ndarray:
    """绕模型Z轴旋转重力；角度是观测坐标中的假设修正，不是电控标定值。"""
    g = np.asarray(gravity, dtype=float)
    c, s = np.cos(correction_rad), np.sin(correction_rad)
    out = g.copy()
    out[..., 0] = c * g[..., 0] - s * g[..., 1]
    out[..., 1] = s * g[..., 0] + c * g[..., 1]
    return out


def plane_jacobian(q_rad, gravity, geometry: PlanarLegGeometry) -> np.ndarray:
    """左右相切高度差对四腿角的局部导数，单位m/rad。"""
    q = np.asarray(q_rad, dtype=float)
    columns = []
    for column in range(4):
        delta = np.zeros_like(q)
        delta[..., column] = 1e-6
        plus = support_heights(q + delta, gravity, geometry)
        minus = support_heights(q - delta, gravity, geometry)
        columns.append(((plus[..., 0] - plus[..., 1]) - (minus[..., 0] - minus[..., 1])) / 2e-6)
    return np.stack(columns, -1)
