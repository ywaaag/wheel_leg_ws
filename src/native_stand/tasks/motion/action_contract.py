"""显式输出合同的唯一物理解码器；不把执行桥单位当网络包络。"""

from dataclasses import dataclass

from .encoding import decode_targets
from .spec import SUPPORTED_LEG_SCALES, SUPPORTED_OUTPUT_VERSIONS, WIDE_OUTPUT_VERSION, CenterSpec


@dataclass(frozen=True)
class ActionDecoder:
    version: str
    center: CenterSpec | None
    leg_scale: float
    wheel_scale: float

    @classmethod
    def from_contract(cls, contract):
        version = contract["action_version"]
        if version not in SUPPORTED_OUTPUT_VERSIONS:
            raise ValueError(f"未知Motion输出合同: {version}")
        wheel = SUPPORTED_OUTPUT_VERSIONS[version]
        leg = SUPPORTED_LEG_SCALES.get(version, 0.15)
        if contract["leg_halfwidth_rad"] != leg or contract["wheel_halfwidth_rad_s"] != wheel:
            raise ValueError("动作版本与物理包络不一致")
        formula = contract["center_formula"]
        if formula["version"] != "flat_two_link_geometry_v1":
            raise ValueError("中心公式版本不支持；连续固件公式不能视为分段表")
        table = contract.get("support_offset_table", {})
        center = CenterSpec(
            **{
                k: float(formula[k])
                for k in (
                    "hip_above_root_m",
                    "wheel_radius_m",
                    "equivalent_upper_m",
                    "equivalent_lower_m",
                    "hip_zero_deg",
                    "active_zero_deg",
                )
            },
            support_offset_rad=tuple(contract.get("support_target_offset_rad", (0.0,) * 4)),
            height_range_m=tuple(contract["height_command_range_m"]),
            height_command_m=0.34,
            support_height_nodes_m=tuple(table.get("height_m", ())),
            support_offset_nodes_rad=tuple(tuple(v) for v in table.get("offset_rad", ())),
        )
        if bool(table) != (
            version
            in (
                "motion_multiheight_support_center_015_80_v1",
                "motion_multiheight020_support_center_015_80_v2",
                WIDE_OUTPUT_VERSION,
            )
        ):
            raise ValueError("输出版本与分段中心身份不一致")
        return cls(version, center, leg, wheel)

    def decode(self, raw, height, *, checks=None):
        return decode_targets(
            raw,
            self.center.centers(height.to(raw), checks=checks),
            leg_halfwidth_rad=self.leg_scale,
            wheel_halfwidth_rad_s=self.wheel_scale,
            checks=checks,
        )

    def physical_to_mean(self, target, height, *, require_representable=True):
        """物理目标转本合同mean；供蒸馏路由，越包络不静默clip成标签。"""
        import torch

        from .encoding import LEGS, WHEELS

        if target.ndim != 2 or target.shape[-1] != 6 or not torch.isfinite(target).all():
            raise ValueError("物理目标必须为有限的batch×6矩阵")
        center = torch.zeros_like(target)
        center[:, LEGS] = self.center.centers(height.to(target))
        scale = torch.full_like(target, self.leg_scale)
        scale[:, WHEELS] = self.wheel_scale
        mean = (target - center) / scale
        representable = (mean.abs() <= 1.0 + 1e-6).all(-1)
        if require_representable and not bool(representable.all()):
            raise ValueError("Teacher物理目标超出Student动作合同，不能直接作为标签")
        return {"mean": mean, "representable": representable}
