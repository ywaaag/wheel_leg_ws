"""Reuse dense Jump rewards with explicit BALANCE and own completion semantics."""

from dataclasses import dataclass, replace

import numpy as np

from native_stand.tasks.jump.dense_reward import DenseReward, Packet, RewardConfig, TaskSpec
from native_stand.tasks.jump.phases import Phase
from native_stand.tasks.jump.unified import BALANCE

VERSION = "jump_unified_dense_reward_wheel48_v2"


@dataclass(frozen=True)
class UnifiedPacket(Packet):
    valid_phases = tuple(range(10))


class UnifiedReward(DenseReward):
    def step(self, packet):
        if not isinstance(packet, UnifiedPacket):
            raise ValueError("explicit ABI4 packet required")
        outcome = super().step(packet)
        if packet.owner == 2 and packet.phase == Phase.SETTLE:
            # Recover height even after an accidental fall without a proven
            # takeoff. Keep all positive landing/completion provenance gates.
            h = (packet.root_height_m - self.spec.entry_height_m) / 0.10
            outcome["terms"]["settle_height"] = (
                -self.config.w_settle_height * min(h * h, 4.0) * packet.dt_s
            )
        if packet.owner == 2 and packet.phase == BALANCE:
            dt = packet.dt_s
            self.jump_control_s += dt
            # Stand-height is enabled only in BALANCE. It cannot oppose crouch,
            # extension or impact. Velocity inputs are command-relative below.
            h = (packet.root_height_m - self.spec.entry_height_m) / 0.05
            tilt = sum((v / 0.35) ** 2 for v in packet.tilt_rad)
            vel = np.linalg.norm(packet.COM_velocity_m_s[:2]) / 0.3
            outcome["terms"]["balance_height"] = -0.5 * min(h * h, 4.0) * dt
            outcome["terms"]["balance_velocity"] = -0.5 * min(vel * vel, 4.0) * dt
            outcome["terms"]["balance_posture"] = -min(tilt, 4.0) * dt
            outcome["terms"]["time"] -= self.config.w_time * dt
        outcome["terms"]["own_completion"] = outcome["terms"].pop("handover")
        outcome["total"] = float(sum(outcome["terms"].values()))
        outcome["jump_control_s"] = self.jump_control_s
        return outcome

    def state_dict(self):
        result = super().state_dict()
        from .outcome import REWARD
        from .skill_course import REWARD as SKILL_REWARD
        from .skill_course import SkillTaskSpec

        result["version"] = (
            SKILL_REWARD
            if isinstance(self.spec, SkillTaskSpec)
            else REWARD
            if not self.config.require_crouch
            else VERSION
        )
        return result

    @classmethod
    def restore(cls, value):
        data = dict(value)
        from .outcome import REWARD
        from .skill_course import REWARD as SKILL_REWARD
        from .skill_course import SkillTaskSpec

        skill = "objective" in data["spec"]
        expected = (
            SKILL_REWARD
            if skill
            else VERSION
            if data["config"].get("require_crouch", True)
            else REWARD
        )
        if data.pop("version") != expected:
            raise ValueError("unified reward version mismatch")
        data["spec"] = (SkillTaskSpec if skill else TaskSpec)(**data["spec"])
        data["config"] = RewardConfig(**data["config"])
        data["previous"] = UnifiedPacket(**data["previous"])
        return cls(**data)

    def critic_fields(self):
        # All old progress/goal memory stays named. Only phase one-hot grows.
        previous = self.previous
        if previous.phase == BALANCE:
            self.previous = replace(previous, phase=0)
        try:
            fields = super().critic_fields()
        finally:
            self.previous = previous
        name = "dense.previous.phase"
        fields[name] = (
            np.eye(10)[previous.phase],
            1.0,
            "actual C phase one-hot including BALANCE9",
        )
        return fields


def command_relative(packet, world_xy_command, *, own_completed=False):
    if not isinstance(packet, UnifiedPacket):
        raise ValueError("unified packet required")
    command = np.asarray(world_xy_command, float)
    if command.shape != (2,) or not np.isfinite(command).all():
        raise ValueError("explicit finite horizontal command required")
    relative = np.asarray(packet.COM_velocity_m_s).copy()
    relative[:2] -= command
    # The inherited field name is a serialization detail. ABI4 supplies only
    # its OWN C completion, gated again by true takeoff/apex/landing reward state.
    return replace(packet, COM_velocity_m_s=tuple(relative), motion_confirmed=bool(own_completed))
