"""Finite landing-first task bookkeeping; independent of simulator and files."""

from dataclasses import asdict, dataclass, field

from .reward import RewardMemory

VERSION = "jump_landing_first_finite_reward_v2"


@dataclass
class TaskMemory:
    initial_kind: str
    horizon_s: float
    elapsed_s: float = 0.0
    had_support: bool = False
    actual_takeoff: bool = False
    previous_contact: bool = False
    stable_s: float = 0.0
    landing_paid: bool = False
    reward: RewardMemory = field(default_factory=RewardMemory)

    def __post_init__(self):
        if self.initial_kind not in ("ground_active_jump", "landing_only") or self.horizon_s <= 0:
            raise ValueError("explicit initial provenance and finite horizon required")
        self.reward.initial_kind = self.initial_kind

    def state_dict(self):
        return asdict(self)

    @classmethod
    def restore(cls, value):
        data = dict(value)
        data["reward"] = RewardMemory(**data["reward"])
        return cls(**data)
