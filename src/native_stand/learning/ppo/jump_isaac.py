"""The existing raw37 PPO with a distinct Isaac training/checkpoint identity."""

from native_stand.tasks.jump.isaac_contract import TRAINING

from .jump_unified import build_policy as shared_policy


def build_policy(job, observations, *, device="cpu", source=None):
    return shared_policy(job, observations, device=device, source=source, version=TRAINING)
