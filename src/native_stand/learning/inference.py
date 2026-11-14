"""Inference-only Actor construction shared by Teachers, playback and export."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from native_stand.artifacts.checkpoint import LoadedSource, load_actor_source
from native_stand.artifacts.io import json_digest
from native_stand.learning.ppo.networks import StudentNetwork


@dataclass(frozen=True)
class InferenceSpec:
    kind: str
    input_dim: int
    hidden_dims: tuple[int, ...]
    output_dim: int
    input_contract: dict
    output_contract: dict
    activation: str = "ELU"

    def __post_init__(self):
        if self.kind not in ("tanh", "direct", "recovery_physical") or self.activation != "ELU":
            raise ValueError("Unsupported explicit Actor semantics/activation")
        declared_mean = self.output_contract.get("mean_activation")
        if declared_mean is not None:
            expected = "linear" if self.kind == "direct" else "tanh"
            if str(declared_mean).lower() != expected:
                raise ValueError("Actor mean activation contradicts output contract")
        if not self.input_contract or not self.output_contract:
            raise ValueError("Inference requires input and output contracts")
        if self.kind == "recovery_physical" and (self.input_dim, self.output_dim) != (31, 6):
            raise ValueError("Recovery physical Actor requires its declared 31/6 contract")

    def record(self):
        return asdict(self)

    @classmethod
    def from_record(cls, record):
        return cls(**{**record, "hidden_dims": tuple(record["hidden_dims"])})


class LinearActor(StudentNetwork):
    def mean(self, inputs):
        return self.mlp(inputs)


def spec_for_job(job, output_contract):
    """Version, never width or filename, determines mean semantics."""
    if job.action.version == "recovery_cubic_physical_Gaussian_v2":
        kind = "recovery_physical"
    elif job.action.version in (
        "jump_unified_raw37_wheel48_v1",
        "jump_isaac_ABI4_raw37_wheel48_v1",
    ):
        kind = "direct"
    else:
        from native_stand.tasks.motion.spec import SUPPORTED_OUTPUT_VERSIONS

        if (
            job.action.version not in SUPPORTED_OUTPUT_VERSIONS
            and job.action.version != "motion_multiheight_support_center_poly5_015_80_v1"
        ):
            raise ValueError("No declared inference semantics: " + job.action.version)
        kind = "tanh"
    return InferenceSpec(
        kind,
        job.model.actor_input,
        tuple(job.model.actor_hidden_dims),
        job.action.dim,
        dict(version=job.model.interface, observation=job.as_run_record()["observation"]),
        dict(output_contract),
    )


def build_actor(spec):
    if spec.kind == "recovery_physical":
        from native_stand.learning.ppo.recovery_cubic import PhysicalActor

        return PhysicalActor(spec.output_contract, hidden_dims=spec.hidden_dims)
    cls = LinearActor if spec.kind == "direct" else StudentNetwork
    return cls(
        actor=True,
        input_dim=spec.input_dim,
        hidden_dims=spec.hidden_dims,
        output_dim=spec.output_dim,
        init_std=torch.ones(spec.output_dim),
    )


@dataclass
class InferenceModel:
    actor: torch.nn.Module
    source: LoadedSource
    spec: InferenceSpec

    def mean(self, inputs):
        with torch.no_grad():
            return self.actor({"student_actor": inputs}, stochastic_output=False)


def from_source(source, spec):
    declared = source.payload.get("inference_spec")
    if declared is not None and json_digest(
        InferenceSpec.from_record(declared).record()
    ) != json_digest(spec.record()):
        raise ValueError("Loaded Actor inference contract differs")
    contract = source.payload.get("output_contract", source.payload.get("contract"))
    if contract is not None and json_digest(contract) != json_digest(spec.output_contract):
        raise ValueError("Loaded Actor output contract differs")
    rng = torch.get_rng_state()
    try:
        actor = build_actor(spec)
        actor.load_state_dict(source.payload["actor"], strict=True)
    finally:
        torch.set_rng_state(rng)
    return InferenceModel(actor.cpu().eval().requires_grad_(False), source, spec)


def load_model(path, spec, *, sha256):
    source = load_actor_source(path)
    if source.sha256 != sha256:
        raise ValueError("Actor source SHA differs")
    return from_source(source, spec)
