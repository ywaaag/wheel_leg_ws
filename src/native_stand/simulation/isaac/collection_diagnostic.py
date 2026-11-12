"""Strict model-bound Motion diagnostics; unsupported interventions are rejected."""

import torch

from native_stand.artifacts.io import file_sha256, json_digest, read_json
from native_stand.learning.inference import spec_for_job


def configure(session, protocol):
    if (
        session.cumulative_physics_steps
        or session.num_envs != 1
        or session.job.model.critic_input not in (877, 881)
    ):
        raise ValueError("Motion diagnostic requires a fresh single-environment session")
    e, job = session.env, session.job
    if session.output_contract is None or session.template is None:
        raise ValueError("Motion diagnostic has no loaded source/physical binding")
    spec = spec_for_job(job, session.output_contract)
    for key, actual in (
        ("input_contract", spec.input_contract),
        ("output_contract", spec.output_contract),
        ("initial_state", {"kind": "model_bound_reset", "seed": job.runtime.seed}),
    ):
        if json_digest(protocol[key]) != json_digest(actual):
            raise ValueError("Motion diagnostic condition unsupported/mismatched: " + key)
    ref = session.template["motion_identified_response"]
    actual_v4 = e.motion_timing.contract
    if (
        protocol["V4"].get("sha256") != ref["contract_sha256"]
        or file_sha256(ref["contract"]) != ref["contract_sha256"]
        or json_digest(actual_v4) != json_digest(read_json(ref["contract"]))
    ):
        raise ValueError("Motion diagnostic actual complete V4 differs")
    scene = job.assets.get("terrain_scene", session.template.get("ground"))
    if scene is None or json_digest(protocol["scene"]) != json_digest(scene):
        raise ValueError("Motion diagnostic scene differs; no friction/terrain fallback")
    if job.assets.get("terrain_scene") and json_digest(e.motion_terrain_config) != json_digest(
        scene
    ):
        raise ValueError("Motion diagnostic declared terrain is not the actual surface")
    e.curriculum_training = False
    raw = e.set_evaluation_command(protocol["command"])
    if raw is None:
        raise ValueError("Motion command owner did not return its actual observations")
    session._obs = session._convert(e, raw)
    # Goal changes can reset/re-sample DR. Read back AFTER that boundary.
    dr, ring = e.motion_timing.dr, e.motion_timing.base.motor_ring
    timing = dict(
        physics_ms=e.physics_dt * 1000,
        PD_ms=job.control.pd_dt_s * 1000,
        policy_ms=e.step_dt * 1000,
        actor_packet_ms=int(dr.packet_age[0]),
        actor_leg_packet_ms=int(dr.leg_packet_age[0]),
        target_ms=int(dr.target_delay[0]),
        wheel_transport_ms=ring.delay[0, [2, 5]].cpu().tolist(),
        wheel_tau_ms=ring.tau[0, [2, 5]].cpu().tolist(),
    )
    if timing != protocol["timing"]:
        raise ValueError("Motion diagnostic actual timing differs; no silent delay override")
    caps = dr.caps[0].cpu().tolist()
    requested_caps = torch.as_tensor(protocol["motor_caps"]).to(dr.caps)
    if requested_caps.shape != (6,) or not torch.allclose(
        requested_caps, dr.caps[0], atol=1e-6, rtol=0
    ):
        raise ValueError("Motion diagnostic actual motor caps differ")
    session._diagnostic = True
    e.collection_snapshot = session.capture_diagnostic
    session.capture_diagnostic()
    return dict(
        input_contract=spec.input_contract,
        output_contract=spec.output_contract,
        scene=scene,
        command=list(protocol["command"]),
        initial_state={"kind": "model_bound_reset", "seed": job.runtime.seed},
        timing=dict(protocol["timing"]),
        V4=dict(protocol["V4"]),
        motor_caps=list(protocol["motor_caps"]),
        engine="Isaac/PhysX",
        integration={"physics_dt_s": e.physics_dt, "policy_dt_s": e.step_dt},
        contact_sampling={"saved_period_s": e.step_dt, "force_threshold_N": 2.0},
        actual_caps=caps,
        actual_timing=timing,
        actual_DR={"profile": int(dr.profiles[0]), "amplitude": float(dr.amplitude[0, 0])},
        command_semantics="height first; declared production ramp; actual command saved per step",
        physics_state_restored=False,
    )
