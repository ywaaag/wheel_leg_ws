"""CPU-only exact-source/relocation check. Never creates a run or steps PPO."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def check(config):
    import torch
    from tensordict import TensorDict

    from native_stand.artifacts.branch_initialization import require_allocated
    from native_stand.artifacts.io import file_sha256, read_json
    from native_stand.artifacts.runtime_paths import RuntimePaths, execution_job
    from native_stand.experiments.config import load_job_config
    from native_stand.learning.ppo.dual_adam import DualAdam
    from native_stand.learning.ppo.source_anchor import KEY
    from native_stand.learning.ppo.supported import assemble_from_full
    from native_stand.simulation.isaac.legacy_support import activate_view
    from native_stand.simulation.isaac.motion_backend import _resolve_source, validate_job
    from native_stand.tasks.motion.action_contract import ActionDecoder
    from native_stand.tasks.motion.course_strategy import resolve

    if torch.cuda.is_initialized():
        raise RuntimeError("Readiness check must run in a fresh CPU-only process")
    job = execution_job(load_job_config(config))
    validate_job(job)
    grant = require_allocated(job)
    source = _resolve_source(job, None)
    payload = source.payload
    paths = RuntimePaths.from_assets(job.assets)
    for key in ("source_policy_retention", "source_anchor_identity"):
        if key in payload:
            payload[key] = paths.view(payload[key])
    resolve(job.curriculum.version).prepare(payload, job.runtime.envs)
    activate_view(job.assets["frozen_source_root"], job.assets["frozen_bundle"])
    import fixed_recovery
    import privileged_dr
    import runtime_support

    assert fixed_recovery.DIM == 863
    assert privileged_dr.DIM == 863
    assert sum(width for _, width in fixed_recovery.EXTRA_FIELDS) == 129
    assert callable(runtime_support.DR.reset)
    # Synthetic CPU rows verify full-state assembly, not physical capability,
    # rollout throughput or 4096-environment GPU memory.
    observations = TensorDict(
        dict(
            student_actor=torch.zeros(2, 31),
            student_critic=torch.zeros(2, job.model.critic_input),
            recovery_actor_valid=torch.ones(2, 1, dtype=torch.bool),
            **{KEY: torch.full((2, 1), -1, dtype=torch.long)},
        ),
        batch_size=[2],
    )
    algorithm = assemble_from_full(payload, observations, job)
    assert isinstance(algorithm.optimizer, DualAdam)
    assert payload["warmup_updates_done"] >= job.ppo.critic_warmup_updates
    assert algorithm.actor_update_enabled
    for name in ("actor", "critic"):
        actual = getattr(algorithm, name).state_dict()
        assert all(torch.equal(actual[k], v.cpu()) for k, v in payload[name].items())
    anchor = algorithm.source_anchor
    assert all(not p.requires_grad for p in anchor.teacher.parameters())
    if payload.get("source_anchor_runtime_state") is not None:
        assert anchor.state_dict() == payload["source_anchor_runtime_state"]
    decoder = ActionDecoder.from_contract(payload["output_contract"])
    with torch.no_grad():
        mean = algorithm.actor.mean(observations["student_actor"])
        target = decoder.decode(mean, torch.tensor([0.20, 0.40]))["requested_target"]
    assert target.shape == (2, 6) and bool(torch.isfinite(target).all())
    framework_files = 0
    framework = job.assets.get("framework_dependency")
    if framework:
        ref = framework["manifest"]
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("SDK manifest identity differs")
        root = Path(framework["root"])
        files = read_json(ref["path"])["files"]
        for row in files:
            if file_sha256(root / row["relative_path"]) != row["sha256"]:
                raise ValueError("SDK source identity differs: " + row["relative_path"])
        framework_files = len(files)
    assert not torch.cuda.is_initialized()
    n, steps = job.runtime.envs, job.runtime.steps_per_env
    result = dict(
        status="CPU_READY_GPU_UNVERIFIED",
        config_path=str(Path(config).resolve()),
        config_sha256=file_sha256(config),
        source=dict(path=str(source.path), sha256=source.sha256),
        source_model_updates=payload["model_online_updates"],
        historical_actual_updates=job.assets["historical_online_updates"],
        new_updates_consumed=0,
        new_authorized_updates=grant["formal_updates"],
        synthetic_CPU_rows=2,
        restored_Actor_Critic_two_Adam=True,
        warmup_updates_already_done=payload["warmup_updates_done"],
        warmup_repeated=False,
        frozen_KL_Teacher=True,
        source_bank_cursor_restored=True,
        std_physical=(
            algorithm.actor.log_std.exp() * mean.new_tensor([0.15, 0.15, 80, 0.15, 0.15, 80])
        ).tolist(),
        course=payload["motion_course_state"]["version"],
        action=payload["action_version"],
        actor_observation_dim=31,
        critic_observation_dim=job.model.critic_input,
        numerical_DR_ABI_path=str(Path(fixed_recovery.__file__).resolve()),
        SDK_source_files_verified=framework_files,
        runtime_versions={
            k: importlib.metadata.version(k) for k in ("torch", "rsl-rl-lib", "tensordict")
        },
        GPU_plan=dict(
            requested_envs=n,
            steps_per_env=steps,
            rollout_observation_min_bytes=4 * n * steps * (31 + job.model.critic_input),
            one_minibatch_forward_activation_min_bytes=4
            * n
            * steps
            // job.ppo.minibatches
            * (sum(job.model.actor_hidden_dims) + sum(job.model.critic_hidden_dims)),
            total_GPU_peak_bytes=None,
            throughput=None,
            simultaneous_training_validated=False,
        ),
        physical_steps=0,
        PPO_updates=0,
        optimizer_steps=0,
        creates_run=False,
        reserves_budget=False,
        starts_engine=False,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.config)
    if args.output:
        from native_stand.artifacts.io import write_json

        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
