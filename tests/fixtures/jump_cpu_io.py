"""Script physical I/O only; real task/C/V4/PPO consumers stay enabled."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from native_stand.artifacts.io import read_json
from native_stand.artifacts.runtime_paths import RuntimePaths
from native_stand.experiments.config import job_from_record, load_job_config
from native_stand.simulation.jump_shared_c import declared_input


def mapped_job(root, data_root):
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    data = read_json(data_root / "runtime_paths.json")
    paths = RuntimePaths(
        tuple(data.get("files", {}).items()),
        tuple(sorted(data.get("prefixes", {}).items(), key=lambda v: -len(v[0]))),
    )
    job = load_job_config(root / "configs/jobs/jump_unified_raw37_wheel48.json")
    record = paths.view(job.as_run_record())
    # Execution sources are supplied by this checkout, not the archive's copy.
    for name, local in (
        ("execution_jump_response", "jump_response.py"),
        ("execution_manifest", "jump_control_source/manifest.json"),
    ):
        record["assets"]["jump_bindings"][name]["path"] = str(
            root / "src/native_stand/simulation" / local
        )
    record["source_root"] = str(root / "src")
    return job_from_record(record)


def feedback(plant):
    e = plant.execution
    q = plant.data.qpos[plant.qids]
    return declared_input(
        now=e.tick,
        gyro=[0] * 3,
        gravity=[0, -1, 0],
        q=q[[0, 1, 3, 4]],
        dq=[0] * 6,
        command=plant.command,
        adopted=np.asarray(e.c.states[0].adopted),
        adopted_sequence=e.c.states[0].adopted_sequence,
        sensor_tick=e.tick,
        imu_tick=e.tick,
        feedback_ticks=[e.tick] * 6,
        accel=[0, -9.81, 0],
        driver_tau=[0] * 4,
        internal_tau=[10.0, 0.0, 10.0, 0.0],
        support_J=[0.1, 0, 0, 1] * 2,
        support_tick=e.tick,
        support_valid=3,
        leg_length=[0.28] * 2,
        leg_spread=[1.0] * 2,
        motion_entry=True,
        upstream_owner=2,
    )


def script_physics(batch, *, fault_at=None):
    for env in batch.envs:
        p = env.plant
        p.feedback = lambda p=p: feedback(p)

        def advance(request, p=p):
            e = p.execution
            if fault_at is not None and e.tick == fault_at:
                raise RuntimeError("scripted substep failure")
            q = torch.tensor(p.data.qpos[p.qids][None], dtype=torch.float32)
            dq = torch.tensor(p.data.qvel[p.vids][None], dtype=torch.float32)
            motor, net = e.step(q, dq)
            p.actual_motor = motor[0].numpy().copy()
            p.actual_NET = net[0].numpy().copy()
            p.physics_steps += 1
            p.data.time += 0.001
            p.scoring_force_valid = True
            p.solved_ground_contact[:] = True

        p.integrate = advance
    batch.reset()
    return batch


def small_job(job, *, envs=2, steps=4):
    return replace(
        job,
        model=replace(job.model, actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        runtime=replace(job.runtime, envs=envs, steps_per_env=steps),
        ppo=replace(job.ppo, epochs=1, minibatches=1),
        diagnostics=dict(
            job.diagnostics, minimum_unique_actor_samples=1, minimum_minibatch_samples=1
        ),
    )
