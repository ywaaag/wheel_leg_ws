"""Bounded actual-engine wiring check through the formal Jump backend.

No optimizer, no capability certification, no candidate/course promotion from
extra scored episodes. The normal public trainer remains the only PPO loop.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from native_stand.artifacts.checkpoint import load_actor_source
from native_stand.artifacts.io import file_sha256, write_json
from native_stand.experiments.config import load_job_config
from native_stand.learning.ppo.jump_unified import DirectActor
from native_stand.simulation.jump_backend import make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--envs", type=int, default=2)
    args = parser.parse_args()
    job = load_job_config(args.config)
    job = replace(job, runtime=replace(job.runtime, envs=args.envs))
    env = None
    receipt = dict(
        status="STARTED",
        scope="bounded PhysX/C/V4 wiring only",
        PPO_updates=0,
        capability_evaluation=False,
        loaded_job_sha256=file_sha256(args.config),
    )
    write_json(args.output / "receipt.json", receipt)
    try:
        env = make_env(job, output=args.output / "execution")
        source = load_actor_source(job.initialization.checkpoint)
        actor = DirectActor(job.model.actor_hidden_dims)
        actor.load_state_dict(source.payload["actor"])
        actor.to(job.runtime.device).eval()
        rows = []
        for index in range(args.steps):
            obs = env.observations()
            with torch.inference_mode():
                raw = actor.mean(obs["student_actor"])
            out = env.step(raw)
            rows.append(
                dict(
                    policy_step=index + 1,
                    reward=out["rewards"].tolist(),
                    phase=out["observations"]["jump_C_phase"].flatten().tolist(),
                    owner=out["extras"]["actual_C_owner"].tolist(),
                    done=out["dones"].tolist(),
                    actual_engine_ticks=out["extras"]["actual_physics_steps"],
                    episode_ticks=out["extras"]["episode_physics_steps"],
                )
            )
            if index == 10 and env.num_envs > 1:
                before = bytes(env.execution.c.states[1])
                env.reset([0])
                assert bytes(env.execution.c.states[1]) == before
        state = env.runtime_state()
        env.load_runtime_state(state)
        assert env.rewards[0].elapsed_s == 0
        receipt.update(
            status="PASS",
            rows=rows,
            completed=env.completed,
            actual_engine_ticks=env.scene.actual_ticks,
            actual_critic_schema=env.critics[0].schema.contract(),
            fresh_reset_restored=True,
            C_PD_calls=env.execution.C_PD_calls,
            Actor_sha256=source.sha256,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    except BaseException as exc:
        receipt.update(status="FAILED", error=repr(exc))
        raise
    finally:
        write_json(args.output / "receipt.json", receipt)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
