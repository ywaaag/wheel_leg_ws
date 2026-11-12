"""CPU component comparison against an isolated checkout; NEVER integrates physics.

Run with PYTHONPATH=<checkout>/src:<delivery>/tests. Script I/O exercises the
real C/FSM, V4, packets, observations and resets, without a training loop.
"""

import argparse
import ctypes as C
import hashlib
import json
import random
import statistics
import subprocess
import tempfile
import time
import tracemalloc
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from fixtures.jump_cpu_io import mapped_job, script_physics

from native_stand.artifacts.io import write_json
from native_stand.simulation import jump_backend


def forbidden(*args, **kwargs):
    raise AssertionError("CPU measurement forbids physical integration/CUDA")


def seed():
    random.seed(11013)
    np.random.seed(11013)
    torch.manual_seed(11013)


def duration(function, count):
    started = time.perf_counter()
    for _ in range(count):
        function()
    return (time.perf_counter() - started) / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, forbidden)
    torch.cuda.init = torch.cuda._lazy_init = forbidden
    args.output.mkdir(parents=True, exist_ok=True)
    job = mapped_job(args.checkout, args.data_root)
    starts, compiles = [], []
    run = subprocess.run
    for _repeat in range(5):
        seed()
        calls = []

        def counted(command, *a, calls=calls, **kw):
            if "-shared" in command:
                calls.append(command)
            return run(command, *a, **kw)

        with tempfile.TemporaryDirectory(prefix="jump-start-") as folder:
            with patch.object(subprocess, "run", counted):
                started = time.perf_counter()
                batch = jump_backend.make_env(
                    replace(job, runtime=replace(job.runtime, envs=3)), output=folder
                )
                starts.append(time.perf_counter() - started)
            compiles.append(len(calls))
            batch.close()
    seed()
    with tempfile.TemporaryDirectory(prefix="jump-components-") as folder:
        batch = script_physics(jump_backend.make_env(job, output=folder))
        env, plant = batch.envs[0], batch.envs[0].plant
        from native_stand.tasks.jump.unified_reward import UnifiedPacket

        # The old unconsumed logger constructs ABI3 Packet at BALANCE9 and
        # faults after1 tick. Correct its packet type ONLY for the scripted
        # component comparison; startup/packing/reset measurements need no fix.
        old_logger = hasattr(plant, "reward_packets")
        plant.scoring_packet = partial(plant.scoring_packet, packet_type=UnifiedPacket)

        def pack():
            env.base._obs = env._obs = None
            env.observations()

        def reset():
            batch.reset()

        pack()
        pack_seconds = [duration(pack, 150) for _ in range(5)]
        reset_seconds = [duration(reset, 8) for _ in range(5)]
        tracemalloc.start()
        for _ in range(20):
            pack()
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        batch.reset()
        trace, hashes, terms = [], [], []
        c = plant.execution.c
        pd_start = plant.execution.C_PD_calls
        initial = batch.observations()
        trace.append(initial["student_critic"].numpy().copy())
        actor = [initial["student_actor"].numpy().copy()]
        reward, motors = [], []
        started = time.perf_counter()
        for index in range(160):
            action = torch.tensor([[0.05, -0.03, 0.1, -0.04, 0.02, -0.1]]) * ((index % 3) - 1)
            out = batch.step(action)
            trace.append(out["observations"]["student_critic"].numpy().copy())
            actor.append(out["observations"]["student_actor"].numpy().copy())
            reward.append(out["rewards"].numpy().copy())
            motors.append(np.r_[plant.actual_motor, plant.actual_NET])
            hashes.append(
                hashlib.sha256(bytes(c.states) + bytes(c.inputs) + bytes(c.frames)).hexdigest()
            )
            terms.append(env.reward.state_dict())
        scripted_seconds = time.perf_counter() - started
        np.savez_compressed(
            args.output / "trace.npz",
            critic=np.asarray(trace),
            actor=np.asarray(actor),
            rewards=np.asarray(reward),
            motor_NET=np.asarray(motors),
        )
        report = dict(
            scope="CPU component/script physical I/O; not MuJoCo rollout throughput",
            integration_calls=0,
            GPU_calls=0,
            training_updates=0,
            startup_envs=3,
            startup_seconds=starts,
            startup_median_s=statistics.median(starts),
            compilations=compiles,
            pack_repeats=150,
            pack_seconds=pack_seconds,
            pack_median_s=statistics.median(pack_seconds),
            reset_repeats=8,
            reset_seconds=reset_seconds,
            reset_median_s=statistics.median(reset_seconds),
            traced_pack_retained_bytes=retained,
            traced_pack_peak_bytes=peak,
            allocation_scope="Python traced retained/peak over20 packs; not total allocations or Torch native bytes",
            scripted_policy_steps=160,
            scripted_ticks=plant.execution.tick,
            historical_logger_packet_type_fix=old_logger,
            scripted_component_seconds=scripted_seconds,
            C_PD_calls=plant.execution.C_PD_calls - pd_start,
            C_state_frame_sha256=hashes,
            reward_memory=terms,
            C_state_bytes=C.sizeof(c.states),
            library_sha256=c.library_sha,
            schema=env.schema.contract(),
        )
        write_json(args.output / "measurement.json", report)
        batch.close()
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"schema", "reward_memory", "C_state_frame_sha256"}
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
