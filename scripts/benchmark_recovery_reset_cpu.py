"""Scripted-I/O CPU reset comparison, not GPU throughput or physical benchmark."""

import argparse
import collections
import statistics
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from native_stand.artifacts.io import file_sha256, write_json
from native_stand.learning.ppo.performance import Measurement
from native_stand.simulation.recovery_isaac.observations import physical_profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_recovery_reset_batching import baseline_module, make_env, reset_class  # noqa: E402


class Operations(TorchDispatchMode):
    def __init__(self):
        self.calls = collections.Counter()
        self.bytes = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        self.calls[name] += 1
        if args and isinstance(args[0], torch.Tensor):
            self.bytes[name] += args[0].numel() * args[0].element_size()
        return func(*args, **(kwargs or {}))


def mark_ended(env):
    env.episode_steps[:] = 100
    env.student_seconds[:] = 1.0
    env.life.last_result = {"done": torch.ones(env.num_envs, dtype=torch.bool)}
    env.HP10_count[:] = 500
    env.HP10_sum[:] = 0


def benchmark(output, n, repeats):
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU benchmark must not use CUDA")
    results = {}
    with (
        tempfile.TemporaryDirectory(prefix="recovery-reset-reference-") as temp,
        pytest.MonkeyPatch.context() as mp,
    ):
        temp = Path(temp)
        old_env = baseline_module(temp, "environment")
        old_physics = baseline_module(temp, "physics")
        old_execution = baseline_module(temp, "execution")
        old_obs = baseline_module(temp, "observations")
        cls = [reset_class(mp, old_env), reset_class(mp)]
        environments = [
            make_env(
                cls[0],
                n=n,
                physics_class=old_physics.PhysicalParameters,
                execution_class=old_execution.Execution,
            ),
            make_env(cls[1], n=n),
        ]
        profiles = [old_obs.physical_profile, physical_profile]
        for e, profile in zip(environments, profiles, strict=True):
            e._reset_idx(None)
            profile(e)
        for name, indices in [
            ("single", torch.tensor([n // 2])),
            ("partial16", torch.arange(min(16, n))),
            ("full", torch.arange(n)),
            ("empty", torch.empty(0, dtype=torch.long)),
        ]:
            results[name] = {}
            for variant, e, profile in zip(
                ("baseline", "optimized"), environments, profiles, strict=True
            ):
                mark_ended(e)
                e.robot.root_view.calls.clear()
                e.robot.body_write_order.clear()
                tolist_rows = []
                original = torch.Tensor.tolist

                def tolist(tensor, tolist_rows=tolist_rows, original=original):
                    tolist_rows.append(
                        dict(shape=list(tensor.shape), bytes=tensor.numel() * tensor.element_size())
                    )
                    return original(tensor)

                failure = None
                with patch.object(torch.Tensor, "tolist", tolist), Operations() as ops:
                    try:
                        e._reset_idx(indices)
                        profile(e)
                    except IndexError as exc:
                        if variant != "baseline" or len(indices):
                            raise
                        failure = repr(exc)
                physical_calls = list(e.robot.root_view.calls)
                timings = []
                for _ in range(0 if failure else repeats):
                    mark_ended(e)
                    timer = Measurement(True, "cpu")
                    with timer.section("reset_and_profile"):
                        e._reset_idx(indices)
                        profile(e)
                    timings.append(timer.seconds["reset_and_profile"])
                results[name][variant] = dict(
                    selected_rows=len(indices),
                    envs=n,
                    median_ms=statistics.median(timings) * 1000 if timings else None,
                    error=failure,
                    samples_ms=[t * 1000 for t in timings],
                    host_scalar_reads=ops.calls["aten._local_scalar_dense.default"],
                    tolist_calls=len(tolist_rows),
                    tolist_exposed_bytes=sum(r["bytes"] for r in tolist_rows),
                    tolist_rows=tolist_rows,
                    clone_calls=ops.calls["aten.clone.default"],
                    clone_input_bytes=ops.bytes["aten.clone.default"],
                    physical_getters=[r for r in physical_calls if r[0] == "get"],
                    material_setters=[r for r in physical_calls if r[0] == "set"],
                    body_setters_first_reset=e.robot.body_write_order[:3],
                    material_full_buffer_elements=n * 4 * 3,
                    profile_rebuilt_rows=len(indices)
                    if variant == "optimized"
                    else (0 if len(indices) == 0 else n),
                    public_measurement="Measurement.section; CPU boundary wall only",
                )
    result = dict(
        scope="CPU scripted I/O, zero engine integration/online PPO; no GPU speed claim",
        baseline="16f81c8",
        current_source={
            str(p.relative_to(ROOT)): file_sha256(p)
            for p in (ROOT / "src/native_stand/simulation/recovery_isaac").glob("*.py")
        },
        envs=n,
        repeats=repeats,
        results=results,
        GPU="NOT_RUN",
        transfer_scope="host exposures and tensor clone inputs measured on CPU; not real PCIe bandwidth or GPU synchronization time",
    )
    write_json(output, result)
    for name, case in results.items():
        print(
            name,
            [
                (
                    k,
                    round(v["median_ms"], 3) if v["median_ms"] is not None else v["error"],
                    v["host_scalar_reads"],
                    v["tolist_calls"],
                )
                for k, v in case.items()
            ],
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    benchmark(args.output, args.envs, args.repeats)
