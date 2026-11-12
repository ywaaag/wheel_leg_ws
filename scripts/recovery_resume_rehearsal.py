"""CPU-only Recovery state rehearsal; never creates an owner or authorizes resume."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from native_stand.artifacts.checkpoint import load_source, save_full
from native_stand.artifacts.handoff import import_checkpoint
from native_stand.artifacts.io import file_sha256, write_json
from native_stand.artifacts.model_refs import ModelIndex
from native_stand.artifacts.runtime_paths import execution_job
from native_stand.experiments.config import load_job_config, with_initialization
from native_stand.learning.policies.factory import build_policy
from native_stand.simulation.driver import Backend
from native_stand.simulation.recovery_adapter import RecoveryAdapter
from native_stand.students.resume import rng_state


def equal_tree(a, b):
    """Exact state comparison, including tensors and pickled course RNG objects."""
    import numpy as np

    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, np.random.Generator):
        equal_tree(a.bit_generator.state, b.bit_generator.state)
    elif isinstance(a, dict | TensorDict):
        assert set(a.keys()) == set(b.keys())
        for key, value in a.items():
            equal_tree(value, b[key])
    elif isinstance(a, list | tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal_tree(x, y)
    elif hasattr(a, "__dict__"):
        assert type(a) is type(b)
        equal_tree(vars(a), vars(b))
    else:
        assert a == b


class SnapshotIO:
    """Replace only physical I/O. No step, reward, course or learning substitute."""

    def __init__(self, runtime):
        self._runtime = runtime
        state = runtime["execution"]
        self.env = SimpleNamespace(
            completed=state["completed"], completed_count=state["completed_count"]
        )

    def observations(self):
        # Two rows only for storage assembly; all 1024 execution rows stay intact.
        obs = self._runtime["execution"]["issued_observations"]
        return TensorDict({k: v[:2].clone() for k, v in obs.items()}, batch_size=[2])

    def load_runtime_state(self, state):
        self._runtime = state
        self.env.completed = state["execution"]["completed"]
        self.env.completed_count = state["execution"]["completed_count"]

    def runtime_state(self):
        return self._runtime

    def close(self):
        pass


def context(job, run, used):
    return SimpleNamespace(
        job=job,
        run_dir=run,
        updates_used=used,
        refs=ModelIndex(run / "models.json"),
        store=SimpleNamespace(update=lambda **kw: None),
        restore_runtime_state=lambda payload: None,
        total_updates_before=lambda stage: used,
        curriculum_state=lambda: {},
    )


@contextmanager
def block_reads(root):
    """Prove relocated factory consumers do not silently read the old root."""
    import builtins
    import io
    from unittest.mock import patch

    def guarded(original):
        def opened(path, *args, **kwargs):
            if isinstance(path, str | Path) and Path(path).resolve().is_relative_to(root):
                raise AssertionError("relocated restore accessed source root: " + str(path))
            return original(path, *args, **kwargs)

        return opened

    with patch("builtins.open", guarded(builtins.open)), patch("io.open", guarded(io.open)):
        yield


def auxiliary_gradients(job, source, observations):
    """Isolate the real factory losses using real full-size sources, not training."""
    from native_stand.learning.ppo.recovery_action_prior import offsets

    obs = observations.clone()
    obs["student_critic"][:, 863] = 0.4  # Declared low Student Swing fixture.
    obs["student_critic"][:, offsets()["kinematics"] + 3] = 0.2
    result = {}
    for name, omit in (
        ("BC", "old_Recovery_action_prior"),
        ("prior", "qualified_physical_BC_anchor"),
    ):
        assets = {k: v for k, v in job.assets.items() if k != omit}
        handle = build_policy(
            dataclasses.replace(job, assets=assets), obs, device="cpu", source=source
        )
        handle["restore"](source.payload)
        ppo = handle["ppo"]
        before = torch.get_rng_state().clone()
        loss = ppo.actor_auxiliary_loss(
            SimpleNamespace(observations=obs), torch.ones(2, dtype=torch.bool)
        )
        gradients = torch.autograd.grad(loss, tuple(ppo.actor.parameters()), allow_unused=True)
        norm = sum(float(g.abs().sum()) for g in gradients if g is not None)
        assert norm > 0 and torch.isfinite(loss)
        torch.set_rng_state(before)
        again = ppo.actor_auxiliary_loss(
            SimpleNamespace(observations=obs), torch.ones(2, dtype=torch.bool)
        )
        assert torch.equal(loss, again)
        result[name] = dict(loss=float(loss.detach()), gradient_L1=norm, RNG_replay_exact=True)
        if name == "prior":
            assert ppo.action_prior_readback["rows"] == 2
            assert "UNKNOWN" in ppo.action_prior_readback["source"]
        del handle, ppo
    return result


def rehearse(config, checkpoint, sha256, output, path_mapping=None):
    output = Path(output).resolve()
    if not output.is_relative_to(Path("/tmp")) or output.exists():
        raise ValueError("CPU演练只写不存在的/tmp目录；不接受正式run")
    if torch.cuda.is_initialized():
        raise ValueError("演练禁止CUDA上下文")
    source = load_source(checkpoint)
    if source.sha256 != sha256:
        raise ValueError("演练来源SHA不同")
    job = with_initialization(load_job_config(config), source.path, source.sha256)
    original_root = Path(job.assets["recovery_isaac_runtime"]["path"]).parents[2]
    if path_mapping is not None:
        assets = copy.deepcopy(job.assets)
        assets["path_mapping"] = dict(path=str(path_mapping), sha256=file_sha256(path_mapping))
        job = execution_job(dataclasses.replace(job, assets=assets))
    origin = output / "before"
    dependencies = origin / "dependencies"
    dependencies.mkdir(parents=True)
    assets = copy.deepcopy(job.assets)
    refs = []
    for name in (
        "recovery_training_contract",
        "qualified_physical_BC_anchor",
        "old_Recovery_action_prior",
    ):
        ref = assets[name]
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("来源依赖身份不同: " + name)
        target = dependencies / (name + Path(ref["path"]).suffix)
        shutil.copyfile(ref["path"], target)
        ref["path"] = str(target)
        refs.append(dict(name=name, **ref))
    job = dataclasses.replace(
        job,
        assets=assets,
        runtime=dataclasses.replace(job.runtime, device="cpu", envs=2, steps_per_env=2),
    )
    io = SnapshotIO(source.payload["runtime_state"])
    handle = build_policy(job, io.observations(), device="cpu", source=source)
    handle["restore"](source.payload)
    used = source.payload["counters"]["updates"]
    state = handle["capture"](
        source.payload["counters"],
        curriculum=source.payload["curriculum"],
        execution_state=source.payload["execution_state"],
    )
    for key in (
        "actor",
        "critic",
        "optimizer",
        "optimizer_parameter_binding",
        "std",
        "rng",
        "CUDA_rng",
    ):
        equal_tree(state[key], source.payload[key])
    # A CPU-derived envelope, not an import of a live owner's authorization.
    # Device RNG is preserved verbatim in CUDA_rng, never executed on CPU.
    state.update(
        runtime_state=source.payload["runtime_state"],
        training_rng=rng_state(),
        run_identity=str(origin),
        rehearsal_only=True,
        rehearsal_source=dict(path=str(source.path), sha256=source.sha256),
    )
    cp = origin / "checkpoints/full.pt"
    save_full(cp, state)
    adapter = RecoveryAdapter(Backend("CPU-snapshot-I/O", lambda job: io))
    ctx = context(job, origin, used)
    adapter.build(ctx, "formal", load_source(cp))
    assert adapter.cursor == io.env.completed_count
    assert adapter.legacy_unscoped == io.env.completed
    captured = adapter.capture(ctx, source.payload["counters"])
    captured.update(run_identity=str(origin), rehearsal_only=True)
    for key in ("actor", "critic", "optimizer", "optimizer_parameter_binding", "std", "CUDA_rng"):
        equal_tree(captured[key], state[key])
    equal_tree(captured["execution_state"], state["runtime_state"])
    assert captured["warmup_updates_done"] == 5
    save_full(cp, captured)
    adapter.close()
    del adapter, handle
    moved = output / "relocated"
    origin.rename(moved)
    assert not origin.exists()
    relocated_assets = copy.deepcopy(assets)
    for ref in refs:
        relocated_assets[ref["name"]]["path"] = str(moved / "dependencies" / Path(ref["path"]).name)
    job = dataclasses.replace(job, assets=relocated_assets)
    cp = moved / "checkpoints/full.pt"
    transported = load_source(cp)
    # Normal same-run restore must reject an unimported physical-directory move.
    second = RecoveryAdapter(
        Backend("CPU-snapshot-I/O", lambda job: SnapshotIO(state["runtime_state"]))
    )
    try:
        second.build(context(job, moved, used), "formal", transported)
    except ValueError as exc:
        assert "身份" in str(exc)
    else:
        raise AssertionError("未授权跨目录恢复未被拒绝")
    # CPU rehearsal remaps ONLY its own temporary identity. No owner terminal is forged.
    transported.payload["run_identity"] = str(moved)
    save_full(cp, transported.payload)
    transported = load_source(cp)
    with block_reads(original_root):
        second.build(context(job, moved, used), "formal", transported)
        restored = second.capture(context(job, moved, used), source.payload["counters"])
        loss_checks = auxiliary_gradients(job, source, second.env.observations())
    for key in (
        "actor",
        "critic",
        "optimizer",
        "optimizer_parameter_binding",
        "std",
        "CUDA_rng",
        "execution_state",
        "recovery_episode_cursor",
        "recovery_training_window",
        "training_candidates",
    ):
        equal_tree(restored[key], captured[key])
    equal_tree(restored["training_rng"], captured["training_rng"])
    manifest = output / "not_authorized_handoff.json"
    write_json(
        manifest,
        dict(version="checkpoint_directory_handoff_v1", run_id="rehearsal", owner_terminated=False),
    )
    try:
        import_checkpoint(
            manifest,
            destination=output / "forbidden/full.pt",
            mapping={},
            expected_run_id="rehearsal",
        )
    except ValueError as exc:
        assert "owner未终止" in str(exc)
    else:
        raise AssertionError("活动owner导入门未生效")
    assert file_sha256(source.path) == source.sha256
    result = dict(
        source=dict(path=str(source.path), sha256=source.sha256, updates=used),
        status="CPU_REHEARSAL_PASSED_NOT_AUTHORIZED",
        formal_import=False,
        actor_critic_two_Adam_binding_std_CPU_RNG_exact=True,
        CUDA_RNG="PRESERVED_EXACT_BYTES; CUDA execution/continuity NOT_RUN",
        legacy_completed="preserved UNKNOWN history; no invented old window/cursor",
        warmup_updates_done=restored["warmup_updates_done"],
        dependency_refs=[dict(name=ref["name"], **relocated_assets[ref["name"]]) for ref in refs],
        original_root_reads_blocked_during_relocated_restore=True,
        actual_full_size_auxiliary_gradients=loss_checks,
        runtime="all 1024 observable execution rows preserved; no PhysX integration/restore",
        same_run_wrong_directory_rejected=True,
        active_owner_import_rejected=True,
        checkpoints=[str(cp)],
        new_PPO_updates=0,
        physical_steps=0,
    )
    write_json(output / "result.json", result)
    second.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--path-mapping", type=Path)
    args = parser.parse_args()
    result = rehearse(args.config, args.checkpoint, args.sha256, args.output, args.path_mapping)
    print(result["status"])


if __name__ == "__main__":
    main()
