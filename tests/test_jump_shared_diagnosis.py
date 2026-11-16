"""Explicit local diagnosis can share hardware; training/default guards stay strict."""

from dataclasses import replace
from pathlib import Path

import pytest

from native_stand.artifacts.io import file_sha256, write_json
from native_stand.simulation.jump_isaac import validate_job
from native_stand.simulation.recovery_isaac.backend import require_idle_gpu

FROZEN_TEST_SCOPE = "HOST"


@pytest.fixture
def resources(tmp_path, monkeypatch):
    import native_stand.simulation.recovery_isaac.backend as backend

    authorization = tmp_path / "authorization.json"
    write_json(authorization, dict(shared_GPU_allowed=True, bounded_physical_io_only=True))
    run = tmp_path / "run.json"
    write_json(run, dict(task="recovery"))
    policy = dict(
        mode="shared_authorized",
        authorization=dict(path=str(authorization), sha256=file_sha256(authorization)),
        storage_path=str(tmp_path),
        minimum_disk_free_bytes=0,
        minimum_free_MiB=8192,
        bounded_physical_io_only=True,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        backend.subprocess,
        "check_output",
        lambda cmd, **kw: "999999999, GPU-test\n"
        if "--query-compute-apps=pid,gpu_uuid" in cmd
        else "0, GPU-test, 24000, 95\n",
    )
    original = backend.Path.read_bytes
    monkeypatch.setattr(
        backend.Path,
        "read_bytes",
        lambda p: b"python\0-m\0native_stand.experiments.worker\0--run-dir\0" + str(run).encode()
        if str(p) == "/proc/999999999/cmdline"
        else original(p),
    )
    return policy, authorization


def test_explicit_zero_PPO_diagnosis_can_share_without_changing_default_guard(resources):
    policy, _ = resources
    actual = require_idle_gpu("GPU-test", policy, allow_bounded_diagnostic_owner=True)
    assert actual["free_MiB"] == 24000
    with pytest.raises(ValueError, match="bounded physical I/O authorization"):
        require_idle_gpu("GPU-test", policy)
    regular = {k: v for k, v in policy.items() if k != "bounded_physical_io_only"}
    with pytest.raises(RuntimeError, match="conflicting training owner"):
        require_idle_gpu("GPU-test", regular, allow_bounded_diagnostic_owner=True)


def test_general_shared_permission_does_not_authorize_bounded_exception(resources):
    policy, authorization = resources
    write_json(authorization, dict(shared_GPU_allowed=True))
    policy["authorization"]["sha256"] = file_sha256(authorization)
    with pytest.raises(ValueError, match="bounded physical I/O authorization"):
        require_idle_gpu("GPU-test", policy, allow_bounded_diagnostic_owner=True)


@pytest.mark.parametrize("envs,updates", [(3, 0), (2, 1)])
def test_diagnosis_permission_cannot_start_training_or_larger_scene(monkeypatch, envs, updates):
    from fixtures.jump_cpu_io import mapped_job

    root = Path(__file__).resolve().parents[1]
    job = mapped_job(
        root, Path("/home/kellen/wheel_leg_training_data"), name="jump_isaac_raw37_wheel48"
    )
    assets = {
        **job.assets,
        "jump_isaac": {
            **job.assets["jump_isaac"],
            "resource_policy": {"bounded_physical_io_only": True},
        },
    }
    job = replace(
        job,
        runtime=replace(job.runtime, envs=envs),
        limits=replace(job.limits, formal_updates=updates),
        assets=assets,
    )
    with pytest.raises(ValueError, match="cannot carry PPO work"):
        validate_job(job)


def test_named_training_permission_is_separate_from_diagnostic_exception(resources):
    policy, authorization = resources
    regular = {k: v for k, v in policy.items() if k != "bounded_physical_io_only"}
    regular["shared_training_task"] = "jump"
    with pytest.raises(ValueError, match="shared-training authorization"):
        require_idle_gpu("GPU-test", regular, shared_training_task="jump")
    write_json(authorization, dict(shared_GPU_allowed=True, shared_training_task="jump"))
    regular["authorization"]["sha256"] = file_sha256(authorization)
    assert require_idle_gpu("GPU-test", regular, shared_training_task="jump")["free_MiB"] == 24000
    # The unchanged Recovery/default consumer still rejects a Recovery owner.
    with pytest.raises(RuntimeError, match="conflicting training owner"):
        require_idle_gpu("GPU-test", regular)
    write_json(authorization.parent / "run.json", dict(task="jump"))
    with pytest.raises(RuntimeError, match="conflicting training owner"):
        require_idle_gpu("GPU-test", regular, shared_training_task="jump")


def test_bounded_exception_cannot_be_used_as_training_permission(resources):
    policy, authorization = resources
    write_json(
        authorization,
        dict(shared_GPU_allowed=True, shared_training_task="jump", bounded_physical_io_only=True),
    )
    policy["authorization"]["sha256"] = file_sha256(authorization)
    policy["shared_training_task"] = "jump"
    with pytest.raises(ValueError, match="shared-training authorization"):
        require_idle_gpu("GPU-test", policy, shared_training_task="jump")
