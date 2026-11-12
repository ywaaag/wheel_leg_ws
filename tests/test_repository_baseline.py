"""Source/data boundary checks that also run from an empty local clone."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

FROZEN_TEST_SCOPE = "PORTABLE"
ROOT = Path(__file__).resolve().parents[1]


def test_portable_guard_rejects_cuda_and_real_physics_calls(tmp_path):
    (tmp_path / "conftest.py").write_bytes((ROOT / "tests/conftest.py").read_bytes())
    (tmp_path / "test_guard.py").write_text(
        "import pytest, torch, importlib.util\n"
        "def test_no_execution():\n"
        " with pytest.raises(AssertionError,match='forbids'): torch.cuda.init()\n"
        " if importlib.util.find_spec('mujoco'):\n"
        "  import mujoco\n"
        "  with pytest.raises(AssertionError,match='forbids'): mujoco.mj_step(None,None)\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "WHEEL_LEG_PORTABLE_CPU_GUARD": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTEST_ADDOPTS": "",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ignore_rules_keep_control_sources_and_exclude_runtime_and_secrets(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    excluded = [
        "runs/x/state.json",
        "artifacts/x.pt",
        "configs/branches/allocation.json",
        "a.npz",
        ".env",
        "key.pem",
        "configs/local/paths.json",
    ]
    included = [
        "src/native_stand/simulation/control_core_source/rl_policy_core.c",
        "src/native_stand/simulation/jump_control_source/rl_control.h",
        "configs/contracts/jump_unified_raw37_wheel48_v1.json",
        "tests/test_new_cli.py",
    ]
    for path in excluded + included:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--", path], cwd=tmp_path, capture_output=True
        )
        assert result.returncode == (0 if path in excluded else 1), path


def test_source_checkout_contains_both_c_abis_and_package_resources():
    for directory in ("control_core_source", "jump_control_source"):
        source = ROOT / "src/native_stand/simulation" / directory
        assert (source / "manifest.json").is_file()
        assert list(source.glob("*.c")) and list(source.glob("*.h"))
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"]["native_stand"]
    assert "simulation/control_core_source/*" in patterns
    assert "simulation/jump_control_source/*" in patterns
    assert (ROOT / "scripts/check_architecture.py").is_file()
    assert (ROOT / ".github/workflows/cpu.yml").is_file()
