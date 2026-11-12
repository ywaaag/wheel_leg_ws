"""统一CLI契约：帮助、dry-run、退出码与不提前加载Isaac。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from native_stand.cli import main

FROZEN_TEST_SCOPE = "PORTABLE"  # 自包含CPU工程契约：tmp fixture + 真实代码，无本机runs/资产依赖

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "jobs" / "motion_supported015_dqgain.json"


def _run(args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "native_stand", *args],
        cwd=str(cwd or PROJECT),
        env={
            "PYTHONPATH": str(PROJECT / "src"),
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
    )


def test_help_does_not_import_isaac_or_init_gpu():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "train" in result.stdout
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from native_stand.cli import main\n"
            "try:\n"
            "    main(['--help'])\n"
            "except SystemExit:\n"
            "    pass\n"
            "heavy=[m for m in sys.modules if m.startswith(('isaacsim','isaaclab','omni'))]\n"
            "import torch\n"
            "print('heavy', heavy, 'cuda', torch.cuda.is_initialized())\n",
        ],
        cwd=str(PROJECT),
        env={"PYTHONPATH": str(PROJECT / "src"), "CUDA_VISIBLE_DEVICES": "",
             "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert "heavy []" in probe.stdout
    assert "cuda False" in probe.stdout


def test_module_entry_matches_console_parser():
    from native_stand.cli import build_parser

    parser = build_parser()
    assert parser.prog == "wheel-leg"
    result = _run(["inspect", "--help"])
    assert result.returncode == 0
    assert "--run" in result.stdout


def test_dry_run_writes_nothing(tmp_path):
    before = set((PROJECT / "runs").glob("*"))
    result = _run(["train", "--config", str(CONFIG), "--dry-run", "--json"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["writes"] is False
    assert payload["starts_engine"] is False
    assert payload["notifies"] is False
    assert payload["total_updates"] == 60
    assert set((PROJECT / "runs").glob("*")) == before


def test_invalid_config_reports_field_and_exit_2(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x", "task": "motion", "unknown_section": {}}))
    result = _run(["train", "--config", str(bad), "--dry-run"], cwd=tmp_path)
    assert result.returncode == 2
    assert "错误" in result.stderr or "配置" in result.stderr


def test_unknown_field_inside_section_is_rejected(tmp_path):
    payload = json.loads(CONFIG.read_text())
    payload.setdefault("control", {})["leg_kp_typo"] = 1.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    result = _run(["train", "--config", str(bad), "--dry-run"], cwd=tmp_path)
    assert result.returncode == 2
    assert "control" in result.stderr


def test_unsupported_backend_is_invalid_not_silent(tmp_path):
    payload = json.loads(CONFIG.read_text())
    payload["backend"] = "isaac"
    config = tmp_path / "isaac.json"
    config.write_text(json.dumps(payload))
    result = _run(["train", "--config", str(config), "--run-id", "_should_not_exist"], cwd=tmp_path)
    assert result.returncode == 2
    assert not (PROJECT / "runs" / "_should_not_exist").exists()


def test_export_requires_explicit_selection(tmp_path):
    code = main(["export", "--run", str(tmp_path)])
    assert code == 2


def test_missing_run_is_invalid():
    assert main(["inspect", "--run", "definitely_missing_run"]) == 2


def test_json_mode_is_machine_readable(tmp_path, capsys):
    code = main(["--json", "train", "--config", str(CONFIG), "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert json.loads(out)["backend"] == "cpu"


@pytest.mark.parametrize("command", ["evaluate", "distill", "diagnose"])
def test_plan_only_commands_are_honest(command, tmp_path):
    argv = [command, "--config", str(CONFIG), "--dry-run"]
    if command == "evaluate":
        # 评估必须指名模型：dry-run也不会把latest当成best。
        assert main(argv) == 2
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"placeholder")
        argv = argv + ["--checkpoint", str(checkpoint)]
    assert main(argv) == 0
