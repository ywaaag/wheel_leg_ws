"""配置单一事实来源：preset组合、严格校验、路径解析与不可变性。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from native_stand.experiments.config import (
    ConfigError,
    load_job_config,
    with_initialization,
)
from native_stand.experiments.plan import build_plan, checkpoint_bytes

FROZEN_TEST_SCOPE = "PORTABLE"  # 自包含CPU工程契约：tmp fixture + 真实代码，无本机runs/资产依赖

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "jobs" / "motion_supported015_dqgain.json"


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload))
    return path


def test_preset_plus_override_keeps_declared_defaults():
    job = load_job_config(CONFIG)
    assert job.preset == "motion_supported_v1"
    assert job.action.leg_scale_rad == 0.15
    assert job.action.wheel_scale_rad_s == 40.0
    assert job.observation.leg_input_scale_rad == 1.5
    assert job.observation.wheel_input_scale_rad_s == 100.0
    assert job.task == "motion"
    assert job.runtime.envs == 8  # 显式覆盖preset的4096


def test_input_and_output_scales_are_independent():
    job = load_job_config(CONFIG)
    assert job.action.leg_scale_rad != job.observation.leg_input_scale_rad
    assert job.action.wheel_scale_rad_s != job.observation.wheel_input_scale_rad_s


def test_unknown_top_level_field_is_rejected(tmp_path):
    payload = json.loads(CONFIG.read_text())
    payload["mystery"] = 1
    with pytest.raises(ConfigError, match="mystery"):
        load_job_config(_write(tmp_path, payload))


def test_unknown_nested_field_is_rejected(tmp_path):
    payload = json.loads(CONFIG.read_text())
    payload.setdefault("ppo", {})["learning_rate_typo"] = 1.0
    with pytest.raises(ConfigError, match="ppo"):
        load_job_config(_write(tmp_path, payload))


def test_timing_order_enforced(tmp_path):
    payload = json.loads(CONFIG.read_text())
    control = payload.setdefault("control", {})
    control["pd_dt_s"] = 0.02
    control["policy_dt_s"] = 0.01
    with pytest.raises(ConfigError, match="时相"):
        load_job_config(_write(tmp_path, payload))


def test_reward_dt_must_match_policy(tmp_path):
    payload = json.loads(CONFIG.read_text())
    payload.setdefault("reward", {})["dt"] = 0.02
    with pytest.raises(ConfigError, match="reward.dt"):
        load_job_config(_write(tmp_path, payload))


def test_sample_limit_enforced(tmp_path):
    payload = json.loads(CONFIG.read_text())
    payload.setdefault("limits", {})["sample_limit"] = 10
    with pytest.raises(ConfigError, match="样本"):
        load_job_config(_write(tmp_path, payload))


def test_relative_paths_resolve_against_config_dir(tmp_path):
    payload = json.loads(CONFIG.read_text())
    payload["initialization"] = {"mode": "full", "checkpoint": "ckpt.pt", "sha256": "a" * 64}
    job = load_job_config(_write(tmp_path, payload))
    assert job.initialization.checkpoint == (tmp_path / "ckpt.pt").resolve()


def test_full_initialization_requires_sha(tmp_path):
    payload = json.loads(CONFIG.read_text())
    payload["initialization"] = {"mode": "full", "checkpoint": "ckpt.pt"}
    with pytest.raises(ConfigError, match="sha256"):
        load_job_config(_write(tmp_path, payload))


def test_plan_is_read_only_and_complete(tmp_path):
    job = load_job_config(CONFIG)
    plan = build_plan(job)
    assert plan["total_updates"] == job.limits.smoke_updates + job.limits.formal_updates
    assert plan["semantics"]["input_scales"]["leg_rad"] == 1.5
    assert plan["semantics"]["output_scales"]["leg_rad"] == 0.15
    assert plan["storage"]["estimated_total_bytes"] > checkpoint_bytes(job)
    assert plan["semantics"]["critic_warmup_updates"] == 5


def test_notifications_require_target(tmp_path):
    payload = json.loads(CONFIG.read_text())
    payload["notifications"] = {"enabled": True}
    with pytest.raises(ConfigError, match="target"):
        load_job_config(_write(tmp_path, payload))


def test_run_id_naming_uses_shanghai_time():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from native_stand.experiments.owner import experiment_run_id

    moment = datetime(2026, 11, 5, 16, 51, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert experiment_run_id("motion_supported", now=moment) == "2026_11_5-16_51__motion_supported"
