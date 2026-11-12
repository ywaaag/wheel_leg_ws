"""Prepare bounded public-CLI A/B jobs or analyze completed metrics. Never launches training."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from pathlib import Path

import numpy as np

from native_stand.artifacts.io import read_json, write_json
from native_stand.experiments.config import load_job_config


def prepare(config, output, baseline, candidate):
    raw = read_json(config)
    job = load_job_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    commands = []
    for label, source in (("baseline", baseline), ("candidate", candidate)):
        value = copy.deepcopy(raw)
        value["source_root"] = str(Path(source).resolve())
        value["name"] = "motion_performance_" + label + "_25"
        value["limits"] = {
            "smoke_updates": 0,
            "formal_updates": 25,
            "sample_limit": 25 * job.runtime.samples_per_update,
            "chain_wall_s": 1800.0,
            "evaluation_wall_s": None,
        }
        value["notifications"] = {"enabled": False}
        value.setdefault("storage", {})["metrics_every_updates"] = 1
        value.setdefault("diagnostics", {})["performance_measurement"] = {
            "warmup_updates": 5,
            "measured_updates": 20,
        }
        path = output / (label + ".json")
        write_json(path, value)
        load_job_config(path)
        command = [
            sys.executable,
            "-m",
            "native_stand",
            "train",
            "--config",
            str(path),
            "--runs-root",
            str(output / "runs"),
        ]
        commands.append(
            "PYTHONPATH="
            + shlex.quote(str(Path(__file__).resolve().parents[1] / "src"))
            + " "
            + shlex.join(command)
        )
    return dict(
        status="NOT_RUN",
        warmup=5,
        measurement=20,
        commands=commands,
        requirements="separate explicit physics/GPU authorization; same model/config/envs/DR/diagnostics/save policy; baseline must record per-update timing and memory",
    )


def compare(first, second):
    runs = [Path(first), Path(second)]
    configs = [read_json(p / "run.json")["config"] for p in runs]
    keys = (
        "model",
        "observation",
        "action",
        "control",
        "reward",
        "ppo",
        "curriculum",
        "storage",
        "runtime",
        "diagnostics",
    )
    for key in keys:
        if configs[0][key] != configs[1][key]:
            raise ValueError("A/B配置不同: " + key)
    if (
        configs[0]["initialization"] != configs[1]["initialization"]
        or configs[0]["assets"] != configs[1]["assets"]
    ):
        raise ValueError("A/B来源/Plant/Teacher/DR身份不同")
    result = []
    for run, config in zip(runs, configs, strict=True):
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        selected = [r for r in rows if 6 <= r["update"] <= 25]
        if {r["update"] for r in selected} != set(range(6, 26)):
            raise ValueError("缺少预热5轮后完整20轮逐轮样本")
        times = np.array([r["seconds_per_update"] for r in selected], float)
        if not np.isfinite(times).all() or (times <= 0).any():
            raise ValueError("实际单轮耗时无效")
        memory = [r.get("peak_memory_allocated_bytes") for r in selected]
        samples = config["runtime"]["envs"] * config["runtime"]["steps_per_env"]
        result.append(
            dict(
                run=str(run),
                median_s=float(np.median(times)),
                P95_s=float(np.quantile(times, 0.95)),
                transitions_per_s=float(samples * 20 / times.sum()),
                peak_memory_allocated_bytes=max(memory)
                if all(x is not None for x in memory)
                else "NOT_RECORDED",
            )
        )
    return dict(
        status="MEASURED",
        warmup_updates=5,
        measured_updates=20,
        results=result,
        scope="actual recorded rollout/PPO update time; saving remains in the same configured schedule",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline-source", required=True)
    p.add_argument("--candidate-source", required=True)
    p = sub.add_parser("compare")
    p.add_argument("baseline")
    p.add_argument("candidate")
    args = parser.parse_args()
    result = (
        prepare(args.config, args.output, args.baseline_source, args.candidate_source)
        if args.operation == "prepare"
        else compare(args.baseline, args.candidate)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
