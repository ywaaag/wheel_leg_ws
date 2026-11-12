"""Exercise the actual frozen episode accumulator on CPU snapshots, no engine."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from native_stand.tasks.motion.progressive_episodes import episode_class
from native_stand.tasks.motion.movement_evidence import episode_class as movement
from native_stand.tasks.motion.highspeed_course import PROGRESSION

path = Path(
    "/home/kellen/wheel_leg_native_train/isolated/2026_11_05__motion_supported2048_dqgain_ppo/onpolicy.py"
)
spec = importlib.util.spec_from_file_location("frozen_training_episodes", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cls = movement(episode_class(module.Episodes), progression_gate=PROGRESSION)
stage = dict(
    commands=[[0.0, 0.0]],
    cohort_keys=[[0.0, 0.0, 0.34, -1, "clean_standing"]],
    cohort_purposes=["clean_standing"],
    identified_profiles=[-1],
    motion_axes=[None],
    motion_endpoints=[False],
    id=0,
)
e = NS(
    num_envs=3,
    device="cpu",
    stage_spec=stage,
    student_actor_valid=torch.tensor([True, False, True]),
    rise_hold=NS(episode=NS(steps=torch.tensor([10, 10, 10]))),
    target_xy=torch.zeros(3, 2),
    recovery_snapshot=torch.zeros(3, 31),
    standing_snapshot=torch.zeros(3, 6),
    wheel_geometry_snapshot=torch.zeros(3, 9),
    antiosc_physical_stats_post=torch.zeros(3, 8),
    command_index=torch.zeros(3, dtype=torch.long),
)
e.recovery_snapshot[:, 6] = 0.34
e.recovery_snapshot[:, 10:12] = 3.0
e.wheel_geometry_snapshot[:, 8] = -1.0
e.student_reward_result = {
    "symmetry_diagnostics": {
        "rolling_common_differential_mps": torch.zeros(3, 2),
        "leg_swing_signed_difference_rad": torch.zeros(3),
        "leg_lengths_m": torch.full((3, 2), 0.3),
    },
    "differential_diagnostics": {"imbalance": torch.zeros(3), "rolling": torch.zeros(3, 2)},
}
stats = cls(e)
end = (torch.tensor([True, True, False]), torch.zeros(3, dtype=torch.bool))
stats.observe(end)
assert stats.count.tolist() == [
    1.0
], "early failure must be counted; Teacher-only must not be counted"
assert stats.n.tolist() == [0.0, 0.0, 1.0]
record = stats.completed_state()
restored = cls(e)
restored.restore_completed(record)
assert restored.n.tolist() == [
    0.0,
    0.0,
    0.0,
], "unfinished old world cannot be spliced into new world"
assert restored.count.tolist() == [1.0]
report = stats.export(100)
resumed = restored.export(100)
# Pure diagnostics can restart their sample counter; behavior evidence must match exactly.
for value in (report, resumed):
    value["movement_statistics"].pop("observed_policy_steps")
assert report == resumed
row = report["rows"][0]
assert row["ended_episodes"] == 1 and row["completed_episodes"] == 0 and row["passed_episodes"] == 0
assert row["student_control_seconds"] == 0.02 or abs(row["student_control_seconds"] - 0.02) < 1e-8
assert abs(row["teacher_control_seconds"] - 0.01) < 1e-8
assert restored.export(200)["rows"][0]["ended_episodes"] == 0
print(
    json.dumps(
        dict(
            early_failure_denominator=1,
            teacher_student_successes=0,
            restored_completed_episodes=1,
            restored_unfinished_episodes=0,
            extra_simulation_steps=0,
        )
    )
)
