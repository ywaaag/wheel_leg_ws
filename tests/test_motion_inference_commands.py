"""Low-height inference uses the same schedule and goal state as production."""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from native_stand.artifacts.io import read_json
from native_stand.simulation.isaac.maneuver_runtime import environment_class
from native_stand.simulation.isaac.motion_commands import set_evaluation_command
from native_stand.tasks.motion.action_contract import ActionDecoder
from native_stand.tasks.motion.maneuver_program import extended_tensor_program

FROZEN_TEST_SCOPE = "PORTABLE"
ROOT = Path(__file__).resolve().parents[1]


class PhysicalIO:
    def _reset_idx(self, ids):
        self.age[ids] = 0
        self.height_adapter.target[ids] = 0.34

    def reset(self):
        self.resets += 1
        self._reset_idx(torch.arange(self.num_envs))

    def set_grouped_evaluation_targets(self, value):
        self.target_xy.copy_(value)

    def _get_observations(self):
        cmd, _ = extended_tensor_program(self.age, self.target_xy, self.maneuver_goal_height)
        self.command = cmd
        return {"actual_command": cmd}


def env():
    e = environment_class(PhysicalIO, {})()
    e.device, e.num_envs = "cpu", 2
    e.curriculum_training = False
    contract = read_json(
        ROOT / "configs/interfaces/motion_multiheight020_support_center_015_80_v2.json"
    )
    e.motion_height_range_m = ActionDecoder.from_contract(contract).center.height_range_m
    e.command_index = torch.zeros(2, dtype=torch.long)
    e.target_xy = torch.zeros(2, 2)
    e.height_adapter = NS(override=None, target=torch.full((2,), 0.34))
    e.maneuver_ready = True
    e.maneuver_goal_height = e.height_adapter.target.clone()
    e.maneuver_goal_table = torch.tensor([0.34])
    e.age, e.resets = torch.zeros(2), 0
    return e


def test_low_goal_changes_real_schedule_and_survives_partial_reset():
    e = env()
    obs = set_evaluation_command(e, [0.25, 0.0, 0.20])
    torch.testing.assert_close(obs["actual_command"][:, 2], torch.full((2,), 0.34))
    assert e.height_adapter.override is None and e.resets == 1
    e.age.fill_(15)
    torch.testing.assert_close(
        e._get_observations()["actual_command"], torch.tensor([[0.25, 0.0, 0.20]]).expand(2, -1)
    )
    e._reset_idx(torch.tensor([0]))
    torch.testing.assert_close(e.maneuver_goal_height, torch.full((2,), 0.20))
    torch.testing.assert_close(e.height_adapter.target, torch.full((2,), 0.34))


@pytest.mark.parametrize("command", [[0, 0, 0.199], [0, 0, 0.401], [float("nan"), 0, 0.34]])
def test_invalid_command_does_not_mutate_goal_or_target(command):
    e = env()
    with pytest.raises(ValueError, match="合同"):
        set_evaluation_command(e, command)
    assert e.resets == 0 and not e.target_xy.any()
    torch.testing.assert_close(e.maneuver_goal_height, torch.full((2,), 0.34))


def test_training_command_cannot_be_overridden_by_inference():
    e = env()
    e.curriculum_training = True
    with pytest.raises(ValueError, match="独立推理"):
        set_evaluation_command(e, [0, 0, 0.20])


def test_frozen_numeric_abi_has_no_historical_course_import():
    from native_stand.simulation.isaac.numeric_bridge import fixed_recovery

    assert fixed_recovery.DIM == 863
    assert sum(w for _, w in fixed_recovery.EXTRA_FIELDS) == 129
    assert len(fixed_recovery.FAMILIES) == 9
    assert fixed_recovery.EXTRA_FIELDS[-1] == ("effective_true_command", 3)
    assert fixed_recovery.AMPLITUDES == (0.25, 0.5, 0.75, 1.0)
