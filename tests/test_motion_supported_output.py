"""显式支撑偏移的真实解码、标签、previous和C数值检查。"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from native_stand.students.motion_supported_output import SupportedKinematicMotionOutput, VERSION
from native_stand.students.motion_kinematic_output import KinematicMotionOutput

FROZEN_TEST_SCOPE = 'PORTABLE'
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/interfaces/motion31_kinematic_support_centered_015_40_v1.design.json'


def config():
    return json.loads(CONFIG.read_text())


def test_zero_action_is_support_target_and_does_not_redefine_joint_zero():
    bound = SupportedKinematicMotionOutput(config()).bind_height(torch.tensor([.34], dtype=torch.float64))
    target = bound.decode(torch.zeros(1, 6, dtype=torch.float64))['requested_target']
    expected = [[-.029116082936525345, -.09327510744333267, 0.,
                  .029116082936525345, -.09327510744333267, 0.]]
    torch.testing.assert_close(target, torch.tensor(expected, dtype=target.dtype), atol=1e-14, rtol=0)
    torch.testing.assert_close(bound.previous_from_adopted(target),
        torch.cat((target[:, [0, 1, 3, 4]]/1.5, target[:, [2, 5]]/100.), -1))


def test_support_offset_is_constant_at_every_height_and_center_is_mirrored():
    c = config(); pure = copy.deepcopy(c)
    pure['action_version'] = 'motion31_kinematic_centered_015_40_v1'
    pure['support_target_offset_rad'] = [0., 0., 0., 0.]
    h = torch.linspace(.3, .4, 101, dtype=torch.float64)
    center = SupportedKinematicMotionOutput(c).centers(h)
    base = KinematicMotionOutput(pure).centers(h)
    torch.testing.assert_close(center-base, h.new_tensor(c['support_target_offset_rad']).expand_as(center))
    torch.testing.assert_close(center[:, 2], -center[:, 0])
    torch.testing.assert_close(center[:, 3], center[:, 1])


def test_residual_is_not_projected_to_symmetry_and_out_of_range_label_is_retained():
    bound = SupportedKinematicMotionOutput(config()).bind_height(torch.tensor([.34]))
    a = torch.tensor([[.2, -.3, .4, -.7, .8, -.6]])
    target = bound.decode(a)['requested_target']
    torch.testing.assert_close(bound.labels(target)['raw_label'], a)
    assert target[0, 0] != -target[0, 3] and target[0, 1] != target[0, 4]
    target[:, 0] += 1.
    label = bound.labels(target)
    assert not label['exactly_representable'].any() and label['raw_label'][0, 0] > 1


@pytest.mark.parametrize('offset', [[-.2, -.3, -.2, -.3], [0., 0., 0., float('nan')],
                                  [0., 0., 0.], [0., 0., 0., float('inf')], None])
def test_invalid_or_nonmirrored_support_offset_is_rejected(offset):
    c = config(); c['support_target_offset_rad'] = offset
    with pytest.raises(ValueError):
        SupportedKinematicMotionOutput(c)


@pytest.mark.parametrize('field,value', [('action_version', 'other'), ('input_version', 'other'),
    ('leg_halfwidth_rad', .25), ('wheel_halfwidth_rad_s', 100.)])
def test_explicit_version_and_scales_remain_required(field, value):
    c = config(); c[field] = value
    with pytest.raises(ValueError):
        SupportedKinematicMotionOutput(c)


def test_three_c_center_versions_are_compiled_and_match_python(tmp_path):
    path = ROOT/'isolated/2026_11_05__motion_interface_research/build_centered_design.py'
    spec = importlib.util.spec_from_file_location('motion_supported_design_builder_test', path)
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    result = builder.c_check(tmp_path, config(), builder.decoder_for(config()))
    assert result['golden_samples'] == 267 and result['max_error_rad'] < 2e-6
    assert result['nonfinite_actions_rejected_without_writes']
    assert result['in_episode_height_change_rejected']
    assert config()['action_version'] == VERSION
