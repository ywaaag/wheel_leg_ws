"""检验选定几何中心及旧31D输入语义；不运行物理或优化器。"""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from native_stand.students.motion_kinematic_output import KinematicMotionOutput, VERSION
from native_stand.students.motion_centered_output import CenteredMotionOutput
from native_stand.students.mcu_interface import encode

FROZEN_TEST_SCOPE = 'PORTABLE'
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/interfaces/motion31_kinematic_centered_015_40_v1.design.json'


def config():
    return json.loads(CONFIG.read_text())


def test_geometry_is_user_selected_formula_in_original_model_joint_coordinates():
    h = torch.tensor([.30, .32, .34, .36, .38, .40, .33], dtype=torch.float64)
    q = KinematicMotionOutput(config()).centers(h)
    rows = []
    for height in h.tolist():
        length = height+.000024-.06
        beta = math.acos((length**2+.21**2-.25**2)/(2*length*.21))
        hip = beta-math.radians(47.1033888)
        active = beta-math.radians(47.1080837)
        rows.append([hip, active, -hip, active])
    torch.testing.assert_close(q, torch.tensor(rows, dtype=h.dtype), atol=5e-16, rtol=0)
    torch.testing.assert_close(q[2], torch.tensor([.21318300528, .21310106382,
                                                 -.21318300528, .21310106382], dtype=h.dtype))
    assert (q[1:, 0][:5] < q[:-1, 0][:5]).all()


def test_geometric_center_does_not_silently_reuse_old_support_target_offset():
    q = KinematicMotionOutput(config()).bind_height(torch.tensor([.34]))
    target = q.decode(torch.zeros(1, 6))['requested_target']
    assert target[0, 0] > .21 and target[0, 1] > .21
    assert config()['support_target_offset_rad'] == [0., 0., 0., 0.]
    label = q.labels(torch.tensor([[-.029116083, -.093275107, 0., .029116083, -.093275107, 0.]]))
    assert not label['exactly_representable'].any()
    assert label['raw_label'].abs().max() > 2.


def test_decode_and_previous_use_separate_physical_scales():
    bound = KinematicMotionOutput(config()).bind_height(torch.tensor([.34]))
    a = torch.tensor([[1., -1., 1., -1., 1., -1.]], requires_grad=True)
    out = bound.decode(a)
    center = torch.zeros_like(a); center[:, [0, 1, 3, 4]] = bound.leg_center_rad
    scale = a.new_tensor([.15, .15, 40., .15, .15, 40.])
    torch.testing.assert_close(out['requested_target'], center+a*scale)
    out['requested_target'].sum().backward()
    torch.testing.assert_close(a.grad, scale.expand_as(a))
    adopted = out['requested_target'].detach()
    previous = bound.previous_from_adopted(adopted)
    expected = torch.cat((adopted[:, [0, 1, 3, 4]]/1.5, adopted[:, [2, 5]]/100.), -1)
    torch.testing.assert_close(previous, expected)
    q = torch.zeros_like(a)
    obs = encode(torch.zeros(1, 3), torch.tensor([[0., -1., 0.]]), q, q,
                 adopted[:, [0, 1, 3, 4]], adopted[:, [2, 5]], q,
                 'Motion', torch.tensor([[0., 0., .34]]))
    before = obs.clone(); bound.validate_actor_input(obs)
    assert torch.equal(obs, before)
    torch.testing.assert_close(obs[:, 16:22], previous)


def test_height_is_bound_once_and_raw_action_is_preserved():
    h = torch.tensor([.34]); bound = KinematicMotionOutput(config()).bind_height(h)
    h[:] = .4
    raw = torch.tensor([[2., -2., 2., -2., 2., -2.]])
    out = bound.decode(raw)
    assert out['raw_action'] is raw and out['raw_clipped'].all()
    obs = torch.zeros(1, 31); obs[:, 24] = .68; obs[:, 25] = obs[:, 28] = 1
    bound.validate_actor_input(obs)
    obs[:, 24] = .7
    with pytest.raises(ValueError, match='高度命令已改变'):
        bound.validate_actor_input(obs)


@pytest.mark.parametrize('height', [.299, .401, float('nan'), float('inf')])
def test_invalid_height_is_rejected(height):
    with pytest.raises(ValueError):
        KinematicMotionOutput(config()).centers(torch.tensor([height]))


@pytest.mark.parametrize('field,value', [('action_version', 'motion31_height_centered_015_40_v1'),
    ('input_version', 'other'), ('leg_halfwidth_rad', .11), ('wheel_halfwidth_rad_s', 100.),
    ('support_target_offset_rad', [-.24, -.3, .24, -.3])])
def test_other_contracts_and_unapproved_support_offsets_are_rejected(field, value):
    c = config(); c[field] = value
    with pytest.raises(ValueError):
        KinematicMotionOutput(c)


@pytest.mark.parametrize('field,value', [('equivalent_upper_m', .01), ('equivalent_lower_m', .001),
    ('wheel_radius_m', -.06), ('hip_above_root_m', float('nan')), ('clearance_m', .001)])
def test_invalid_geometry_is_rejected_before_any_action(field, value):
    c = copy.deepcopy(config()); c['center_formula'][field] = value
    with pytest.raises(ValueError):
        KinematicMotionOutput(c)


def test_historical_linear_contract_remains_explicit_and_unchanged():
    old = json.loads((CONFIG.parent/'motion31_height_centered_015_40_v1.design.json').read_text())
    center = CenteredMotionOutput(old).centers(torch.tensor([.34]))
    torch.testing.assert_close(center, torch.tensor([[-.02994, -.09317, .02994, -.09317]]))
    assert config()['action_version'] == VERSION
    with pytest.raises(ValueError):
        CenteredMotionOutput(config())


@pytest.mark.parametrize('filename', ['motion31_kinematic_centered_015_40_v1.design.json',
                                    'motion31_height_centered_015_40_v1.design.json'])
def test_real_compiled_c_decoder_matches_each_explicit_python_contract(tmp_path, filename):
    path = ROOT/'isolated/2026_11_05__motion_interface_research/build_centered_design.py'
    spec = importlib.util.spec_from_file_location('motion_centered_design_builder_test', path)
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    c = json.loads((CONFIG.parent/filename).read_text())
    result = builder.c_check(tmp_path, c, builder.decoder_for(c))
    assert result['golden_samples'] == 267
    assert result['max_error_rad'] < 2e-6 and result['max_error_rad_s'] < 1e-5
    assert result['nonfinite_actions_rejected_without_writes']
    assert result['in_episode_height_change_rejected']
