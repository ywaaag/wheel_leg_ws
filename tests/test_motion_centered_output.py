"""新动作的真实数学实现检查；不运行物理，不修改旧编码器或模型。"""
import copy
import json
from pathlib import Path

import pytest
import torch

from native_stand.students.motion_centered_output import CenteredMotionOutput, VERSION
from native_stand.students.mcu_interface import encode
from native_stand.students.interface import targets as legacy_targets

FROZEN_TEST_SCOPE = 'PORTABLE'

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/interfaces/motion31_height_centered_015_40_v1.design.json'


def config():
    return json.loads(CONFIG.read_text())


def test_linear_formula_at_knots_and_middle_height_uses_command_not_current_height():
    d = CenteredMotionOutput(config())
    heights = torch.tensor([.3,.32,.34,.36,.38,.4,.33])
    delta = -3.23492*(heights-.34)
    rows = torch.stack((delta-.02994,delta-.09317,.02994-delta,delta-.09317),-1)
    torch.testing.assert_close(d.centers(heights), rows)
    torch.testing.assert_close(d.centers(torch.tensor([.33])), ((rows[1]+rows[2])/2)[None])


def test_height_binding_is_copied_and_in_episode_height_change_is_rejected():
    d=CenteredMotionOutput(config());h=torch.tensor([.34]);bound=d.bind_height(h)
    h[0]=.4
    obs=torch.zeros(1,31);obs[:,24]=.68;obs[:,25]=obs[:,28]=1
    bound.validate_actor_input(obs)
    obs[:,24]=.7
    with pytest.raises(ValueError,match='高度命令已改变'):bound.validate_actor_input(obs)


@pytest.mark.parametrize('height',[.299,.401,float('nan')])
def test_illegal_height_is_rejected_without_clamping(height):
    with pytest.raises(ValueError):CenteredMotionOutput(config()).bind_height(torch.tensor([height]))


def test_interleaved_leg_wheel_channels_and_raw_sample_are_preserved():
    e=CenteredMotionOutput(config()).bind_height(torch.tensor([.34]))
    a=torch.tensor([[2.,-2.,.5,.25,-.25,-2.]])
    out=e.decode(a)
    center=torch.zeros_like(a);center[:,[0,1,3,4]]=e.leg_center_rad
    scale=torch.tensor([.15,.15,40.,.15,.15,40.])
    torch.testing.assert_close(out['requested_target'],center+a.clamp(-1,1)*scale)
    assert out['raw_action'] is a
    assert out['raw_clipped'].tolist()==[[True,True,False,False,False,True]]
    assert not out['mechanical_clipped'].any()


def test_output_mapping_retains_policy_gradient_inside_the_range():
    e=CenteredMotionOutput(config()).bind_height(torch.tensor([.34]))
    a=torch.zeros(1,6,requires_grad=True)
    e.decode(a)['requested_target'].sum().backward()
    torch.testing.assert_close(a.grad,torch.tensor([[.15,.15,40.,.15,.15,40.]]))


def test_full_physical_target_labels_round_trip_and_outside_is_not_hidden():
    e=CenteredMotionOutput(config()).bind_height(torch.tensor([.34]))
    a=torch.tensor([[.2,-.3,.1,.4,-.5,-.2]])
    physical=e.decode(a)['requested_target'];labels=e.labels(physical)
    assert labels['exactly_representable'].all()
    torch.testing.assert_close(labels['raw_label'],a)
    physical[:,0]+=1.
    assert e.labels(physical)['raw_label'][0,0]>1
    assert not e.labels(physical)['exactly_representable'].any()


def test_previous_uses_actual_adopted_after_mechanical_protection_and_old_normalization():
    e=CenteredMotionOutput(config()).bind_height(torch.tensor([.34]))
    limits=torch.tensor([[[-.05,.05]]*4]);out=e.decode(torch.ones(1,6),limits)
    adopted=out['protected_target']
    previous=e.previous_from_adopted(adopted)
    torch.testing.assert_close(previous,torch.cat((adopted[:,[0,1,3,4]]/1.5,adopted[:,[2,5]]/100.),-1))
    assert out['mechanical_clipped'].any()
    # 真正采用来源控制器目标时仍要如实编码，不能按新包络偷偷裁掉previous。
    adopted[0,0]=.7
    assert e.previous_from_adopted(adopted)[0,0]==pytest.approx(.7/1.5)


def test_input_remains_byte_identical_to_original_31d_encoder():
    e=CenteredMotionOutput(config()).bind_height(torch.tensor([.34]))
    q=torch.tensor([[.2,.21,0.,-.2,.21,0.]]);dq=torch.tensor([[.1,-.1,2.,-.2,.2,-2.]])
    adopted=e.decode(torch.zeros(1,6))['requested_target']
    o=encode(torch.zeros(1,3),torch.tensor([[0.,-1.,0.]]),q,dq,
             adopted[:,[0,1,3,4]],adopted[:,[2,5]],torch.zeros_like(q),'Motion',torch.tensor([[.5,0.,.34]]))
    before=o.clone();e.validate_actor_input(o)
    assert torch.equal(o,before)
    torch.testing.assert_close(o[:,16:22],e.previous_from_adopted(adopted))
    assert o[0,24]==pytest.approx(.68)


def test_motion_only_and_std_physical_units():
    e=CenteredMotionOutput(config()).bind_height(torch.tensor([.34]))
    obs=torch.zeros(1,31);obs[:,24]=.68;obs[:,26]=obs[:,28]=1
    with pytest.raises(ValueError,match='Motion'):e.validate_actor_input(obs)
    torch.testing.assert_close(e.normalized_std()*torch.tensor([.15,.15,40.,.15,.15,40.]),
                               torch.tensor([.05,.05,2.,.05,.05,2.]))


@pytest.mark.parametrize('field,value',[('action_version','legacy'),('leg_halfwidth_rad',.11),
                                      ('wheel_halfwidth_rad_s',100.),('input_version','other')])
def test_declared_version_does_not_silently_accept_other_scales(field,value):
    c=copy.deepcopy(config());c[field]=value
    with pytest.raises(ValueError):CenteredMotionOutput(c)


def test_existing_action_decoder_remains_1p5_and_100():
    a=torch.tensor([[.2,-.2,.3,.4,-.4,-.5]])
    out=legacy_targets(a,torch.zeros_like(a))
    torch.testing.assert_close(out['leg_requested'],a[:,[0,1,3,4]]*1.5)
    torch.testing.assert_close(out['wheel_executed'],a[:,[2,5]]*100.)
