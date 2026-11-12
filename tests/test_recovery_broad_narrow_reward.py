FROZEN_TEST_SCOPE='PORTABLE'
import pytest
import torch
from native_stand.students.recovery_broad_narrow_reward import reward,configuration,FIELDS,kernel


def inputs(n=3):
    z=lambda k:torch.zeros(n,k)
    b=lambda value:torch.full((n,),value,dtype=torch.bool)
    return dict(height=torch.full((n,),.34),h0=torch.full((n,),.168),angles=z(2),
        contacts=torch.full((n,2),3.),clearance=torch.full((n,),.05),leg_q=z(4),
        angular_pitch_roll=z(2),velocity_vx_vy_yaw=z(3),applied_torque=z(6),
        previous_targets=z(6),executed_targets=z(6),entered=b(False),controlled=b(True),
        valid=b(True),fault=b(False),entry_event=b(False),success_event=b(False),failure_event=b(False),age=torch.ones(n))


def test_broad_guides_far_errors_narrow_resolves_small_errors():
    x=torch.tensor([[.30,.30],[.05,.05],[0.,0.]])
    broad=kernel(x,[.30,.30]);narrow=kernel(x,[.06,.06])
    assert broad[0]>.3 and narrow[0]<1e-8
    assert narrow[1]<broad[1] and broad[-1]==narrow[-1]==1
    v=inputs();r=reward(**v)
    assert r['fields']==FIELDS and r['raw'].shape==(3,11)
    assert torch.allclose(r['total'],torch.full((3,),.07))


def test_symmetry_is_physical_mirror_not_equal_signed_hips():
    v=inputs();v['leg_q'][0]=torch.tensor([.4,-.2,-.4,-.2])
    v['leg_q'][1]=torch.tensor([.4,-.2,.4,-.2])
    r=reward(**v)
    assert torch.equal(r['diagnostics']['mirror_error_rad'][0],torch.zeros(2))
    assert r['total'][0]>r['total'][1]
    assert r['diagnostics']['mirror_narrow'][1]<r['diagnostics']['mirror_broad'][1]


def test_low_idle_airborne_and_tilt_do_not_win_over_supported_level():
    v=inputs();v['height'][0]=.168;v['contacts'][1]=0;v['angles'][1]=torch.tensor([1.08,.4])
    r=reward(**v)
    assert r['total'][0]==0 and r['total'][1]<r['total'][2]
    assert r['diagnostics']['precision_gate'][0]==r['diagnostics']['precision_gate'][1]==0
    assert r['raw'][1,FIELDS.index('posture_huber')]>0


def test_dt_events_masks_and_failure_window_are_separate():
    v=inputs();v['entered'][:]=True;v['entry_event'][0]=True;v['success_event'][1]=True
    v['failure_event'][2]=True;v['age'][0]=1.25
    r=reward(**v)
    assert r['events'][0,0]==1.5 and r['events'][1,1]==10 and r['events'][2,1]==0
    torch.testing.assert_close(r['total'],r['weighted_density'].sum(-1)*.01+r['events'].sum(-1))
    v['controlled'][0]=False;v['fault'][1]=True;v['valid'][2]=False
    assert torch.equal(reward(**v)['total'],torch.zeros(3))
    # Caller must provide Certificate success, not an unverified window pulse.
    v=inputs();v['success_event'][0]=True
    with pytest.raises(ValueError,match='Certificate'):reward(**v)


def test_hold_costs_do_not_suppress_pre_entry_velocity_and_no_rng_draw():
    v=inputs();v['velocity_vx_vy_yaw'][:]=2
    rng=torch.get_rng_state().clone();r=reward(**v)
    assert torch.equal(rng,torch.get_rng_state())
    assert (r['raw'][:,FIELDS.index('post_entry_velocity')]==0).all()
    v['entered'][:]=True
    assert (reward(**v)['total']<r['total']).all()
    c=configuration();c['weights']['lift_trend']=100
    with pytest.raises(ValueError,match='configuration differs'):reward(**v,reward_configuration=c)
