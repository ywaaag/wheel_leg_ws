FROZEN_TEST_SCOPE = 'PORTABLE'
import copy

import numpy as np

from native_stand.students.motion_cross_engine import score
from native_stand.students.motion_priority_audit import category,review,first_quality_failure,summarize


def trace(n=3000):
    zero=lambda *shape:np.zeros((n,*shape))
    state=zero(7);state[:,3]=.34
    return dict(time=np.arange(1,n+1)*.01,state=state,native_posture=zero(2),root_velocity_w=zero(3),
        lengths=zero(2)+.34,heading_offset=zero(),swing_difference=zero(),contact_resultant=zero(2)+30,
        motor=zero(6),driver_cap=zero(6)+5,driver=zero(6),requested_target=zero(6),adopted_target=zero(6),
        q=zero(6),dq=zero(6),support_velocity_xy=zero(2,2),rolling_geometry_valid=np.ones((n,2),dtype=bool),
        undesired_ground_contact=zero(),rolling=zero(2),differential_target=zero(),pd_request=zero(6))


def test_low_speed_only_diagnostic_does_not_change_standing():
    assert category({'command':[0.,0.,.34]})=='standing'
    for cmd in ([.5,0.,.3],[0.,-1.5,.4]):assert category({'command':cmd})=='low_speed_diagnostic'
    for cmd in ([2.,0.,.34],[-4.,0.,.34],[0.,-2.,.34]):assert category({'command':cmd})=='highspeed'


def test_native_spin_center_is_added_without_overwriting_historical_checks():
    a=trace();row={'command':[0.,.5,.34]};a['state'][:,2]=.5;a['root_velocity_w'][:,0]=.15
    a['rolling'][:]=[.1,-.1];a['differential_target'][:]=.1
    old=score(a,row);before=copy.deepcopy(old)
    assert old['accepted'] is True
    new=review(a,row,old)
    assert old==before and new['historical_pass_preserved']
    assert not new['mapped_checks']['spin_center'] and not new['mapped_gate_passed']
    assert new['native_equivalent_certified'] is False and new['unmapped_native_checks']


def test_standing_design_goal_does_not_rewrite_frozen_zero_gate():
    a=trace();row={'command':[0.,0.,.34]};a['root_velocity_w'][:,0]=.12
    old=score(a,row);new=review(a,row,old)
    assert old['accepted'] and new['mapped_gate_passed']
    assert not new['standing_design_goal_passed']
    assert not new['standing_design_goal_checks']['horizontal_mean']
    assert new['standing_design_goal_checks']['horizontal_p95']


def test_quality_state_machine_replayed_with_real_grace_and_streak():
    a=trace(100);a['native_posture'][:,0]=.4
    r=first_quality_failure(a,{'command':[0.,0.,.34]})
    assert r=={'observed':True,'controlled_time_s':.75,'causes':['pitch']}
    assert len(a['time'])==100  # no truncation or synthetic reset


def test_joint_mirror_and_quality_hard_failure_are_distinct_checks():
    a=trace();a['q'][:,0]=.7;row={'command':[0.,0.,.34]}
    new=review(a,row,score(a,row))
    assert not new['mapped_checks']['joint_mirror_quality']
    assert new['native_quality_termination_replay']['causes']==['joint_mirror']
    assert new['native_quality_termination_replay']['controlled_time_s']==.01


def test_native_full_tilt_and_smoothness_checks_are_explicit():
    a=trace();a['native_posture'][:]=[.4,.4]
    a['requested_target'][1000:1800,2]=np.arange(800)%2*10.
    row={'command':[2.,0.,.34]}
    new=review(a,row,score(a,row))
    assert new['metrics']['tilt_full_max_rad']>.5
    assert not new['mapped_checks']['tilt_full']
    assert not new['mapped_checks']['antiosc_wheel_delta']


def test_incomplete_high_speed_is_not_a_pass_and_low_speed_failure_is_not_required():
    rows=[]
    for i in range(60):
        rows.append(dict(model='model',termination='WINDOW_COMPLETE',review=dict(
            category='standing' if i<12 else 'highspeed' if i<24 else 'low_speed_diagnostic',
            full_window_completed=True,historical_pass_preserved=True,mapped_gate_passed=i<24,
            standing_design_goal_passed=i<12)))
    result=summarize(rows,['model'])[0]
    assert result['all_required_mapped_passed'] and result['groups']['low_speed_diagnostic']['mapped_passed']==0
    rows[15]['review']['mapped_gate_passed']=False;rows[15]['review']['full_window_completed']=False
    rows[15]['termination']='FALL'
    assert not summarize(rows,['model'])[0]['all_required_mapped_passed']
