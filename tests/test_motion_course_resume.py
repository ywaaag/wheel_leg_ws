"""Portable course strategy and full training-state regression."""
import copy
import json
from pathlib import Path

import pytest

from native_stand.tasks.motion import c0_boundary_course,c0_fast_yaw_course
from native_stand.tasks.motion.course_strategy import resolve

FROZEN_TEST_SCOPE = 'PORTABLE'
FIXTURES=Path(__file__).parent/'fixtures'


def source95():
    state=json.loads((FIXTURES/'motion_C0_course_state.json').read_text())
    strategy=resolve(c0_boundary_course.VERSION)
    stage=strategy.stage_config(state['stage'],4096,state=state)
    assert len(stage['commands'])==95
    state['assessment_keys']=stage['cohort_keys']
    mixes=[dict(target=target,mix=.125) for target in stage['commands']]
    return dict(motion_course_state=state,reward_refinement_state=copy.deepcopy(mixes),
        symmetry_mix_state=copy.deepcopy(mixes),online_updates_including_smoke=3302)


def test_source95_preparation_preserves_layout_and_mixes():
    source=source95();strategy=resolve(c0_boundary_course.VERSION)
    state=strategy.prepare(source,4096)
    assert len(state['assessment_keys'])==95
    assert len(source['reward_refinement_state'])==len(source['symmetry_mix_state'])==95
    assert all(row['mix']==.125 for row in source['symmetry_mix_state'])


def test_explicit_yaw_floor_transition_rebinds_all_cohorts():
    source=source95();strategy=resolve(c0_fast_yaw_course.VERSION)
    state=strategy.prepare(source,4096)
    stage=strategy.stage_config(state['stage'],4096,state=state)
    # Removing yaw<1 rehearsal changes95 source cohorts to86 target cohorts.
    assert len(stage['commands'])==86
    assert len(source['reward_refinement_state'])==len(source['symmetry_mix_state'])==len(state['assessment_keys'])==86
    assert state['frontiers']['yaw_positive']==2
    old_keys = {tuple(k) for k in source95()['motion_course_state']['assessment_keys']}
    for key,row in zip(state['assessment_keys'],source['symmetry_mix_state'],strict=True):
        assert row['mix'] == (.125 if tuple(key) in old_keys else 0.)


def review(state,good_axes=(),bad_axes=()):
    strategy=resolve(state['version']);stage=strategy.stage_config(state['stage'],4096,state=state)
    rows=[]
    for i,key in enumerate(stage['cohort_keys']):
        axis=stage['motion_axes'][i]
        passed=axis in good_axes or stage['cohort_purposes'][i]=='clean_standing'
        rows.append(dict(key=key,command=stage['commands'][i],purpose=stage['cohort_purposes'][i],
            profile=stage['identified_profiles'][i],axis=axis,endpoint=stage['motion_endpoints'][i],
            ended_episodes=64,completed_episodes=64,pass_fraction=1. if passed else .4,
            quality_pass_fraction=0. if axis in bad_axes else 1.))
    return dict(stage_revision=state['stage'],rows=rows)


def test_two_confirmation_only_one_axis_and_preserve_other_streak():
    strategy=resolve(c0_boundary_course.VERSION);source=source95();state=strategy.prepare(source,4096)
    state['last_training_update']=-1
    old=state['frontiers'].copy()
    state,_=strategy.advance(state,review(state,['forward']),100,envs=4096)
    state,_=strategy.advance(state,review(state,['forward','yaw_positive']),200,envs=4096)
    assert state['frontiers']['forward']==old['forward']+1
    assert state['frontiers']['backward']==old['backward']
    assert state['pass_streak']['yaw_positive']==1


def test_quality_regression_is_direction_local_and_low_speed_failure_is_not_regression():
    strategy=resolve(c0_boundary_course.VERSION);source=source95();state=strategy.prepare(source,4096)
    state['last_training_update']=-1
    old=state['frontiers'].copy()
    for update in (100,200,300):state,_=strategy.advance(state,review(state,bad_axes=['backward']),update,envs=4096)
    assert state['frontiers']['backward']==old['backward']-1
    assert state['frontiers']['forward']==old['forward']
    assert state['frontiers']['yaw_negative']==old['yaw_negative']
    # Every command failed following while physical quality stayed good: no other regression.
    assert state['regression_streak']['forward']==0


def test_unknown_course_rejected():
    with pytest.raises(ValueError,match='未知'):resolve('unknown95')
