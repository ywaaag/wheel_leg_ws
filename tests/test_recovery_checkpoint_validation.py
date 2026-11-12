"""Explicit multi-model PPO0 never expands the training assessment matrix."""
import copy
import pytest
from native_stand.students.checkpoint_validation import validate,VERSION

FROZEN_TEST_SCOPE='PORTABLE'


def fixture(models=15):
    cases=[]
    for u in range(models):
        for j,(seed,yaw) in enumerate([(20261010,0.),(20261010,.01),(20261011,.01),(20261012,.01)]):
            cases.append(dict(case_id=f'u{u}_{j}',checkpoint=f'model{u}.pt',sha256=str(u)*64,
                seed=seed,yaw_rad=yaw,role='RecoveryStudent',kind='student',
                family='swing_candidate_from_historical_r1',command_height_m=.34))
    return dict(evaluation_only=True,budget=dict(smoke_updates=0,formal_updates=0,sample_limit=0),
                student_training={'version':'student_recovery_broad_narrow_ppo_v1'},
                student_evaluation={'version':VERSION,'cases':cases})


def test_accepts60_explicit_cases():
    validate(fixture())


@pytest.mark.parametrize('change', ['online','duplicate','partial','condition','controller','height'])
def test_rejects_unsafe_matrix(change):
    s=copy.deepcopy(fixture())
    if change=='online':s['budget']['formal_updates']=1
    if change=='duplicate':s['student_evaluation']['cases'][4:8]=copy.deepcopy(s['student_evaluation']['cases'][:4])
    if change=='partial':s['student_evaluation']['cases'].pop()
    if change=='condition':s['student_evaluation']['cases'][1]['seed']=7
    if change=='controller':s['student_evaluation']['cases'][1]['kind']='teacher_combination'
    if change=='height':s['student_evaluation']['cases'][1]['command_height_m']=.40
    with pytest.raises(ValueError):validate(s)
