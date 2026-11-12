FROZEN_TEST_SCOPE='PORTABLE'
import pytest
from native_stand.students.requests import source_role

def test_readonly_motion_source_role_is_explicit_and_hash_bound():
    spec={'evaluation_only':True,'source_checkpoint':'model.pt','source_sha256':'a'*64,
          'student_evaluation':{'cases':[{'kind':'student','checkpoint':'model.pt','sha256':'a'*64,'role':'MotionZeroStudent'}]}}
    assert source_role(spec)=='MotionZeroStudent'
    spec['student_evaluation']['cases'][0]['sha256']='b'*64
    with pytest.raises(ValueError,match='role/SHA'):source_role(spec)

def test_training_and_unmatched_cases_keep_original_role_rules():
    assert source_role({'student_training':{'version':'student_motion31_training_v1'}})=='MotionZeroStudent'
    assert source_role({'student_training':{'version':'recovery'}})=='RecoveryStudent'
    assert source_role({'evaluation_only':True,'student_evaluation':{'cases':[]}})=='RecoveryStudent'


def test_public_student_reset_stage_rejects_training_slots_in_smaller_evaluation():
    from native_stand.students.requests import validate_evaluation_stage
    stage={'id':0,'commands':[[0.,0.]],'weights':[1.],'sampling_mode':'fixed_slots',
           'sampling_slots':[0]*512,'slot_counts':[512],'enforce_whitelist':True}
    spec={'evaluation_only':True,'evaluation':{'envs':64},'curriculum':{'stages':[stage]}}
    with pytest.raises(ValueError,match='fixed slots must cover'):validate_evaluation_stage(spec)
    stage.update(sampling_slots=[0]*64,slot_counts=[64])
    validate_evaluation_stage(spec)
