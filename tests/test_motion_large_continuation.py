FROZEN_TEST_SCOPE='PORTABLE'
import copy
from pathlib import Path
import torch
import pytest
from native_stand.students.motion_large_continuation import latest


def test_latest_numeric_and_incomplete_rejects_without_fallback(tmp_path):
 fields={'actor':{},'critic':{},'optimizer':{},'training_rng':{},'reward_refinement_state':[]}
 for n in (9,100):torch.save({**fields,'update':n},tmp_path/f'model_update_{n}.pt')
 assert latest(tmp_path)[0].name=='model_update_100.pt'
 torch.save({'update':200},tmp_path/'model_update_200.pt')
 with pytest.raises(ValueError,match='incomplete'):latest(tmp_path)


def test_large_resume_keeps_independent_original_teacher(tmp_path):
 from native_stand.students.model import Student
 from native_stand.students.motion_large_combo import network,VERSION
 from native_stand.students.motion_training import declaration
 from native_stand.students.motion_tracking_reward import COMBO_VERSION
 from native_stand.students.motion_source_retention import declaration as retention,add_metadata
 from native_stand.students.motion_startup import smoke_profile
 from native_stand.students.ppo import assemble
 from native_stand.students.data import sha
 from native_stand.recovery.controlled_ppo import KEY
 from tensordict import TensorDict
 teacher=Student(True,version='multiskill_mcu_v2');p=tmp_path/'teacher.pt'
 torch.save({'interface_version':'multiskill_mcu_v2','actor':teacher.state_dict()},p)
 actor=Student(True,version='multiskill_mcu_v2',network_architecture=network());source=tmp_path/'latest.pt'
 cfg=declaration('a'*64,reward_version=COMBO_VERSION)
 cfg.update(network_architecture=network(),course_version=VERSION,course_stage=0,source_policy_retention=retention(sha(p)),continuation_teacher={'checkpoint':str(p),'sha256':sha(p)},smoke_wiring=smoke_profile())
 torch.save({'interface_version':'multiskill_mcu_v2','actor':actor.state_dict(),'network_architecture':network(),'training_recipe':cfg},source)
 obs=TensorDict({'student_actor':torch.zeros(8,31),'student_critic':torch.zeros(8,49),KEY:torch.ones(8,1,dtype=torch.bool)},batch_size=[8]);obs=add_metadata(obs,torch.zeros(8,2),cfg['source_policy_retention'])
 assembled=assemble(source,obs,steps_per_env=4,recipe=cfg)
 assert assembled.actor.mlp[0].weight.shape[0]==512
 assert assembled.source_retention.teacher.mlp[0].weight.shape[0]==128
 assert torch.equal(assembled.source_retention.teacher.mlp[0].weight,teacher.mlp[0].weight)
 cfg['continuation_teacher']['sha256']='b'*64
 with pytest.raises(ValueError,match='SHA'):assemble(source,obs,steps_per_env=4,recipe=cfg)
