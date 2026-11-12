"""CPU behavior of the current-plant data adapter, no simulator launch."""
FROZEN_TEST_SCOPE='PORTABLE'
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from motion_mcu_distillation import episode,make_dataset,compare,uncertainty,solver_profile,VERSION,PRECISE_VERSION


def arrays():
 x=np.zeros((3000,31),np.float32);x[:,4]=-1;x[:,24]=.68;x[:,25]=x[:,28]=1
 return dict(time=np.arange(1,3001)*.01,actor_input=x,effective_command=np.tile([0,0,.34],(3000,1)),q=np.full((3000,6),99.))


class Teacher:
 def mean(self,x):return x[:,:6]*.1


def test_labels_use_saved_pre_action_inputs_and_whole_episode_split(tmp_path):
 a=arrays();case=dict(case_id='zero',target=[0,0,.34],variant=0);teacher=Teacher()
 train=episode(a,teacher,case,0,5,0);val=episode(a,teacher,case,0,6,1)
 assert set(train['phase'].tolist())==set(range(5))
 torch.testing.assert_close(train['y'],teacher.mean(train['x']))
 request=dict(cases=[case],teacher='frozen',teacher_sha256='a'*64,parent_spec='spec',parent_spec_sha256='b'*64)
 dataset=make_dataset([train,val],request,tmp_path/'data',{})
 d=torch.load(dataset,weights_only=False);assert not set(d['episode'][d['split']==0].tolist())&set(d['episode'][d['split']==1].tolist())
 bad=episode(a,teacher,case,0,5,1)
 with pytest.raises(ValueError,match='episode split'):make_dataset([train,bad],request,tmp_path/'bad',{})


def test_stress_timing_is_post_pd_and_actor_input_unchanged():
 r=uncertainty(dict(variant=2))
 assert r['actor_age_ms']==10 and r['PD_age_ms']==12 and r['motor_delay_ms']==[12,12,9,12,12,13]
 assert r['spring_scale']==1.1


def test_teacher_failure_cannot_be_a_fidelity_certificate():
 base=dict(case_id='zero',target=[0,0,.34],variant=0,termination='FALL',physical_seconds=1.,hold_complete=False)
 result=compare([base],[base])
 assert result['passed']==0 and not result['capability_certified']
 with pytest.raises(ValueError,match='paired'):compare([base],[dict(base,case_id='different')])


def test_full_window_preservation_ranks_before_partial_case_count():
 def row(cid,termination='WINDOW_COMPLETE',response=0.):
  return dict(case_id=cid,target=[0,0,.34],variant=0,termination=termination,physical_seconds=30. if termination=='WINDOW_COMPLETE' else 10.,
   hold_complete=True,response=response,pitch_peak=.01,roll_peak=.01,height_MAE=.001,leg_difference_MAE=.001,both_contact_fraction=1.,driver_limit_fraction=0.)
 teacher=[row(str(i)) for i in range(3)]
 safe=[row(str(i),response=.06) for i in range(3)]
 partial=[row('0'),row('1'),row('2',termination='FALL')]
 assert compare(teacher,safe)['rank']>compare(teacher,partial)['rank']


def test_precision_distillation_is_explicit_and_old_request_keeps_old_solver():
 old=dict(version=VERSION)
 assert solver_profile(old)=='legacy_xml_v1'
 with pytest.raises(ValueError,match='historical'):
  solver_profile(dict(old,solver_profile='closure_contact_tight_v1'))
 new=dict(version=PRECISE_VERSION,objective='teacher_fidelity',
  plant_profile='motion_mcu_currentplant_v1',solver_profile='closure_contact_tight_v1')
 assert solver_profile(new)=='closure_contact_tight_v1'
 for key in ['objective','plant_profile','solver_profile']:
  bad=dict(new);bad.pop(key)
  with pytest.raises(ValueError):solver_profile(bad)


def test_relative_fidelity_does_not_require_teacher_zero_drift_or_absolute_task_pass():
 def row(response,solver='closure_contact_tight_v1'):
  return dict(case_id='zero',target=[0,0,.34],variant=0,termination='WINDOW_COMPLETE',physical_seconds=30.,
   hold_complete=True,response=response,pitch_peak=.01,roll_peak=.001,height_MAE=.002,
   leg_difference_MAE=.001,both_contact_fraction=1.,driver_limit_fraction=0.,
   solver_profile=solver,plant_profile='motion_mcu_currentplant_v1',absolute_task_passed=False)
 result=compare([row(-.15)],[row(-.16)])
 assert result['all_fidelity_passed'] and not result['capability_certified']
 with pytest.raises(ValueError,match='solver/plant'):
  compare([row(-.15)],[row(-.16,'legacy_xml_v1')])


def test_v3_standing_dataset_records_quota_and_keeps_episode_split(tmp_path):
 a=arrays();teacher=Teacher();cases=[dict(case_id='zero',target=[0,0,.34],variant=0),dict(case_id='forward',target=[.5,0,.34],variant=0)]
 request=dict(version=PRECISE_VERSION,objective='teacher_fidelity',solver_profile='closure_contact_tight_v1',
  plant_profile='motion_mcu_currentplant_v1',standing_batch_fraction=.5,cases=cases,
  teacher='frozen',teacher_sha256='a'*64,parent_spec='spec',parent_spec_sha256='b'*64)
 parts=[episode(a,teacher,cases[0],0,10,0),episode(a,teacher,cases[0],0,11,1)]
 path=make_dataset(parts,request,tmp_path/'new_data',{})
 import json
 m=json.loads(path.with_name('manifest.json').read_text())
 assert m['cohort_sampling_weights']==[.5,.5] and m['solver_profile']=='closure_contact_tight_v1'
 assert m['objective']=='teacher_fidelity' and m['plant_profile']=='motion_mcu_currentplant_v1'
