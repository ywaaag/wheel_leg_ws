"""Full27 PPO0 contract regression; CPU only, no simulator imports."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy,json
from pathlib import Path
import pytest
from native_stand.students.motion_confirmation import PARALLEL_VERSION,parallel_commands,verify


def test_full_course_is_unique27_with_final_endpoints():
 values=parallel_commands();assert len(values)==27 and len({tuple(v) for v in values})==27
 for target in ([4.,0.,.34],[-4.,0.,.34],[0.,4.,.34],[0.,-4.,.34]):assert target in values
 assert all(not (v[0] and v[1]) for v in values)


def fixture(tmp_path,monkeypatch):
 import torch
 p=tmp_path/'parent.json';cp=tmp_path/'model.pt';cp.write_bytes(b'fixture')
 from native_stand.students.data import sha
 keys=('reward','student_training','motion_course','motion_parallel','motion_parallel_assessment','motion_assessment','action','control','ground','observation','history','ppo','student_interface','network_architecture','performance','device','asset_path','asset_sha256')
 parent={k:{} for k in keys};parent['student_training']={'storage_max_bytes':100};parent['asset_sha256']='asset';parent['evaluation']={'scored_envs':8,'seconds':31.52};parent['student_evaluation']={'version':'motion31_parallel_highspeed_assessment_v1'};p.write_text(json.dumps(parent))
 s=copy.deepcopy(parent);s.update(source_checkpoint=str(cp),source_sha256=sha(cp),evaluation_only=True,
  motion_confirmation={'version':PARALLEL_VERSION,'parent_spec':str(p),'parent_spec_sha256':sha(p),'checkpoint_sha256':sha(cp),'seeds':[20261014]},
  budget={k:0 for k in ('source_eval_updates','smoke_updates','formal_updates','sample_limit','formal_samples')})
 s['student_evaluation']['cases']=[{'command':cmd,'seed':20261014,'checkpoint':str(cp),'sha256':sha(cp),'role':'MotionZeroStudent','kind':'student','family':'motion_native_normal','yaw_rad':0.,'seconds':31.52} for cmd in parallel_commands()]
 monkeypatch.setattr(torch,'load',lambda *a,**kw:{'spec_sha256':sha(p),'asset_sha256':'asset','critic_privileged_version':'motion_symmetry_privileged85_v1'})
 return s


@pytest.mark.parametrize('change',['none','missing','ppo','reward','count','source','seconds'])
def test_parallel_confirmation_rejects_drift(tmp_path,monkeypatch,change):
 s=fixture(tmp_path,monkeypatch)
 if change=='none':assert verify(s);return
 if change=='missing':s['student_evaluation']['cases'].pop()
 if change=='ppo':s['budget']['formal_updates']=1
 if change=='reward':s['reward']['bad']=1
 if change=='count':s['evaluation']['scored_envs']=1
 if change=='source':s['source_sha256']='x'*64
 if change=='seconds':s['student_evaluation']['cases'][0]['seconds']=10
 with pytest.raises(ValueError):verify(s)


def test_public_health_routes_confirmation_before_training_parent(tmp_path,monkeypatch):
 s=fixture(tmp_path,monkeypatch)
 from native_stand.students.motion_spin_experiment import verify as spin
 from native_stand.students.motion_course_requests import verify as course
 assert spin(s) and course(s)
