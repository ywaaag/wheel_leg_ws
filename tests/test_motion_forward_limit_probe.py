FROZEN_TEST_SCOPE='PORTABLE'
import copy,json
from pathlib import Path
import pytest
import torch
from native_stand.students.motion_confirmation import verify,LIMIT_VERSION,LIMIT_COMMANDS
from native_stand.students.motion_guard import protocol
from native_stand.students.motion_tracking_reward import configuration,QUALITY_VERSION
from native_stand.students.data import sha


def fixture(tmp_path,monkeypatch):
 reward=configuration(QUALITY_VERSION)
 parent={'reward':reward,'student_training':{'reward':reward},**{k:{'identity':k} for k in ('action','control','ground','observation','history','ppo','student_interface')}}
 p=tmp_path/'parent.json';p.write_text(json.dumps(parent));cp=tmp_path/'source.pt';cp.write_bytes(b'source')
 monkeypatch.setattr(torch,'load',lambda *a,**k:{'critic_privileged_version':'motion_privileged49_v1','training_recipe':{'reward':reward},'recipe_version':'student_motion31_training_v1'})
 spec=copy.deepcopy(parent);spec.update(source_checkpoint=str(cp),source_sha256=sha(cp),evaluation_only=True,
 motion_confirmation={'version':LIMIT_VERSION,'parent_spec':str(p),'parent_spec_sha256':sha(p),'checkpoint_sha256':sha(cp),'seeds':[20261015]},
 motion_assessment=protocol(),budget={'smoke_updates':0,'formal_updates':0,'sample_limit':0,'formal_samples':0},
 evaluation={'scored_envs':8,'seconds':31.52,'protocol_identity':protocol()['version']},
 student_evaluation={'version':protocol()['version'],'cases':[{'seed':20261015,'command':v+[.34],'checkpoint':str(cp),'sha256':sha(cp),'family':'motion_native_normal','role':'MotionZeroStudent','seconds':31.52} for v in LIMIT_COMMANDS]})
 return spec


def test_limit_probe_requires_four_exact_endpoints_and_ppo0(tmp_path,monkeypatch):
 s=fixture(tmp_path,monkeypatch);assert verify(s)
 bad=copy.deepcopy(s);bad['student_evaluation']['cases'][0]['command'][0]=2.
 with pytest.raises(ValueError,match='matrix'):verify(bad)
 bad=copy.deepcopy(s);bad['budget']['formal_updates']=1
 with pytest.raises(ValueError,match='PPO0'):verify(bad)


def test_limit_probe_rejects_physical_drift_and_unknown_version(tmp_path,monkeypatch):
 s=fixture(tmp_path,monkeypatch);s['control']={'drift':True}
 with pytest.raises(ValueError,match='contract'):verify(s)
 s=fixture(tmp_path,monkeypatch);s['motion_confirmation']['version']='unknown'
 with pytest.raises(ValueError,match='schema'):verify(s)
