FROZEN_TEST_SCOPE='PORTABLE'
import torch,pytest
from native_stand.students.parallel_checkpoint_probe import Actors
from native_stand.students.checkpoint_validation import validate,FAST_VERSION

class Actor(torch.nn.Module):
 def __init__(self,value):super().__init__();self.value=value
 def forward(self,obs,stochastic_output=False):return torch.ones(len(obs['student_actor']),6)*self.value

def test_dispatch_partial_init_reset_and_permuted_ids():
 a=Actors([Actor(.1),Actor(.2),Actor(.3)])
 x={'student_actor':torch.zeros(2,31)}
 y=a.act_for_envs(x,torch.tensor([2,0]))
 torch.testing.assert_close(y,torch.tensor([[.3]*6,[.1]*6]))
 assert a.act_for_envs({'student_actor':torch.zeros(0,31)},torch.empty(0,dtype=torch.long)).shape==(0,6)

def test_single_condition_explicit_contract_no_duplicate_models():
 cases=[dict(case_id=f'u{i}',checkpoint=f'{i}.pt',kind='student',role='RecoveryStudent',family='swing_candidate_from_historical_r1',yaw_rad=0.,seed=20261010,command_height_m=.34) for i in range(5)]
 s=dict(evaluation_only=True,budget=dict(formal_updates=0,smoke_updates=0,sample_limit=0),student_training={'version':'student_recovery_broad_narrow_ppo_v1'},student_evaluation={'version':FAST_VERSION,'cases':cases},evaluation={'envs':5,'scored_envs':1})
 validate(s)
 s['evaluation']['envs']=64
 with pytest.raises(ValueError):validate(s)

def test_existing_owner_finalizer_produces_hash_bound_report_and_route(tmp_path):
 import json,numpy as np
 from native_stand.students.parallel_checkpoint_probe import finalize
 from native_stand.students.data import sha
 from native_stand.command_v6.completion import notify
 a=np.zeros((3,1,39));p=np.zeros((3,1,10));a[:,:,6]=.34
 np.savez_compressed(tmp_path/'trace.npz',state=a,posture=p,first_episode_valid=np.ones((3,1),dtype=bool),controller=np.ones((3,1,1))*3)
 r={'entry_index':None,'hold_complete':False,'accepted':False,'failed_checks':['not_entered_within_2p5s']}
 (tmp_path/'evaluation_00.json').write_text(json.dumps({'case':{'checkpoint':'model_update_2900.pt'},'checkpoint_sha256':'a'*64,'per_env':[r],'trace':{'path':'trace.npz','sha256':sha(tmp_path/'trace.npz')}}))
 (tmp_path/'terminal_report.json').write_text(json.dumps({'attempt_id':'exact','conditions':[{'status':'COMPLETED','result':'evaluation_00.json'}]}))
 spec={'authorization':{'hapi_session':'current'},'notifications':{'enabled':True}}
 finalize(tmp_path,spec,'COMPLETED')
 assert sha(tmp_path/'completion_report.md')==json.loads((tmp_path/'report_ready.json').read_text())['report_sha256']
 (tmp_path/'manifest.json').write_text(json.dumps({'hapi_session':'current','notification_route':{'owner_session':'current','target_session':'current'}}))
 (tmp_path/'source_snapshot').mkdir();(tmp_path/'source_snapshot/spec.json').write_text(json.dumps(spec))
 sent=[]
 def send(session,message):sent.append(session);return {'status':'sent'}
 assert notify(tmp_path,send=send)['status']=='sent'
 notify(tmp_path,send=send)
 assert sent==['current']
