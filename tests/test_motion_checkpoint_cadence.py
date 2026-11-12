FROZEN_TEST_SCOPE='PORTABLE'
import copy
from types import SimpleNamespace
import pytest
import torch
from native_stand.students.motion_runner import checkpoint_plan,save_unevaluated,motion_observations


def test_save100_eval250_union_no_duplicate_at500_and_old_default():
 recipe={'milestones':[250,500,750,1000],'checkpoint_every_updates':100}
 decisions={i:checkpoint_plan(recipe,'formal',i,1000) for i in range(1,1001)}
 assert [i for i,(save,evaluate) in decisions.items() if save]==[100,200,250,300,400,500,600,700,750,800,900,1000]
 assert [i for i,(save,evaluate) in decisions.items() if evaluate]==[250,500,750,1000]
 assert decisions[500]==(True,True)
 old={'milestones':[250,500,750,1000]}
 assert checkpoint_plan(old,'formal',100,1000)==(False,False)
 assert checkpoint_plan(old,'formal',250,1000)==(True,True)
 assert checkpoint_plan(recipe,'formal',995,995)==(True,True)
 assert checkpoint_plan(recipe,'smoke',4,5)==(False,False)
 assert checkpoint_plan(recipe,'smoke',5,5)==(True,True)


def test_periodic_checkpoint_only_registers_unevaluated_latest(monkeypatch,tmp_path):
 from native_stand.students import motion_runner,motion_yaw_course
 source={'checkpoint':'source.pt','sha256':'a'*64,'update':0,'stage':0,'complete_matrix':True,
         'evaluations':[{'case':{'command':command+[.34]},'per_env':[{'env':i,'accepted':True,'survival_passed':True,'tracking':{'MAE':.1}} for i in range(8)]} for command in motion_yaw_course.commands(0)]}
 source['per_env']=[r for record in source['evaluations'] for r in record['per_env']]
 original=copy.deepcopy(source);candidates=[source];progress={};writes={};saved=[]
 def fake_save(output,spec,ppo,update):
  saved.append(update);return {'checkpoint':str(tmp_path/f'model_update_{update}.pt'),'sha256':'b'*64,'update':update,'stage':0}
 monkeypatch.setattr(motion_runner,'save',fake_save)
 monkeypatch.setattr(motion_runner,'write',lambda path,data:writes.setdefault(path.name,copy.deepcopy(data)))
 result=save_unevaluated(tmp_path,{},None,100,progress,candidates,lambda values:motion_yaw_course.choose(values,{}))
 assert saved==[100] and not result['evaluated'] and not result['complete_matrix']
 selection=writes['candidate_selection.json']
 assert selection['latest_checkpoint']['update']==100
 assert selection['best_new_evaluated_candidate'] is None and selection['accepted_candidate'] is None
 assert selection['best_overall']['checkpoint']=='source.pt' and source==original
 assert set(writes)=={'candidate_registry.json','candidate_selection.json','progress.json'}
 assert progress['latest_checkpoint']==result


def test_checkpoint_interval_schema_positive_integer_and_old_recipe_unchanged():
 from native_stand.students.motion_training import declaration
 from native_stand.students.training_recipe import validate_instance
 recipe=declaration();source=recipe['source_actor_sha256'];original=copy.deepcopy(recipe)
 for bad in (0,-1,True,100.,'100'):
  r={**recipe,'checkpoint_every_updates':bad}
  with pytest.raises(ValueError,match='checkpoint interval'):validate_instance(r,source)
 recipe['checkpoint_every_updates']=100
 assert validate_instance(recipe,source)==recipe
 assert validate_instance(original,source)==original


def test_metadata_rebuilt_after_evaluation_reset_without_changing_actor_input():
 from native_stand.students.motion_source_retention import declaration,KEY
 from native_stand.recovery.controlled_ppo import KEY as VALID
 env=SimpleNamespace(num_envs=2,target_xy=torch.tensor([[0.,.5],[.5,0.]]),actor_valid_pre=lambda:torch.tensor([True,True]))
 raw={'student_actor':torch.zeros(2,31),'student_critic':torch.zeros(2,49)}
 cfg=declaration('a'*64)
 obs=motion_observations(env,raw,cfg)
 assert obs[KEY].squeeze(-1).tolist()==[-1,1]
 env.target_xy=torch.tensor([[0.,0.],[0.,-.5]])
 after=motion_observations(env,raw,cfg)
 assert after[KEY].squeeze(-1).tolist()==[0,-1]
 assert obs[KEY].squeeze(-1).tolist()==[-1,1]
 assert torch.equal(after['student_actor'],raw['student_actor'])
 assert after[VALID].all()
 assert KEY not in motion_observations(env,raw)
