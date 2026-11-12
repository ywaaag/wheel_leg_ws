FROZEN_TEST_SCOPE='PORTABLE'
import copy
import pytest
from native_stand.students.motion_quick_backfill import validate,VERSION
from native_stand.students.motion_quick_evaluation import configuration

def request():
 return dict(version=VERSION,experiment='fixture',parent_spec='/parent/spec.json',parent_spec_sha256='a'*64,parent_output='/parent/formal',output='/new',session='current',physical_gpu=0,gpu_uuid='GPU-test',authorization='AUTHORIZED',configuration=configuration(),checkpoints=[{'update':u,'checkpoint':f'/parent/formal/model_update_{u}.pt','sha256':'b'*64} for u in (100,200)],wall_limit_s=1800,output_max_bytes=384*2**20,online_PPO=0)

def test_fixed_authorized_scope_rejects_expansion_and_wrong_session():
 assert validate(request(),current_session='current',verify_inputs=False)
 for key,value in [('online_PPO',1),('wall_limit_s',3600),('authorization','UNAUTHORIZED'),('physical_gpu',1)]:
  r=request();r[key]=value
  with pytest.raises(ValueError):validate(r,verify_inputs=False)
 with pytest.raises(ValueError):validate(request(),current_session='other',verify_inputs=False)
 r=request();r['checkpoints'][1]['update']=300
 with pytest.raises(ValueError):validate(r,verify_inputs=False)

def test_matrix_screen_v2_explicit_scope_and_bounds():
 from native_stand.students.motion_quick_backfill import MATRIX_VERSION
 r=request();r.update(version=MATRIX_VERSION,wall_limit_s=21600,output_max_bytes=512*2**20)
 r['checkpoints']=[dict(update=u,checkpoint=f'/parent/formal/model_update_{u}.pt',sha256='b'*64) for u in (600,700,800)]
 assert validate(r,current_session='current',verify_inputs=False)
 for key,value in [('wall_limit_s',21601),('output_max_bytes',2**31),('online_PPO',1)]:
  invalid=copy.deepcopy(r);invalid[key]=value
  with pytest.raises(ValueError):validate(invalid,verify_inputs=False)
 r['checkpoints'].append(r['checkpoints'][0])
 with pytest.raises(ValueError):validate(r,verify_inputs=False)

def test_grouped_v3_explicit_scope():
 from native_stand.students.motion_quick_backfill import GROUPED_VERSION
 r=request();r.update(version=GROUPED_VERSION,wall_limit_s=21600,output_max_bytes=1024*2**20)
 r['checkpoints']=[dict(update=u,checkpoint=f'/parent/formal/model_update_{u}.pt',sha256='b'*64) for u in (100,200,300,400,700,800,900)]
 assert validate(r,current_session='current',verify_inputs=False)

def test_baseline_ranking_prefers_old_retention_and_separates_source(tmp_path):
 import json,hashlib
 from native_stand.students.motion_quick_backfill import baseline_ranking
 parent=tmp_path/'run'/'formal';parent.mkdir(parents=True);out=tmp_path/'screen';out.mkdir()
 commands=[[0,0,.34],[.5,0,.34],[1,0,.34],[1.5,0,.34],[2,0,.34],[0,.5,.34],[0,-.5,.34]]
 request={'parent_output':str(parent),'checkpoints':[]}
 for u,failed in [(100,5),(200,0)]:
  cp=parent/f'model_update_{u}.pt';cp.write_bytes(str(u).encode());digest=hashlib.sha256(cp.read_bytes()).hexdigest()
  rows=[]
  for i,c in enumerate(commands):
   ok=i!=failed
   rows.append({'command':c,'per_env':[{'accepted':ok,'hold_complete':True,'survival_passed':True,'tracking':{'MAE':.01,'pitch_hold_abs_max_rad':.08},'failed_checks':[] if ok else ['tracking']}]})
  folder=out/f'update_{u}';folder.mkdir();(folder/'summary.json').write_text(json.dumps({'status':'COMPLETED','commands':rows}))
  request['checkpoints'].append({'update':u,'checkpoint':str(cp),'sha256':digest})
 result=baseline_ranking(request,out)
 assert result['best_new_baseline_candidate']['update']==100
 assert result['source_fallback']==[] and result['accepted'] is False
