"""CPU-only deterministic fixture; never simulator or real-model optimizer."""
import copy,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
FROZEN_TEST_SCOPE='PORTABLE'
from native_stand.students import motion_distillation_pipeline as p
from native_stand.students.model import Student,load
from native_stand.students.data import sha

@pytest.fixture
def bundle(tmp_path):
 from native_stand.students.motion_spin_privileged import network
 asset=tmp_path/'asset';asset.write_text('fixture-only, no simulator')
 spec=tmp_path/'spec.json';spec.write_text(json.dumps({'asset_path':str(asset),'asset_sha256':sha(asset),'action':{'leg_scale_rad':1.5,'wheel_scale_rad_s':100.}}))
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(1);teacher=Student(True,version='multiskill_mcu_v2',network_architecture=network())
 cp=tmp_path/'teacher.pt';torch.save({'actor':teacher.state_dict(),'interface_version':'multiskill_mcu_v2','role':'MotionZeroStudent','network_architecture':network(),'training_recipe':{'network_architecture':network()},'spec_sha256':sha(spec),'asset_sha256':sha(asset)},cp)
 run=tmp_path/'capture';run.mkdir()
 for c,command in enumerate(p.COMMANDS):
  x=np.zeros((50,4,31),np.float32);x[...,4]=-1;x[...,22:25]=np.array(command)/np.array([4,4,.5]);x[...,25]=1;x[...,28]=1
  x[:,:,0]=np.linspace(-.01,.01,50)[:,None];s=np.zeros((50,4,39),np.float32);s[...,29]=1
  trace=run/f'trace_{c:02d}.npz';np.savez_compressed(trace,actor_input=x,state=s,controller=np.full((50,4,1),3),first_episode_valid=np.ones((50,4),bool),effective_command_pre=np.tile(command,(50,4,1)))
  d={'checkpoint_sha256':sha(cp),'case':{'command':command,'seed':1},'trace':{'path':trace.name,'sha256':sha(trace)},'per_env':[{'accepted':True}]*4}
  (run/f'evaluation_{c:02d}.json').write_text(json.dumps(d))
 return cp,spec,run

def test_plan_zero_write_and_prepare_episode_disjoint(bundle,tmp_path):
 cp,spec,run=bundle;before=set(tmp_path.rglob('*'));plan=p.plan(cp,spec,run)
 assert set(tmp_path.rglob('*'))==before and plan['optimizer_updates']==0 and plan['simulation']==0
 out=tmp_path/'data';p.prepare(cp,spec,run,out)
 d=torch.load(out/'dataset.pt',weights_only=False)
 assert not set(d['episode'][d['split']==0].tolist())&set(d['episode'][d['split']==1].tolist())
 assert set(d['cohort'].tolist())==set(range(7))
 teacher,_,_=p.validate_teacher(cp,spec)
 with torch.no_grad():torch.testing.assert_close(d['y'],teacher.mean(d['x']))
 with pytest.raises(FileExistsError):p.prepare(cp,spec,run,out)

def test_tiny_fixture_bc_checkpoint_and_mujoco_plan(bundle,tmp_path):
 cp,spec,run=bundle;dataset=tmp_path/'data';p.prepare(cp,spec,run,dataset);old=sha(cp)
 result=p.fit(dataset/'dataset.pt',tmp_path/'fit',steps=2,wall_s=30,batch_size=32)
 assert result['offline_optimizer_updates']==2 and result['closed_loop']=='NOT_RUN' and sha(cp)==old
 student=tmp_path/'fit/best_validation.pt';actor,saved=load(student,'MotionZeroStudent');assert actor.mlp[0].weight.shape==(128,31)
 p.validate_student(saved,spec)
 assert 'optimizer' not in saved and 'critic' not in saved and 'recipe_version' not in saved
 cmd=p.mujoco_command(student,spec,tmp_path/'sim',30);assert '--run' in cmd
 bad=copy.deepcopy(saved);bad['distillation']['teacher_sha256']='x'*64
 with pytest.raises(ValueError,match='Teacher'):p.validate_student(bad,spec)

def test_capture_wrong_command_or_teacher_rejected(bundle,tmp_path):
 cp,spec,run=bundle;ep=run/'evaluation_00.json';d=json.loads(ep.read_text());d['checkpoint_sha256']='x'*64;ep.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='checkpoint'):p.plan(cp,spec,run)

def test_raw_observation_readback_corruption_rejected(bundle):
 cp,spec,run=bundle;tp=run/'trace_00.npz'
 with np.load(tp) as z:arrays={k:z[k].copy() for k in z.files}
 arrays['actor_input'][0,0,23]=1
 np.savez_compressed(tp,**arrays);ep=run/'evaluation_00.json';d=json.loads(ep.read_text());d['trace']['sha256']=sha(tp);ep.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='readback'):p.plan(cp,spec,run)

def test_cli_fit_default_does_not_optimize(bundle,tmp_path):
 cp,spec,run=bundle;out=tmp_path/'2026_10_18-16_00__test_fit'
 cmd=[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/motion_distill_pipeline.py'),'fit','--teacher',str(cp),'--parent-spec',str(spec),'--dataset',str(tmp_path/'missing.pt'),'--output-dir',str(out)]
 child=subprocess.run(cmd,capture_output=True,text=True,check=True);assert json.loads(child.stdout)['status']=='PLAN_ONLY' and not out.exists()

def test_cli_cannot_execute_without_scope(bundle,tmp_path):
 cp,spec,run=bundle;out=tmp_path/'2026_10_18-16_01__test_fit'
 cmd=[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/motion_distill_pipeline.py'),'prepare','--teacher',str(cp),'--parent-spec',str(spec),'--capture-run',str(run),'--output-dir',str(out),'--execute']
 child=subprocess.run(cmd,capture_output=True,text=True);assert child.returncode!=0 and not out.exists()

def test_student_rollout_replay_labels_no_teacher_takeover(bundle,tmp_path):
 cp,spec,run=bundle;dataset=tmp_path/'data';p.prepare(cp,spec,run,dataset)
 p.fit(dataset/'dataset.pt',tmp_path/'fit',steps=1,wall_s=30,batch_size=32)
 student=torch.load(tmp_path/'fit/best_validation.pt',weights_only=False)
 sim=tmp_path/'sim';sim.mkdir()
 (sim/'terminal_report.json').write_text(json.dumps({'status':'COMPLETED','model_unchanged':True}))
 producer=tmp_path/'producer.py';producer.write_text('old producer')
 (sim/'manifest.json').write_text(json.dumps({'distilled_student':student['distillation'],'input_sha256':{str(cp):sha(cp),str(producer):sha(producer)}}))
 producer.write_text('new producer unrelated to saved inputs')
 for i,c in enumerate(p.COMMANDS):
  (sim/f'command_{i:02d}.json').write_text(json.dumps({'target':c[:2]}))
  np.savez_compressed(sim/f'command_{i:02d}.npz',actor_input=np.zeros((20,31),np.float32),time=np.arange(20)*.01)
 out=tmp_path/'replay';r=p.collect_replay(sim,cp,spec,out)
 assert r['samples']==140 and not r['teacher_control']
 manifest=json.loads((out/'replay_manifest.json').read_text())
 assert len(manifest['historical_source_code_drift'])==1
 old=cp.read_bytes();cp.write_bytes(b'changed checkpoint')
 with pytest.raises(ValueError,match='input changed'):p.collect_replay(sim,cp,spec,tmp_path/'bad_replay')
 cp.write_bytes(old)
 d=torch.load(out/'replay.pt',weights_only=False);assert not d['split'].any()
 teacher,_,_=p.validate_teacher(cp,spec)
 with torch.no_grad():torch.testing.assert_close(d['y'],teacher.mean(d['x']))
 p.fit(dataset/'dataset.pt',tmp_path/'refit',steps=1,wall_s=30,batch_size=32,initial_student=tmp_path/'fit/best_validation.pt',replay=out/'replay.pt')


def test_mujoco_student_identity_rejects_large_disguise(bundle,tmp_path):
 # Same metadata validator used by public planner; zero mj_step.
 from types import SimpleNamespace
 cp,spec,run=bundle;data=tmp_path/'data';p.prepare(cp,spec,run,data);p.fit(data/'dataset.pt',tmp_path/'fit',steps=1,wall_s=30,batch_size=32)
 d=torch.load(tmp_path/'fit/best_validation.pt',weights_only=False)
 assert p.validate_student(d,spec)['parent_spec_sha256']==sha(spec)
 d['network_architecture']={'version':'pretend-large'}
 with pytest.raises(ValueError,match='pretend'):p.validate_student(d,spec)


def test_relabel_new_teacher_preserves_states_and_episode_split(bundle,tmp_path):
 cp,spec,run=bundle;dataset=tmp_path/'old';p.prepare(cp,spec,run,dataset)
 saved=torch.load(cp,weights_only=False);saved['actor']['mlp.6.bias']+=.01
 new=tmp_path/'new.pt';torch.save(saved,new)
 output=tmp_path/'new_labels';before=sha(dataset/'dataset.pt')
 result=p.relabel(new,spec,dataset/'dataset.pt',output)
 assert result['files_written']==0 and not output.exists()
 p.relabel(new,spec,dataset/'dataset.pt',output,execute=True)
 old=torch.load(dataset/'dataset.pt',weights_only=False);data=torch.load(output/'dataset.pt',weights_only=False)
 assert sha(dataset/'dataset.pt')==before and data['teacher_sha256']==sha(new)
 for k in ('x','split','episode','cohort','phase'):torch.testing.assert_close(data[k],old[k])
 actor,_,_=p.validate_teacher(new,spec)
 with torch.inference_mode():torch.testing.assert_close(data['y'],actor.mean(data['x']))
 assert not torch.equal(data['y'],old['y']) and 'not new-Teacher rollouts' in data['provenance']
 with pytest.raises(FileExistsError):p.relabel(new,spec,dataset/'dataset.pt',output,execute=True)


def explicit_dataset(bundle,tmp_path):
 cp,spec,run=bundle;out=tmp_path/'explicit_data';p.prepare(cp,spec,run,out)
 data=torch.load(out/'dataset.pt',weights_only=False);manifest=json.loads((out/'manifest.json').read_text())
 extra=data['cohort']==0
 for key in ('x','y','phase','split'):
  data[key]=torch.cat([data[key],data[key][extra]])
 data['episode']=torch.cat([data['episode'],data['episode'][extra]+1000])
 data['cohort']=torch.cat([data['cohort'],torch.full((int(extra.sum()),),7)])
 data['cohorts']=[dict(case_id=f'c{i}',target=c,variant=0) for i,c in enumerate(p.COMMANDS)]+[dict(case_id='stress_zero',target=p.COMMANDS[0],variant=2)]
 torch.save(data,out/'dataset.pt');manifest.update(command_set_version='motion31_explicit_commands_v2',cohorts=data['cohorts'],dataset_sha256=sha(out/'dataset.pt'))
 (out/'manifest.json').write_text(json.dumps(manifest));return out,data,manifest


def test_explicit_cohorts_beyond_seven_are_really_trained(bundle,tmp_path):
 out,data,manifest=explicit_dataset(bundle,tmp_path)
 result=p.fit(out/'dataset.pt',tmp_path/'fit8',steps=2,wall_s=30,batch_size=32)
 assert set(result['sampling'])=={c['case_id'] for c in data['cohorts']}
 assert sum(v['draws'] for v in result['sampling'].values())==64
 assert result['sampling']['stress_zero']['draws']>0
 assert len(result['history'][-1]['per_cohort'])==8


def test_explicit_cohort_id_and_episode_overlap_rejected(bundle,tmp_path):
 out,data,m=explicit_dataset(bundle,tmp_path)
 m['cohorts'][7]['case_id']='c0'
 with pytest.raises(ValueError,match='duplicate'):p.dataset_cohorts(data,m)
 data['cohorts'][7]['case_id']='stress_zero';data['episode'][data['split']==1]=data['episode'][data['split']==0][0]
 torch.save(data,out/'dataset.pt');m['dataset_sha256']=sha(out/'dataset.pt');(out/'manifest.json').write_text(json.dumps(m))
 with pytest.raises(ValueError,match='episode split'):p.fit(out/'dataset.pt',tmp_path/'bad',steps=1,wall_s=30,batch_size=32)


def test_explicit_standing_quota_really_changes_draws_without_validation_leak(bundle,tmp_path):
 out,data,m=explicit_dataset(bundle,tmp_path)
 # This fixture has two standing cohorts (0/7), six moving cohorts.
 m['cohort_sampling_weights']=[.25]+[.5/6]*6+[.25]
 (out/'manifest.json').write_text(json.dumps(m))
 result=p.fit(out/'dataset.pt',tmp_path/'weighted',steps=4,wall_s=30,batch_size=512)
 sampling=result['sampling'];total=sum(v['draws'] for v in sampling.values())
 standing=sampling['c0']['draws']+sampling['stress_zero']['draws']
 assert total==2048 and abs(standing/total-.5)<.01
 assert all(row['unique_available']>0 and row['draws']>0 for row in sampling.values())
 m['cohort_sampling_weights'][0]=-1
 (out/'manifest.json').write_text(json.dumps(m))
 with pytest.raises(ValueError,match='sampling weights'):p.fit(out/'dataset.pt',tmp_path/'bad_weights',steps=1,wall_s=30,batch_size=32)


def test_export_records_current_control_not_legacy_constants(bundle,tmp_path):
 cp,spec,run=bundle
 raw=json.loads(spec.read_text());raw['control']=dict(physics_dt_s=.001,pd_dt_s=.005,policy_dt_s=.01,leg_kp=40.,leg_kd=2.,wheel_kv=.2)
 raw['mcu_execution']={'profile':{'version':'motion_mcu_post_pd_execution_v1'}};spec.write_text(json.dumps(raw))
 saved=torch.load(cp,weights_only=False);saved['spec_sha256']=sha(spec);torch.save(saved,cp)
 for ep in run.glob('evaluation_*.json'):
  d=json.loads(ep.read_text());d['checkpoint_sha256']=sha(cp);ep.write_text(json.dumps(d))
 out,data,m=explicit_dataset(bundle,tmp_path);fit=tmp_path/'fit_export'
 p.fit(out/'dataset.pt',fit,steps=1,wall_s=30,batch_size=32)
 exported=p.export_onnx(fit/'best_validation.pt',spec,out/'dataset.pt',tmp_path/'onnx')
 assert exported['control']['leg_Kp']==40 and exported['control']['wheel_Kv']==.2
 assert exported['parameters']==14630 and len(exported['cohorts'])==8
 assert exported['verification']['max_abs_error']<=1e-5
