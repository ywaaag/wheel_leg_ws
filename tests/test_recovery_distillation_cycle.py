"""Recovery aggregation uses actual pre-action states and preserves heldout tensors."""
FROZEN_TEST_SCOPE='PORTABLE'
import numpy as np
import torch
import pytest
from native_stand.students.recovery_distillation_cycle import eligible,append_replay,query


def episode():
 x=np.zeros((30,31),np.float32);x[:,24]=.68;x[:,26]=x[:,28]=1
 s=np.zeros((30,39),np.float32);s[:,6]=.34;s[:,13]=.01;s[:,29]=1
 p=np.zeros((30,10),np.float32)
 return x,s,p


@pytest.mark.parametrize('fault',['nonfinite','closure','penetration','tilt','invalid'])
def test_unreliable_measurement_query_filtered(fault):
 x,s,p=episode()
 if fault=='nonfinite':x[0,0]=np.nan
 if fault=='closure':s[0,12]=.006
 if fault=='penetration':s[0,13]=-.001
 if fault=='tilt':p[0,0]=.31
 if fault=='invalid':s[0,29]=0
 mask=eligible(x,s,p)
 assert not mask[0] and mask[1:].all()


class Teacher:
 def mean(self,x):return x.new_tensor([.6,.2,.04,-.6,-.2,-.05]).expand(len(x),-1)


def test_teacher_labels_physical_limit_and_scale_not_teacher_executed_action():
 ranges=torch.tensor([[-.3,.3]]*6)[None]
 y=query(torch.zeros(2,31),Teacher(),ranges)
 torch.testing.assert_close(y[0],torch.tensor([.2,.2,.04,-.2,-.2,-.05]))


def test_aggregation_train_only_original_heldout_unchanged():
 x,s,p=episode();data={k:torch.zeros((2,31) if k=='x' else (2,6) if k=='y' else (2,),dtype=torch.float32 if k in ('x','y') else torch.long) for k in ('x','y','stage','episode','split')}
 data['split'][:]=torch.tensor([1,2]);before={k:v.clone() for k,v in data.items()}
 stats=append_replay(data,[(x,s,p,26,np.arange(30),0,'native')],Teacher(),torch.tensor([[-2,2]]*6)[None],2)
 assert len(data['x'])==32 and stats[0]['unique_phase_samples']==[10,15,2,3]
 assert not data['split'][2:].any()
 for key in data:torch.testing.assert_close(data[key][:2],before[key])
 assert data['x'][2:,26].all() and not data['x'][2:,25].any()


def test_teacher_relative_fidelity_not_perfect_zero_required(tmp_path,monkeypatch):
 import json
 from native_stand.students.data import sha
 from native_stand.students.recovery_distillation_cycle import fidelity
 monkeypatch.chdir(tmp_path);run=tmp_path/'run';run.mkdir();refs=[];results=[]
 for i in range(4):
  common={'version':'student_autonomous_swing_v1','accepted':True,'settle_s':.45,'velocity_means':[.08,.001,.02]}
  t=tmp_path/f't{i}.json';t.write_text(json.dumps({'per_env':[common]*8}));refs.append({'path':str(t),'sha256':sha(t),'case_id':str(i)})
  p=run/f's{i}.json';p.write_text(json.dumps({'per_env':[{**common,'velocity_means':[.10,.001,.02]}]}));results.append({'path':p.name,'sha256':sha(p),'case_id':str(i)})
 manifest=tmp_path/'teacher.json';manifest.write_text(json.dumps({'evidence':refs}));(run/'summary.json').write_text(json.dumps({'status':'COMPLETED','scored_episodes':4,'results':results}))
 report=fidelity(run,manifest,{'velocity_gap_abs':[.05,.02,.03],'entry_time_gap_s':.1})
 assert report['passed'] and report['rows'][0]['student_mean_abs_vx_vy_yaw'][0]==.10
 # Matching Teacher does not authorize a different scoring protocol.
 p=run/'s0.json';d=json.loads(p.read_text());d['per_env'][0]['version']='other_protocol';p.write_text(json.dumps(d));results[0]['sha256']=sha(p)
 (run/'summary.json').write_text(json.dumps({'status':'COMPLETED','scored_episodes':4,'results':results}))
 with pytest.raises(ValueError,match='protocol'):fidelity(run,manifest,report['criteria'])


def test_mujoco_fidelity_can_copy_teacher_drift_without_task_pass(tmp_path):
 import json
 from native_stand.students.recovery_distillation_cycle import compare_mujoco,CPU_CRITERIA
 for label in ['teacher','student']:
  p=tmp_path/label;p.mkdir();s=np.zeros((100,39),np.float32);s[:,6]=.34;s[:,7]=1.;s[:,10:12]=50;s[:,13]=.01
  np.savez_compressed(p/'trajectory.npz',state=s,posture=np.zeros((100,10)),policy_time_s=np.arange(100)*.01)
  (p/'manifest.json').write_text(json.dumps({'model_xml_sha256':'same','protocol':{'policy_dt':.01},'initial_yaw_rad':0.}))
  (p/'terminal_report.json').write_text(json.dumps({'status':'COMPLETED','model_unchanged':True,'result':{'controlled_s':1.,'accepted':False,'score':{'settle_s':.5}}}))
 result=compare_mujoco(tmp_path/'teacher',tmp_path/'student',CPU_CRITERIA)
 assert result['passed'] and not result['student_accepted'] and not result['teacher_accepted']
 assert result['metrics']['vx_rmse_mps']==0
 p=tmp_path/'student/manifest.json';m=json.loads(p.read_text());m['model_xml_sha256']='different';p.write_text(json.dumps(m))
 with pytest.raises(ValueError,match='physical'):compare_mujoco(tmp_path/'teacher',tmp_path/'student',CPU_CRITERIA)


def test_cpu_task_never_dispatches_native_supervisor(tmp_path,monkeypatch):
 import json
 from native_stand.students import recovery_distillation_cycle as w
 block=tmp_path/'initial';block.mkdir();latest={'student':'small.pt','dataset':'dataset.pt'}
 (block/'summary.json').write_text(json.dumps({'rounds':[latest]}));(block/'request.json').write_text('{}')
 template=tmp_path/'template.json';template.write_text('{}')
 monkeypatch.setattr(w,'validate_task',lambda r:{'initial_BC_actual':13400})
 monkeypatch.setattr(w,'run',lambda r,extra_replays=():{'status':'COMPLETED','rounds':[latest],'offline_optimizer_steps':50})
 monkeypatch.setattr(w,'cpu_matrix',lambda *a:{'passed':True,'rows':[]})
 monkeypatch.setattr(w.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('native dispatch forbidden')))
 r={'version':w.CPU_TASK_VERSION,'initial_block':str(block),'native_request_template':str(template),'output':str(tmp_path/'task'),'minimum_free_disk_bytes':1,'criteria':w.CPU_CRITERIA}
 state=w.run_task(r)
 assert state['status']=='COMPLETED_MUJOCO_FIDELITY' and state['total_BC_steps']==13450
 assert state['completed_blocks'][0]['Isaac_episodes']==0
