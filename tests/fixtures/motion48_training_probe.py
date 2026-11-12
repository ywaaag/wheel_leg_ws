import sys,json,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from tensordict import TensorDict
from native_stand.experiments.config import load_job_config
from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter
from native_stand.tasks.motion.maneuver_statistics import episode_class,RESPONSE_VERSION
from native_stand.students.resume import rng_state,restore_rng
sys.path.insert(0,'tests')
from test_phase3_fixes import _production_context
job=load_job_config('configs/jobs/motion48_turn_response_DR_full_resume.json')
work=Path(tempfile.mkdtemp(prefix='motion48_adapter_'));context=_production_context(work,job,smoke=100)
class Adapter(MotionProductionAdapter):
 def _open_environment(self):
  self.env=NS(num_envs=job.runtime.envs,device='cpu',stage_spec=self.spec['curriculum']['stages'][0],transient_events=NS(summary=lambda:{}),identified_dr=NS(episodes=torch.zeros(job.runtime.envs,dtype=torch.long)))
  self.env.onpolicy_episodes=episode_class(object,version=RESPONSE_VERSION)(self.env)
  self.obs=TensorDict({'student_actor':torch.zeros(2,31),'student_critic':torch.zeros(2,877),'recovery_actor_valid':torch.ones(2,1,dtype=torch.bool),'motion_diagnostic_cohort':torch.zeros(2,1),'supported_source_retention_cohort':torch.full((2,1),-1,dtype=torch.long)},[2])
  self.modules={'rng_state':rng_state}
 def _assemble_training(self):
  from native_stand.learning.ppo.supported import assemble_from_full
  self.algorithm=assemble_from_full(self.source,self.obs,self.job,fresh=False)
  restore_rng(dict(self.source['training_rng'],torch_cuda=None))
  self._verify_restore();self.fixed_std=self.algorithm.actor.log_std.detach().clone()
 def _write_restoration(self):pass
 def physical_readback(self):return {'physics_steps':0}
a=Adapter()
try:
 a.build(context,'smoke',None)
 context.updates_used=100
 a._update_scope={'stats':NS(export=lambda:{}),'diagnostics':{},'detail':{},'channel_clips':torch.zeros(6),'valid_samples':torch.tensor(1.)}
 a.after_update(context,100,{})
 data=a.capture(context,{'updates':100})
 assert all(torch.equal(x,y) for x,y in zip(data['training_rng']['torch_cuda'],a.source['training_rng']['torch_cuda'],strict=True))
 ref=context.save(data,update=100,stage='smoke');a.after_save(context,ref)
 from native_stand.artifacts.io import read_json,write_json,file_sha256
 from native_stand.artifacts.checkpoint import load_source
 from native_stand.artifacts.handoff import import_checkpoint,VERSION
 saved=load_source(ref['checkpoint'])
 old_record=read_json(context.run_dir/'run.json')
 deps=list(saved.payload['required_dependencies'].values())
 manifest=dict(version=VERSION,run_id=old_record['logical_run_id'],owner_terminated=True,
  checkpoint={'path':ref['checkpoint'],'sha256':ref['sha256']},dependencies=deps,
  original_run_identity=str(context.run_dir.resolve()),config_contract_sha256=saved.payload['config_contract_sha256'],
  budget={'updates_used':100,'remaining':0,'limits':{'smoke':100,'formal':0}},parent_settlement={'updates':2097})
 manifest_path=work/'handoff.json';write_json(manifest_path,manifest)
 new_context=_production_context(work/'relocated',job,smoke=100)
 new_record=read_json(new_context.run_dir/'run.json');new_record['logical_run_id']=old_record['logical_run_id'];write_json(new_context.run_dir/'run.json',new_record)
 mapping={r['path']:r['path'] for r in deps};mapping[ref['checkpoint']]=ref['checkpoint']
 loaded=import_checkpoint(manifest_path,destination=new_context.run_dir/'checkpoints/import.pt',mapping=mapping,expected_run_id=old_record['logical_run_id'])
 new_context.updates_used=100
 b=Adapter();b.build(new_context,'smoke',loaded)
 assert b.state['consumed_window']==a.state['consumed_window']
 assert b.algorithm.source_anchor.state_dict()==a.algorithm.source_anchor.state_dict()
 assert b.source['warmup_updates_done']==50
 new_context.metrics.close()
 print(json.dumps({'cross_directory_resume':True,'checkpoint':ref,'stage':a.state['stage'],'cohorts':len(a.state['assessment_keys']),'first_publication':True,'KL':type(a.algorithm.source_anchor).__name__,'PPO_updates':0,'physics_steps':0,'source_sha256':a.source_sha}))
finally:context.metrics.close()
