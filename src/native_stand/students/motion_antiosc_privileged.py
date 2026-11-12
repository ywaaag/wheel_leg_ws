"""Explicit140D value state; actor31, old110 prefix and reward semantics preserved."""
import copy,json
from pathlib import Path
import torch
from .data import sha
VERSION='motion_antiosc_privileged140_v1'
NETWORK='motion31_antiosc_critic140_v1'
FIELDS=(('actual_common_leg_armature_kg_m2',1,.35),('current_physical_frame',16,1.),
 ('PD_request_torque_Nm',6,'effort_limits'),('previous_policy_actual_torque_Nm',6,'effort_limits'),('reward_narrow_mix',1,1.))

def network():
 from .motion_timing import network as base
 n=base();n.update(version=NETWORK,critic_input=140);return n

def contract():
 from .motion_timing import contract as base
 c=base();c.update(critic_dim=140,critic_privileged_version=VERSION)
 c['critic_privileged'] += [f'{name}:{width}/{scale}' for name,width,scale in FIELDS]
 c['critic_current_frame']='same gyro/gravity/q_leg/dq axes, order and scales as actor first16; current physical time, actor remains delayed'
 c['critic_previous_torque']='last policy boundary before current actual; reset to declared0, not previous PD substep'
 return c

def critic_input(base110,armature,current_frame,request,previous_torque,mix):
 n=len(base110)
 if base110.shape!=(n,110) or armature.shape!=(n,1) or current_frame.shape!=(n,16) or request.shape!=(n,6) or previous_torque.shape!=(n,6) or mix.shape!=(n,1):raise ValueError('explicit140 privileged shape differs')
 xs=(base110,armature,current_frame,request,previous_torque,mix)
 if any(x.dtype!=base110.dtype or x.device!=base110.device for x in xs):raise ValueError('privileged140 dtype/device differs')
 limits=base110.new_tensor([40,40,5.5,40,40,5.5])
 return torch.cat((base110,armature/.35,current_frame,request/limits,previous_torque/limits,mix),-1)

def expand_source(source):
 from .motion_timing import network as old
 if source.get('network_architecture')==network() and source.get('critic_privileged_version')==VERSION:return source
 if source.get('network_architecture')!=old() or source.get('critic_privileged_version')!='motion_timing_privileged110_v1':raise ValueError('Critic140 requires explicit110 source')
 out=copy.deepcopy(source);weight=out['critic']['mlp.0.weight']
 if weight.shape!=(512,110):raise ValueError('source110 weight differs')
 def pad(x):return torch.cat((x,torch.zeros(x.shape[0],30,dtype=x.dtype,device=x.device)),1)
 out['critic']['mlp.0.weight']=pad(weight)
 ids=[pid for group in out['optimizer']['param_groups'] for pid in group['params']]
 actor_names=list(out['actor']);critic_names=list(out['critic'])
 if len(ids)!=len(actor_names)+len(critic_names):raise ValueError('Critic140 Adam parameter order differs')
 state=out['optimizer']['state'].get(ids[len(actor_names)+critic_names.index('mlp.0.weight')],{})
 for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
  if name in state:
   if state[name].shape!=weight.shape:raise ValueError('Critic140 source Adam shape differs')
   state[name]=pad(state[name])
 out.update(network_architecture=network(),critic_privileged_version=VERSION)
 return out

def build(parent,checkpoint,*,experiment,spec_path):
 from .motion_antioscillation import VERSION as COURSE
 p=Path(parent).resolve();old=json.loads(p.read_text());root=p.parents[3];run=root/'runs'/old['experiment']
 b=json.loads(p.with_name('budget.json').read_text());terminal=json.loads((run/'formal/terminal_report.json').read_text());chain=json.loads((run/'chain_terminal_report.json').read_text())
 if old['motion_timing']['version']!=COURSE or old['motion_timing'].get('critic_upgrade') or old['student_training']['critic_privileged_version']!='motion_timing_privileged110_v1':raise ValueError('unmodified antiosc110 parent required')
 if terminal.get('stop_kind')!='user_signal' or terminal['status']!='STOPPED_BOUNDED' or chain['status']!='STOPPED_BOUNDED' or any(a['status']=='RUNNING' for a in b['attempts'].values()):raise ValueError('settled user-stopped antiosc parent required')
 # This bounded migration was explicitly requested before any formal optimizer update.
 if b['used']['formal']!=0 or terminal['actual_updates']!=0 or b['used']['smoke']!=5:raise ValueError('this migration binds real smoke5/formal0; no consumption guess')
 cp=Path(checkpoint).resolve()
 if str(cp)!=old['source_checkpoint'] or sha(cp)!=old['source_sha256']:raise ValueError('formal0 migration must retain original final500, not smoke actor')
 data=torch.load(cp,map_location='cpu',weights_only=False);expand_source(data)
 paths=[p,p.with_name('budget.json'),run/'formal/terminal_report.json',run/'chain_terminal_report.json',cp]
 inputs={str(q):sha(q) for q in paths}
 s=copy.deepcopy(old);s.update(experiment=experiment,network_architecture=network(),student_interface=contract(),
  observation={**old['observation'],'critic_dim':140})
 s['motion_timing'].update(parent_spec=str(p),parent_spec_sha256=sha(p),critic_upgrade={
  'version':VERSION,'parent_run':str(run),'inputs':inputs,'historical_online_updates':5,'remaining_online_updates':9995,
  'original_online_limit':10000,'historical_wall_s':chain['chain_wall_s'],'Actor':'original final500; formal0, smoke not initialization','new_columns':'zero weights and matching Adam moments','physical_state_restored':False})
 s['student_training'].update(critic_privileged_version=VERSION,network_architecture=network(),critic='source110_zero_pad_to140',optimizer='source_Adam_zero_new_columns')
 s['motion_training']['spec_path']=str(Path(spec_path).resolve())
 s['motion_course']['configuration'].update(formal_updates_max=9990,smoke_updates_max=5,stage_updates_max=9990)
 s['budget'].update(smoke_updates=5,formal_updates=9990,formal_samples=9990*4096*128,sample_limit=9995*4096*128)
 s['artifacts']['run_name']=experiment
 s['student_dependencies'].update(inputs)
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'User explicitly approves normal stop and Critic110-to140 restart now; actor/reward/course/PD/delays/PPO unchanged, original10000 total less real parent smoke5; new smoke5+formal9990 inclusive9995, GPU0 shared, unlimited training wall, no budget reset'}
 return s

def verify(spec):
 parent=spec['motion_timing']['parent_spec'];expected=build(parent,spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'])
 for k in ('authorization','resources','notifications'):expected[k]=spec[k]
 if expected!=spec:raise ValueError('Critic140 migration spec has undeclared drift')
 from .motion_timing import validate_recipe
 validate_recipe(spec['student_training'],spec['source_sha256'])
 claim=Path(parent).parent.parent/'.motion_critic140_claims'/(sha(parent)+'.json')
 if claim.exists() and json.loads(claim.read_text())['destination']!=spec['experiment']:raise ValueError('Critic140 parent authorization already consumed')
 return True
