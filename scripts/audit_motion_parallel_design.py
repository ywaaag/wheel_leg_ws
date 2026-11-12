"""CPU-only proposal/curves/Actor readback; no runtime registration or training budget."""
import argparse,json,math
from pathlib import Path
import torch
from native_stand.students import motion_parallel_design as design
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.mcu_interface import actor_input,task

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--parent-spec',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path)
 a=p.parse_args();actor,cp=load(a.source,'MotionZeroStudent');spec=json.loads(a.parent_spec.read_text())
 if cp.get('spec_sha256')!=sha(a.parent_spec) or cp.get('critic_privileged_version')!='motion_symmetry_privileged85_v1':raise ValueError('exact source/spec/Critic85 required')
 data=torch.load(a.dataset,map_location='cpu',weights_only=False);x=data['x'][torch.linspace(0,len(data['x'])-1,64).long(),:22]
 curves=[]
 for axis in (0,1):
  for target in (-4.,-3.,-2.,-1.,-.5,.5,1.,2.,2.5,3.,4.):
   for actual in (0.,target*.5,target,target*1.5,-target):
    for mix in (0.,.75):
     z=torch.tensor([actual],dtype=torch.float64,requires_grad=True);command=z.new_zeros(1,2);command[:,axis]=target
     v=torch.stack((z if axis==0 else z*0,z*0),-1);y=z if axis==1 else z*0
     r=design.signals(command,v,y,new_task=torch.ones(1,dtype=torch.bool),mix=z.new_full((1,),mix));value=r['density'].sum();grad=torch.autograd.grad(value,z)[0].item()
     curves.append({'axis':'vx_mps' if axis==0 else 'yaw_rad_s','target':target,'actual':actual,'mix':mix,'weighted_density':value.item(),'dt_contribution':value.item()*.01,'state_derivative':grad,'derivative_toward_target':(target-actual)*grad if target!=actual else None,'finite':math.isfinite(grad),'flat_wrong_state':target!=actual and abs(grad)<1e-4})
 readback=[]
 values={(0.,0.),*design.BASE}
 for lane in design.LEVELS:
  for level in range(len(design.LEVELS[lane])):values.update(design.endpoints(lane,level))
 with torch.inference_mode():
  for vx,yaw in sorted(values):
   cmd=torch.tensor([[vx,yaw,.34]],dtype=torch.float32).repeat(len(x),1);obs=actor_input(x,task('Motion',cmd));action=actor(obs)
   assert torch.allclose(obs[:,22:25],cmd/cmd.new_tensor([4,4,.5])) and (obs[:,25]==1).all() and (obs[:,26]==0).all() and torch.isfinite(action).all()
   readback.append({'target':[vx,yaw],'observed_command_first':obs[0,22:25].tolist(),'action_min':action.amin(0).tolist(),'action_max':action.amax(0).tolist(),'pass':True,'scope':'CPU frozen-state command intervention only; not runtime or closed-loop ability'})
 from native_stand.students.grouped_evaluation import describe_plan
 from native_stand.students.motion_parallel_assessment import protocol as assessment_protocol,safety_protocol
 sampling=design.allocation(design.initial_state())
 cases=[{'case_id':f'parallel_initial_{i}','kind':'student','role':'MotionZeroStudent','family':'motion_native_normal','seed':spec['student_evaluation']['cases'][0]['seed'],'yaw_rad':0.,'seconds':31.52,'checkpoint':str(a.source.resolve()),'sha256':sha(a.source),'command':r['target']+[.34]} for i,r in enumerate(sampling)]
 evaluation_plan=describe_plan(cases,maximum_commands=4)
 assessment=assessment_protocol(safety_protocol(spec))
 from native_stand.students.motion_reward_audit import audit
 result={'status':'CPU_DESIGN_VALIDATED_NOT_EXECUTABLE','configuration':design.configuration(),'allocation':design.allocation(design.initial_state()),'evaluation_plan':evaluation_plan,'assessment_protocol':assessment,'curves':curves,
 'curve_summary':{'points':len(curves),'finite':all(v['finite'] for v in curves),'flat_wrong_states':sum(v['flat_wrong_state'] for v in curves),'all_derivatives_toward_target':all(v['derivative_toward_target'] is None or v['derivative_toward_target']>0 for v in curves)},
 'source_actor_readback':readback,'source_fixed_std':actor.log_std.exp().tolist(),'source_reward_audit':audit(spec['student_training']['reward']),
 'identity':{str(a.source.resolve()):sha(a.source),str(a.parent_spec.resolve()):sha(a.parent_spec),str(a.dataset.resolve()):sha(a.dataset),str(Path(design.__file__).resolve()):sha(design.__file__)},
 'pending':['new explicit reward/runtime/course dispatch and full composite reward tests','state-normalized new vs old PPO/retention gradient competition not measured','source eight-env readback for new reverse/yaw and center geometry; source new-skill failure must not block learning','authorized wiring smoke actual Actor/Critic update and command→obs→reward→storage path','approved finite PPO budget, resources, storage estimate, evaluation matrix duration','scoring CPU implementation exists, but future runtime must explicitly attach matching protocol; no historical rescoring'],
 'interpretation':'Reward derivatives are physical state-error slopes, not policy gradients. Actor nonzero std proves exploration parameterization only, not useful physical exploration or convergence. No claim CPU proves PhysX wiring.',
 'simulation':0,'PPO_updates':0,'offline_optimizer_updates':0,'ledger_reserved':False,'files_written':0 if a.output is None else 3}
 if a.output:
  from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
  validate_new_experiment_name(a.output.name);a.output.mkdir(parents=True,exist_ok=False)
  (a.output/'reward_snapshot.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
  (a.output/'proposal.json').write_text(json.dumps({'configuration':result['configuration'],'source':str(a.source.resolve()),'source_sha256':sha(a.source),'allocation':result['allocation'],'authorization':'UNAUTHORIZED_RUNTIME'},indent=2,ensure_ascii=False)+'\n')
  (a.output/'validation.json').write_text(json.dumps({k:result[k] for k in ('status','curve_summary','identity','pending','interpretation','simulation','PPO_updates','offline_optimizer_updates','ledger_reserved','evaluation_plan')},indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({k:result[k] for k in ('status','curve_summary','simulation','PPO_updates','files_written')},indent=2))
if __name__=='__main__':main()
