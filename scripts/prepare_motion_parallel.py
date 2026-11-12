"""Parallel high-speed course CPU plan/draft. Never reserve or launch training."""
import argparse,json
from pathlib import Path
from native_stand.students.motion_parallel_experiment import build,verify
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.students.requests import validate
from native_stand.students.motion_reward_audit import audit
from native_stand.students.grouped_evaluation import describe_plan

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--formal-updates',type=int,default=4995,help='proposed only; not approved budget')
 p.add_argument('--write-draft',action='store_true')
 p.add_argument('--warm-quick-evaluation',dest='warm_quick_evaluation',action='store_true',default=True,help='default: explicit v2 warm-App screening; physics parity NOT_VERIFIED')
 p.add_argument('--legacy-quick-evaluation',dest='warm_quick_evaluation',action='store_false',help='explicit fallback to frozen v1 isolated child screening')
 p.add_argument('--motion-hotpath',action='store_true',help='opt-in state-epoch cache and merged timeout safety transfer; physical parity pending')
 p.add_argument('--disable-contact-history',action='store_true',help='opt-in history_length0; body coverage unchanged')
 reduction=p.add_mutually_exclusive_group()
 reduction.add_argument('--motion-runtime-reduction',dest='motion_runtime_reduction',action='store_true',default=True,help='default: Motion initialization subset/batched geometry/diagnostic buffers; physics NOT_VERIFIED')
 reduction.add_argument('--no-motion-runtime-reduction',dest='motion_runtime_reduction',action='store_false',help='explicit fallback: disable runtime reduction and its hotpath dependency unless --motion-hotpath is supplied')
 a=p.parse_args();root=Path(__file__).resolve().parents[1]
 out=root/'artifacts'/new_experiment_name('motion3250_parallel_course_draft');s=build(a.parent_spec,a.checkpoint,experiment=out.name,spec_path=out/'spec_draft.json',formal_updates=a.formal_updates)
 if a.warm_quick_evaluation:
  from native_stand.students.motion_quick_evaluation import warm_configuration
  s['checkpoint_quick_evaluation']=warm_configuration()
 s['performance'].update(motion_hotpath=a.motion_hotpath or a.motion_runtime_reduction,motion_runtime_reduction=a.motion_runtime_reduction)
 if a.disable_contact_history:s['performance']['contact_history_disabled']=True
 verify(s);errors=validate(s)
 if errors:raise ValueError(errors)
 from native_stand.command_v6.tools.storage_budget import estimate
 storage=estimate(s,'formal')
 r={'status':'CPU_PLAN_VALID','submit_ready':False,'reason_codes':['AUTHORIZATION_MISSING'],'source_sha256':s['source_sha256'],'proposed_online_updates':a.formal_updates+5,'proposed_storage_bytes':s['artifacts']['storage_max_bytes'],'budget_authorized':False,'checkpoint_quick_evaluation':s.get('checkpoint_quick_evaluation'),'warm_physical_equivalence':'NOT_VERIFIED','evaluation_plan':describe_plan(s['student_evaluation']['cases'],maximum_commands=4),'storage_estimate':storage,'simulation':0,'PPO':0,'ledger_reserved':False,'files_written':0}
 r['performance']=s['performance']
 if a.write_draft:
  out.mkdir(parents=True,exist_ok=False)
  for name,value in [('spec_draft.json',s),('reward_snapshot.json',audit(s['reward']))]:
   with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
  r.update(files_written=3,output=str(out))
  with (out/'preparation.json').open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
