"""Prepare opt-in Recovery1500 R1/R2 DR, public supervisor only; no-write default."""
import argparse,copy,json
from pathlib import Path
from native_stand.students import recovery_dr as dr
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.contact_quality import recipe
from native_stand.students.requests import validate
from native_stand.command_v6.tools.experiment_preparation import publish
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'configs/experiments/2026_10_17-17_59__recovery31_standing_contact_quality/spec.json'
SOURCE=ROOT/'runs/2026_10_17-17_59__recovery31_standing_contact_quality/formal/model_update_1500.pt'

def build(parent=PARENT,checkpoint=SOURCE,*,name=None):
 parent,checkpoint=Path(parent).resolve(),Path(checkpoint).resolve()
 if sha(checkpoint)!=dr.SOURCE:raise ValueError('fixed1500 source SHA differs')
 actor,cp=load(checkpoint,'RecoveryStudent');s=copy.deepcopy(json.loads(parent.read_text()))
 if cp['spec_sha256']!=sha(parent) or cp['critic_privileged_version']!='recovery_contact64_v2' or cp['asset_sha256']!=dr.ASSET:raise ValueError('1500 parent/critic/asset differs')
 name=name or new_experiment_name('recovery1500_friction_pd_dr5000')
 r=recipe(dr.SOURCE);r['critic']='source_same_semantic_contact64_v2'
 r['milestones']=list(range(250,4989,250))+[4988];r['final_seeds']=[20271101,20271102,20271103]
 s.update(experiment=name,source_checkpoint=str(checkpoint),source_sha256=dr.SOURCE,student_training=r,
  recovery_domain_randomization=dr.configuration(),seed=20261018,
  authorization={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'registration required for latest explicit5000 unlimited-wall authorization'},
  notifications={'enabled':False},restore_source_rng=False,restore_recovery_curriculum=False,
  phase_initialization=dr.configuration()['initialization'],recipe_parent_spec=str(parent),recipe_parent_spec_sha256=sha(parent),
  experiment_variable='R1 actual wheel/Plane average-pair friction then R2 shared-LR independent Kp/Kd/Kv gains; reset-only; same reward/actor/critic semantics',
  resources={'mode':'shared','sharing_authorized':True,'authorization_note':'用户同意Recovery DR接线并授权5000累计PPO更新、不限时；GPU1共享，不抢占或自动换卡'},execution_source_root=str(ROOT/'src'))
 s['device'].update(physical_index=1,uuid='GPU-dfc076b6-b082-3065-512a-e21064067db1',logical='cuda:0')
 s['budget'].update(smoke_updates=12,formal_updates=4988,sample_limit=5000*1024*128,formal_samples=4988*1024*128,
  source_eval_updates=0,source_eval_wall_s=3600,smoke_wall_s=None,formal_wall_s=None,chain_wall_s=None,evaluation_wall_s=3600)
 s['evaluation'].update(checkpoint_every_updates=100,full_every_updates=250,full_at_final=True)
 s['artifacts'].update(run_name=name,storage_max_bytes=r['storage_max_bytes'],terminal_reserve_bytes=512*2**20)
 s['student_evaluation']['max_output_bytes']=128*2**20
 for row in s['student_evaluation']['cases']:row.update(checkpoint=str(checkpoint),sha256=dr.SOURCE)
 s['student_dependencies'].update({str(parent):sha(parent),str(checkpoint):dr.SOURCE})
 errors=validate(s)
 if errors:raise ValueError(errors)
 book={'authorization':'UNAUTHORIZED','experiment':name,'limits':{'source_eval':1,'smoke':12,'formal':4988,'evaluate':0,'play':0},'used':dict.fromkeys(('source_eval','smoke','formal','evaluate','play'),0),'attempts':{},'sample_limit':5000*1024*128,'source_eval_wall_used_s':0.,'total_online_PPO_limit':5000,'hapi_session':None}
 return s,book

def prepare(*,write=False):
 s,book=build();folder=ROOT/'configs/experiments'/s['experiment']
 report={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/s['experiment']),'status':'PREPARED_UNAUTHORIZED','source':s['source_checkpoint'],'source_sha256':dr.SOURCE,'dr_version':dr.VERSION,'updates':5000,'smoke':12,'formal':4988,'raw_samples':5000*1024*128,'wall_s':None,'storage_bytes':4*2**30,'simulation':0,'optimizer_steps':0}
 if write:publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':report},s['student_dependencies'],completion_marker=True)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args();print(json.dumps(prepare(write=a.write),indent=2))
