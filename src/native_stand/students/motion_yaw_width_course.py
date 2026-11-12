"""Fixed ±.5 yaw A/B: no promotion, equal maximum1000-update arms."""
import copy
from . import motion_yaw_course as base
VERSION='motion31_yaw_width_fixed_v1'
ASSESSMENT=base.ASSESSMENT
TARGETS=(.5,)
protocol=base.protocol
extend_mixture=base.extend_mixture

def configuration():
 c=base.configuration();c.update(version=VERSION,targets=[.5],formal_updates_max=1000,stage_updates_max=1000,
  scope='single-variable yaw denominator A/B; fixed first stage; no promotion; no retention loss')
 return c

def commands(stage):
 if stage!=0:raise ValueError('fixed yaw experiment cannot unlock another stage')
 return base.commands(0)

def stage_config(stage,envs=512):
 commands(stage);return base.stage_config(0,envs)

def cases(spec,stage):
 commands(stage);return base.cases(spec,0)

def effective_spec(spec,stage):
 commands(stage);return base.effective_spec(spec,0)

def advance(state,candidate):
 # Reuse exact same matrix gate, but never promote/stop before frozen budget.
 original=copy.deepcopy(state)
 out,decision=base.advance(state,candidate)
 out['stage']=0;out['stage_started_update']=original['stage_started_update']
 decision.update(advanced=False,all_stages_passed=False)
 return out,decision

def choose(candidates,accepted_by_stage):
 r=base.choose(candidates,accepted_by_stage)
 proof=accepted_by_stage.get('0')
 r['accepted_candidate']=next((c for c in candidates if c.get('update',0)>0 and c.get('complete_matrix') is True and proof and c.get('sha256')==proof['sha256']),None)
 r['capability_scope']='fixed±.5 yaw and old positive straight; two complete matrices, single evaluation seed only'
 return r
