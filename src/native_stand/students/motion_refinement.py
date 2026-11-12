"""Explicit-data refinement state; no simulator/files/RNG, never unlocks a speed."""
import math

ABSOLUTE_EVIDENCE_VERSION='motion31_absolute_refinement_evidence_v1'

def initial(commands):
 return [{'target':list(c),'mix':0.,'consecutive_eligible':0} for c in commands]

def advance(state,records,policy):
 if len(state)!=len(records):raise ValueError('refinement complete command matrix required')
 result=[];decisions=[]
 for old,record in zip(state,records):
  if record.get('case',{}).get('command',[])[:2]!=old['target']:raise ValueError('refinement command identity differs')
  rows=record.get('per_env',[]);eligible=len(rows)==8 and old['target'][0]!=0
  for row in rows:
   t=row.get('tracking',{});checks=row.get('checks',{})
   metrics=[t.get(k) for k in ('MAE','direction','gain')]
   finite=all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in metrics)
   required=policy['required_checks']
   absolute_evidence=(row.get('version') in ('motion31_posture_reacquisition_assessment_v1','motion31_decoupled_height_assessment_v1')
                      and row.get('refinement_evidence_version')==ABSOLUTE_EVIDENCE_VERSION)
   if row.get('version') in ('motion31_absolute_posture_assessment_v4','motion31_yaw_capability_v1') or absolute_evidence:
    required=[k for k in required if k not in ('pitch_source_guard','roll_source_guard')]+['pitch_absolute_quality','roll_absolute_quality']
   if absolute_evidence:
    # This protocol also owns swing/height/geometry gates beyond the legacy list.
    # Untagged historical records keep their original refinement replay semantics.
    eligible=eligible and row.get('accepted') is True
   eligible=eligible and finite and all(checks.get(k) is True for k in required)
   if 'quality_complete_window' in checks:
    eligible=eligible and all(checks.get(k) is True for k in ('quality_complete_window','roll_quality','leg_balance_quality','wheel_alignment_quality','joint_mirror_quality'))
   if finite:eligible=eligible and metrics[0]<=policy['MAE_max'] and metrics[1]>=policy['direction_min'] and policy['gain_range'][0]<=metrics[2]<=policy['gain_range'][1]
  streak=old['consecutive_eligible']+1 if eligible else 0;mix=old['mix']
  if streak>=policy['confirmations']:
   mix=min(policy['maximum_mix'],mix+policy['mix_increment']);streak=0
  result.append({**old,'mix':mix,'consecutive_eligible':streak})
  decisions.append({'target':old['target'],'eligible':eligible,'previous_mix':old['mix'],'next_mix':mix,'standing_certified':False})
 return result,decisions
