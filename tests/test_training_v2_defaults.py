FROZEN_TEST_SCOPE='PORTABLE'
import copy,ast
from pathlib import Path
import pytest
import torch
from native_stand.students.performance import (new_motion_defaults,new_recovery_defaults,settings,recovery_training_v2,recovery_step_flags,fixed_masked_counts,parent_motion_performance_matches)


def test_motion_new_defaults_and_parent_upgrade_are_explicit():
 old={'version':'student_performance_v1','evaluation_gpu_buffer':True}
 before=copy.deepcopy(old);new=new_motion_defaults(old)
 assert old==before and new['combined_step_checks'] and new['fixed_shape_statistics']
 assert parent_motion_performance_matches(new,old)
 assert parent_motion_performance_matches(old,old)
 changed=copy.deepcopy(new);changed['trace_buffer_max_bytes']*=2
 assert not parent_motion_performance_matches(changed,old)
 assert settings({'performance':old})['version']=='student_performance_v1'


@pytest.mark.parametrize('version',['student_recovery_broad_narrow_ppo_v1','student_recovery_stop_ppo_v2'])
def test_recovery_safe_v2_enables_trainer_only(version):
 s={'student_training':{'version':version},'performance':new_recovery_defaults()}
 p=settings(s);assert recovery_training_v2(s)
 assert p['training_gpu_statistics'] and p['evaluation_gpu_buffer'] and p['failure_tail_on_device']
 assert not p['combined_step_checks'] and not p['fixed_shape_statistics'] and not p['evaluation_group_commands'] and not p['evaluation_reuse_environment']
 wrong=copy.deepcopy(s);wrong['performance']['combined_step_checks']=True
 with pytest.raises(ValueError):settings(wrong)
 old=copy.deepcopy(s);old['performance']['version']='student_performance_v1'
 for k in ('combined_step_checks','fixed_shape_statistics','evaluation_group_commands','evaluation_group_max_commands'):old['performance'].pop(k)
 assert not recovery_training_v2(old)


@pytest.mark.parametrize('fault',['none','runtime','initial','nan','owner','timeout'])
def test_recovery_flags_match_original_predicates_and_preserve_rng(fault):
 n=64;s=torch.zeros(n,39);r=torch.ones(n);valid=torch.ones(n,dtype=torch.bool);owner=valid.clone();truncated=torch.zeros(n,dtype=torch.bool)
 if fault=='runtime':s[63,4]=1
 if fault=='initial':s[51,30]=1
 if fault=='nan':r[50]=float('nan')
 if fault=='owner':owner[49]=False
 if fault=='timeout':truncated[48]=True
 expected=[not bool(truncated.any()),not bool(s[:,4].bool().any()),not bool(s[:,30].bool().any()),bool(torch.isfinite(r).all()),torch.equal(valid,owner)]
 rng=torch.get_rng_state().clone();actual=recovery_step_flags(s,r,valid,owner,truncated).tolist()
 assert actual==expected and torch.equal(rng,torch.get_rng_state())


def test_fixed_counts_equal_dynamic_with_empty_cohorts():
 ids=torch.tensor([0,3,1,3,0]);mask=torch.tensor([True,False,True,True,False])
 assert torch.equal(fixed_masked_counts(ids,mask,4),torch.bincount(ids[mask],minlength=4))
 assert fixed_masked_counts(ids,torch.zeros_like(mask),4).eq(0).all()


def test_current_training_preparers_freeze_v2_defaults():
 root=Path(__file__).resolve().parents[1]
 for filename in ('prepare_motion31_training.py','prepare_motion31_large_combo.py','prepare_motion31_course.py','prepare_motion31_forward.py','prepare_motion31_yaw_retention.py','prepare_motion31_yaw_width_ab.py'):
  source=(root/'scripts'/filename).read_text();ast.parse(source);assert "s['performance']=new_motion_defaults(s.get('performance'))" in source
 for filename in ('prepare_recovery_broad_narrow.py','prepare_recovery_stop.py'):
  source=(root/'scripts'/filename).read_text();ast.parse(source);assert "s['performance']=new_recovery_defaults(s.get('performance'))" in source
