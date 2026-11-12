FROZEN_TEST_SCOPE='PORTABLE'
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

from native_stand.students import motion_posture_design as design


def test_exact_quotas_and_task_identity_keep_source_range():
    rows=design.allocation()
    assert sum(r['slots'] for r in rows)==4096
    assert rows[0]['slots']==1024 and rows[1]['slots']==1024
    assert sum(r['slots'] for r in rows[2:])==2048
    assert max(r['slots'] for r in rows[2:])-min(r['slots'] for r in rows[2:])==1
    assert len({tuple(r['cohort_key']) for r in rows})==23
    commands=design.commands()
    assert [4.,0.] not in commands and [-4.,0.] not in commands and [0.,-4.] not in commands
    assert [0.,4.] in commands  # This direction already has complete source proof.
    assert len(commands)==22
    with pytest.raises(ValueError):design.allocation(64)


def test_nominal_and_stress_zero_cannot_share_retention_eligibility():
    targets=torch.tensor([[0.,0.],[0.,0.],[-3.,0.],[0.,4.],[0.,-4.],[.5,0.]])
    variants=torch.tensor([0,2,0,0,0,0]);valid=torch.tensor([True,True,True,True,True,False])
    ids=design.retention_ids(targets,variants,valid)
    assert ids[0]==0 and ids[1]==-1 and ids[2]>=0 and ids[3]>=0 and ids[4]==-1 and ids[5]==-1


def test_loss_balances_cohorts_and_detaches_teacher():
    teacher=torch.zeros(4,6,requires_grad=True)
    mean=torch.tensor([[1.]*6,[1.]*6,[1.]*6,[2.]*6],requires_grad=True)
    ids=torch.tensor([0,0,0,1]);std=torch.ones(6)
    loss,_,counts=design.balanced_policy_loss(mean,teacher,std,ids,.01)
    assert counts[:2].tolist()==[3,1]
    assert float(loss.detach())==pytest.approx(.0125)  # .01*(.5+2)/2, not weighted by3:1.
    loss.backward();assert teacher.grad is None and (mean.grad>0).all()
    empty,_,_=design.balanced_policy_loss(mean,teacher,std,torch.full((4,),-1),.01)
    assert float(empty.detach())==0


@pytest.mark.parametrize('error',[.004,.015,.032,.1])
def test_dense_leg_cost_has_correct_sign_and_does_not_saturate(error):
    x=torch.tensor(error,requires_grad=True);r=design.posture_terms(x,x*0)
    grad=torch.autograd.grad(r['new_leg_density'],x)[0]
    assert grad<0 and r['new_leg_density']<r['old_leg_density']
    assert torch.allclose(r['density_delta'],r['new_leg_density']-r['old_leg_density'])


def test_roll_cost_is_dense_within_acceptance_band_and_symmetric():
    x=torch.tensor(.034,requires_grad=True);r=design.posture_terms(x*0,x)
    assert r['old_roll_density']==0 and r['new_roll_density']<0
    assert torch.autograd.grad(r['new_roll_density'],x)[0]<0
    assert r['new_roll_density']==design.posture_terms(x*0,-x)['new_roll_density']
    assert r['dt_s']==.01


def test_design_is_not_a_training_recipe_or_new_authorization():
    d=design.declaration()
    assert d['status']=='CPU_DESIGN_NOT_RUNTIME_RECIPE'
    assert d['progression'] is False and d['extra_height_or_speed'] is False
    assert 'authorization' not in d and 'budget' not in d


def test_excluded_stress_samples_cannot_pull_the_teacher_anchor():
    mean=torch.ones(3,6,requires_grad=True);teacher=torch.zeros_like(mean)
    loss,_,_=design.balanced_policy_loss(mean,teacher,torch.ones(6),torch.tensor([0,-1,21]))
    gradient=torch.autograd.grad(loss,mean)[0]
    assert not gradient[1].any() and (gradient[[0,2]]>0).all()


def test_cpu_plan_cli_rejects_training_authorization_before_reading_or_writing(tmp_path):
    root=Path(__file__).resolve().parents[1];output=tmp_path/'design'
    env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
    env.pop('HAPI_SESSION_ID',None)
    result=subprocess.run([sys.executable,str(root/'scripts/prepare_motion_mcu.py'),
        '--posture-plan','--parent-spec',str(tmp_path/'absent_spec.json'),
        '--checkpoint',str(tmp_path/'absent_checkpoint.pt'),
        '--measured-assessment',str(tmp_path/'absent_assessment.json'),
        '--write','--design-output-dir',str(output),'--authorize'],
        env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==2 and 'CPU design only' in result.stderr
    assert not output.exists()


def test_runtime_stage_preserves_duplicate_zero_and_partial_reset_identity():
    from native_stand.command_v6.domain.command_sampling import resolve
    for n in (1,64,4096):
        stage=design.stage_config(0,n);resolve(stage,n)
        assert len(stage['commands'])==23 and sum(stage['slot_counts'])==n
        slots=torch.tensor(stage['sampling_slots']);variants=torch.tensor(stage['mcu_variants'])[slots]
        selected=torch.tensor([0,n-1])
        assert torch.equal(variants[selected],torch.tensor(stage['mcu_variants'])[slots[selected]])
        if n>1:assert int((variants==2).sum())==n//4


def test_dense_runtime_reward_replaces_only_two_costs_and_preserves_event():
    fields=('leg_balance','unrelated','roll_excess');raw=torch.zeros(3,3)
    density=raw.clone();dt=raw.clone();events=torch.tensor([0.,-1.,0.])
    result=dict(fields=fields,raw=raw,density=density,dt=dt,total=events.clone(),eligible=torch.tensor([True,True,False]))
    posture=torch.zeros(3,10);posture[:,1]=.034;posture[:,2]=.015
    design.apply_runtime_reward(result,posture)
    assert result['raw'][0,0]>0 and result['raw'][0,2]>0 and not result['raw'][2].any()
    torch.testing.assert_close(result['total'],result['dt'].sum(-1)+events)
    assert not result['raw'][:,1].any() and result['posture_extension_version']==design.RUNTIME_REWARD_VERSION
    torch.testing.assert_close(result['dt'],result['density']*.01)


def test_new_recipe_enables_only_nominal_anchor_and_budgeted_warmup():
    from native_stand.students.motion_posture_preparation import recipe
    from native_stand.students.training_recipe import validate_instance
    from native_stand.students.motion_source_retention import nominal_declaration,add_metadata,KEY
    from native_stand.recovery.controlled_ppo import KEY as VALID
    from tensordict import TensorDict
    value=recipe(design.SOURCE_SHA)
    assert validate_instance(value,design.SOURCE_SHA)==value and value['warmup_updates_per_phase']==20
    assert value['checkpoint_every_updates']==100 and value['milestones'][-1]==9500
    obs=TensorDict({VALID:torch.ones(4,1,dtype=torch.bool)},batch_size=[4])
    target=torch.tensor([[0.,0.],[0.,0.],[-3.,0.],[0.,4.]])
    with pytest.raises(ValueError,match='variant'):add_metadata(obs,target,nominal_declaration(design.SOURCE_SHA))
    out=add_metadata(obs,target,nominal_declaration(design.SOURCE_SHA),variants=torch.tensor([0,2,0,0]))
    assert out[KEY].squeeze().tolist()==[0,-1,10,16]


def test_fixed_course_full_success_never_unlocks_another_speed_or_stops():
    template=dict(kind='student',role='MotionZeroStudent',checkpoint='/source.pt',sha256='a'*64,
        family='motion_native_normal',seed=1,yaw_rad=0.,seconds=31.52)
    spec={'student_evaluation':{'cases':[template]}}
    rows=[dict(case=r,per_env=[dict(env=i,accepted=True,controlled_seconds=30.) for i in range(8)]) for r in design.cases(spec)]
    state=dict(stage=0,last_evaluation_update=-1,consecutive_passes=0,accepted_by_stage={})
    candidate=dict(stage=0,update=500,checkpoint='/500.pt',sha256='b'*64,complete_matrix=True,evaluations=rows)
    new,decision=design.advance(state,candidate)
    assert decision['evidence_valid'] and decision['final_confirmed'] and not decision['stop'] and not decision['advanced']
    assert new['stage']==0 and len(design.refinement_records(rows,0))==23
    rows[0]['case']['_mcu_variant']=2
    assert not design.advance(state,candidate)[1]['evidence_valid']


def test_real_cpu_ppo_update_with22_retained_cohorts_and_stress_exclusion():
    from tensordict import TensorDict
    from rsl_rl.storage import RolloutStorage
    from native_stand.recovery.controlled_ppo import KEY as VALID
    from native_stand.students.model import Student
    from native_stand.students.motion_source_retention import ppo_class,SourcePolicyRetention,nominal_declaration,KEY
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        actor=Student(True,version='multiskill_mcu_v2');actor.log_std.requires_grad_(False)
        critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version='motion_privileged49_v1')
        obs=TensorDict({'student_actor':torch.randn(24,31)*.02,'student_critic':torch.randn(24,49)*.02,
            VALID:torch.ones(24,1,dtype=torch.bool),KEY:torch.tensor([*range(22),-1,-1])[:,None]},batch_size=[24])
        storage=RolloutStorage('rl',24,2,obs,[6],device='cpu')
        ppo=ppo_class()(actor,critic,storage,learning_rate=3e-5,schedule='fixed',num_learning_epochs=1,
            num_mini_batches=1,entropy_coef=.001,value_loss_coef=.5,use_clipped_value_loss=False,device='cpu')
        ppo.source_retention=SourcePolicyRetention(actor,nominal_declaration(design.SOURCE_SHA))
        for _ in range(2):
            ppo.act(obs);ppo.process_env_step(obs,torch.randn(24)*.05,torch.zeros(24,dtype=torch.bool),{})
        ppo.compute_returns(obs);ppo.update()
        summary=ppo.mask_summary['source_policy_retention']
        assert summary['eligible_rollout_transitions']==44 and len(summary['selected_by_cohort_over_epochs'])==22
        assert all(not p.requires_grad and p.grad is None for p in ppo.source_retention.teacher.parameters())
        assert all(torch.isfinite(p).all() for p in actor.parameters())
