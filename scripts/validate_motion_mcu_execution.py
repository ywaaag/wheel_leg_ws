"""Bounded authorized Isaac工程读回; no PPO/optimizer update or new training owner."""
import argparse
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False);s=json.loads(a.spec.read_text());started=time.monotonic()
    import sys
    sys.path.insert(0,s['asset_module_root'])
    from isaaclab.app import AppLauncher
    app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
    import torch
    from native_stand.students.evaluator import environment
    from native_stand.students.model import load
    from native_stand.students.motion_mcu_course import effective_spec
    from native_stand.control_readback import check_control
    actor,_=load(s['source_checkpoint'],'MotionZeroStudent');actor=actor.to('cuda:0');env=None
    result={'status':'INCOMPLETE','simulation':'explicit MCU engineering diagnostic','PPO_updates':0,'optimizer_steps':0,'stages':[]}
    try:
        stages=[s['motion_training']['stage']] if s.get('mcu_measured') or s.get('mcu_posture') else (0,2,3)
        if s.get('mcu_posture'):
            from native_stand.students.motion_posture_design import effective_spec
        elif s.get('mcu_measured'):
            from native_stand.students.motion_mcu_measured import effective_spec
        critic_dim=s['observation']['critic_dim']
        for stage in stages:
            current=effective_spec(s,stage);env=environment(current,'MotionZeroStudent',envs=64)
            env.curriculum_training=True;env.select_initial_family('motion_native_normal','train');obs,_=env.reset()
            assert obs['student_actor'].shape==(64,31) and obs['student_critic'].shape==(64,critic_dim)
            assert torch.allclose(obs['student_actor'][:,24],torch.full((64,),.68,device=env.device))
            posture_readback={}
            if s.get('mcu_posture'):
                from native_stand.students.motion_runner import motion_observations
                from native_stand.students.motion_source_retention import KEY
                expected=torch.tensor(env.stage_spec['slot_counts'],device=env.device)
                assert torch.equal(torch.bincount(env.command_index,minlength=23),expected)
                variants=torch.tensor(env.stage_spec['mcu_variants'],device=env.device)[env.command_index]
                assert torch.equal(env.motion_timing.variant,variants)
                metadata=motion_observations(env,obs,s['student_training']['source_policy_retention'])
                assert (metadata[KEY][variants==2]==-1).all() and (metadata[KEY][variants==0]>=0).all()
                posture_readback=dict(cohort_counts=expected.cpu().tolist(),stress_excluded_from_teacher=True)
            control=check_control(env)
            if not torch.allclose(env.executed_leg,env.robot.data.joint_pos.torch[:,env.ids][:,[0,1,3,4]],atol=1e-5):
                raise RuntimeError('normal reset INIT_PROXY disagrees with actual leg q')
            with torch.inference_mode():
                for step in range(220 if stage==0 else 30):
                    obs,reward,*_=env.step(actor(obs))
                    if not torch.isfinite(obs['student_critic']).all() or not torch.isfinite(reward).all():
                        raise RuntimeError('MCU runtime nonfinite input/reward')
                    if env.recovery_snapshot[:,4].bool().any():raise RuntimeError('MCU runtime fault')
            t=env.motion_timing
            if s.get('mcu_posture'):
                from native_stand.students.motion_posture_design import huber,RUNTIME_REWARD_VERSION
                rr=env.student_reward_result;pst=env.standing_snapshot;mask=rr['eligible']
                assert rr['posture_extension_version']==RUNTIME_REWARD_VERSION
                assert torch.allclose(rr['raw'][:,rr['fields'].index('leg_balance')],huber((pst[:,2]-pst[:,3])/.008)*mask)
                assert torch.allclose(rr['raw'][:,rr['fields'].index('roll_excess')],huber(pst[:,1]/.05)*mask)
                assert torch.allclose(rr['dt'],rr['density']*.01)
                posture_readback.update(reward_extension=RUNTIME_REWARD_VERSION,reward_sign_and_dt=True)
            if not torch.allclose(t.actual,t.motor_actual+t.spring_actual,atol=1e-5):raise RuntimeError('independent spring/net torque mismatch')
            if not torch.all(t.driver.abs()<=t.driver.new_tensor(current['mcu_execution']['profile']['driver_limits_nm'])+1e-5):raise RuntimeError('driver limit did not execute')
            result['stages'].append({'stage':stage,'steps':220 if stage==0 else 30,'actor_critic':[31,critic_dim],
                'timing':t.summary(),'control':control,'previous_scope':'local adopted physical target',
                'posture_readback':posture_readback,'ability_certified':False})
            # The next explicit reset must safely consume buffers written inside inference_mode.
            env._reset_idx(torch.tensor([1,3],device=env.device));env._get_observations()
            env.close();env=None
        result['status']='PASSED'
    except BaseException as exc:
        import traceback
        result.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if env is not None:env.close()
        result['wall_s']=time.monotonic()-started
        (a.output_dir/'readback.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':result['status'],'wall_s':result['wall_s'],'stages':len(result['stages'])}),flush=True)
        app.close()


if __name__=='__main__':main()
