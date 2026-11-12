"""Bounded real shared-environment wiring readback, zero optimizer updates."""
import argparse
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    s=json.loads(a.spec.read_text());started=time.monotonic()
    import sys
    sys.path.insert(0,s['asset_module_root'])
    from isaaclab.app import AppLauncher
    app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
    import torch
    from native_stand.students.evaluator import environment
    from native_stand.students.model import load
    from native_stand.students import motion_height_pd as course
    from native_stand.students.motion_runner import motion_observations
    from native_stand.students.motion_source_retention import KEY
    from native_stand.command_v6.completion import write
    actor,_=load(s['source_checkpoint'],'MotionZeroStudent');actor=actor.to('cuda:0');env=None
    report=dict(status='INCOMPLETE',PPO_updates=0,optimizer_steps=0,stages=[])
    try:
        for high in (False,True):
            state=course.initial_state()
            if high:state.update(stage=1,pd_level=2,height_level=2)
            current=course.effective_spec(s,state['stage'],state)
            env=environment(current,'MotionZeroStudent',envs=64)
            env.curriculum_training=True;env.select_initial_family('motion_native_normal','train');obs,_=env.reset()
            h=env.height_adapter.target;expected=torch.tensor(env.stage_spec['height_targets_m'],device=env.device)[env.command_index]
            assert torch.equal(h,expected)
            assert torch.allclose(env.root_height,h,atol=2e-4)
            assert torch.allclose(obs['student_actor'][:,24],h/.5,atol=1e-6)
            assert obs['student_critic'].shape==(64,312)
            assert torch.allclose(obs['student_critic'][:,31],env.root_height-h,atol=2e-4)
            assert torch.allclose(obs['student_critic'][:,140:143],(h/.5)[:,None].expand(-1,3),atol=1e-6)
            assert torch.allclose(env.executed_leg,env.robot.data.joint_pos.torch[:,env.ids][:,[0,1,3,4]],atol=1e-5)
            variants=env.motion_timing.variant
            metadata=motion_observations(env,obs,s['student_training']['source_policy_retention'])
            assert (metadata[KEY][variants!=0]==-1).all()
            assert (metadata[KEY][variants==0]>=0).all()
            assert (env.motion_timing.actual_armature-.012).abs().max()<1e-6
            steps=220 if not high else 30
            with torch.inference_mode():
                for _ in range(steps):
                    obs,reward,*_=env.step(actor(obs))
                    assert obs['student_actor'].shape==(64,31) and obs['student_critic'].shape==(64,312)
                    assert torch.isfinite(reward).all() and torch.isfinite(obs['student_critic']).all()
                    assert not env.recovery_snapshot[:,4].bool().any()
            rr=env.student_reward_result;actual=env.recovery_snapshot[:,6]
            height_raw=1/(1+((actual-env.motion_action_command[:,2])/.04).square())
            assert torch.allclose(rr['raw'][:,rr['fields'].index('height')],height_raw*rr['eligible'],atol=1e-6)
            assert torch.allclose(rr['dt'],rr['density']*.01,atol=1e-6)
            if not high:assert int(env.robust_handoffs)>0
            chosen=torch.tensor([0,20,63],device=env.device)
            env._reset_idx(chosen);reset_obs=env._get_observations()
            assert torch.allclose(reset_obs['student_actor'][chosen,24],env.height_adapter.target[chosen]/.5)
            assert torch.allclose(env.motion_timing.actual[chosen],torch.zeros(3,6,device=env.device),atol=1e-5)
            report['stages'].append(dict(stage=state['stage'],heights_m=sorted(set(expected.cpu().tolist())),
                PD_max_ms=env.stage_spec['PD_feedback_max_ms'],steps=steps,
                actor_critic=[31,312],height_reward_readback=True,previous_and_partial_reset=True,
                retention_excludes_new_tasks=True,ownership=env.robustness_summary()))
            env.close();env=None
        report.update(status='PASSED',wall_s=time.monotonic()-started)
    except BaseException as error:
        report.update(status='FAILED',failure=repr(error),wall_s=time.monotonic()-started);raise
    finally:
        write(a.output_dir/'readback.json',report)
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
