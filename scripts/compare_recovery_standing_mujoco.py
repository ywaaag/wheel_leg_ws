"""Normal standing cross-policy diagnostic using the shared Recovery MuJoCo loop."""
import json,time
from pathlib import Path
import numpy as np
import torch
from native_stand.students.data import sha

class LegacyMotion:
    """Reconstruct native25 from current physical31, with independent native history."""
    def __init__(self,actor,command_height):
        self.actor=actor;self.height=command_height;self.hist=torch.zeros(1,125);self.first=True
    def mean(self,x):
        from tensordict import TensorDict
        from native_stand.command_v6.domain.contract import frame,history
        q=x.new_zeros(1,6);q[:,[0,1,3,4]]=x[:,6:10]*1.5
        prev=x.new_zeros(1,6);prev[:,[0,1,3,4]]=x[:,16:20]*1.5/.5;prev[:,[2,5]]=x[:,20:22]
        f=frame(x[:,:3]/.25,x[:,3:6],x.new_tensor([[0.,0.,self.height]]),q,x[:,10:16]/.05,prev.clamp(-1,1),torch.zeros_like(q))
        self.hist=history(self.hist,f,torch.tensor([self.first]));self.first=False
        return self.actor(TensorDict({'policy':f,'history':self.hist},batch_size=[1]),stochastic_output=False)

def motion_targets(raw,reference,limits):
    from native_stand.command_v6.domain.contract import targets
    clipped,legs,wheels=targets(raw,reference,leg_scale=.5,combined_radius=0.,wheel_scale=100.)
    return dict(leg_requested=legs,leg_executed=legs.clamp(limits[:,[0,1,3,4],0],limits[:,[0,1,3,4],1]),wheel_executed=wheels)

def execute(args):
    from validate_recovery31_mujoco import plan,run_case
    from validate_native_mujoco import state
    from native_stand.command_v6.learning import model as network
    from native_stand.command_v6.completion import write
    from native_stand.command_v6.tools.artifact_io import npz
    from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
    from tensordict import TensorDict
    if args.viewer:raise ValueError('standing comparison is a headless bounded two-episode diagnostic')
    if args.motion_spec is None or args.motion_checkpoint is None:raise ValueError('explicit frozen Motion spec/checkpoint required')
    torch.set_num_threads(1);start=time.monotonic()
    spec,model,refs,student,pose,xml,manifest=plan(args)
    ms=json.loads(args.motion_spec.read_text())
    expected='5d3389d5b8e33b7915d12d21308702f3105248485741b27ec91d0aaf4b90b89e'
    if sha(args.motion_checkpoint)!=expected:raise ValueError('fixed B250 SHA differs')
    if ms['action']['leg_scale_rad']!=.5 or ms['action']['wheel_scale_rad_s']!=100 or ms['action']['leg_pair_radius_rad']!=0:raise ValueError('B250 action contract differs')
    saved=torch.load(args.motion_checkpoint,map_location='cpu',weights_only=False)
    actor=network(TensorDict({'policy':torch.zeros(1,25),'history':torch.zeros(1,125)},batch_size=[1]),actor=True,spec=ms)
    actor.load_state_dict(saved['actor'],strict=True);actor.eval();actor.requires_grad_(False)
    motion=LegacyMotion(actor,.4)
    data=__import__('mujoco').MjData(model);__import__('mujoco').mj_forward(model,data)
    # Original B250 direct reset: q=dq=0, root z=.4, CAD quaternion xyzw=[sqrt(.5),0,0,sqrt(.5)].
    assert np.allclose(data.qpos[refs['qpos']],0) and np.allclose(data.qvel,0)
    assert np.allclose(data.qpos[:7],[0,0,.4,2**-.5,2**-.5,0,0],atol=1e-7)
    manifest.update(version='normal_standing_cross_policy_mujoco_v1',initial_state=state(model,data,refs)[2],initial_qpos=data.qpos.tolist(),initial_qvel=data.qvel.tolist(),motion_checkpoint=str(args.motion_checkpoint),motion_sha256=expected,motion_spec_sha256=sha(args.motion_spec),commands={'B250':[0,0,.4],'Recovery1500':[0,0,.34]},max_episodes=2,simulation=0)
    if not args.run:print(json.dumps(manifest,indent=2));return
    validate_new_experiment_name(args.output_dir.name);args.output_dir.mkdir(parents=True,exist_ok=False)
    manifest['inputs_sha256'].update({str(args.motion_spec.resolve()):sha(args.motion_spec),str(args.motion_checkpoint.resolve()):expected,str(Path(__file__).resolve()):sha(Path(__file__))})
    write(args.output_dir/'manifest.json',manifest);(args.output_dir/'diagnostic_model.xml').write_text(xml)
    results=[];error=None
    try:
        for name,policy,decoder in [('B250',motion,motion_targets),('Recovery1500',student,None)]:
            if time.monotonic()>start+args.wall_seconds-5:raise TimeoutError('comparison wall budget')
            with torch.inference_mode():result,arrays=run_case(spec,model,refs,policy,pose,start+args.wall_seconds-5,normal_start=True,decode_targets=decoder)
            npz(args.output_dir/(name+'_trajectory.npz'),arrays)
            windows=[]
            for lo,hi in [(0,1),(1,3),(3,5),(5,8),(8,12.5)]:
                mask=(arrays['policy_time_s']>=lo)&(arrays['policy_time_s']<hi)
                if mask.any():windows.append(dict(window_s=[lo,hi],samples=int(mask.sum()),vx_mean=float(arrays['state'][mask,7].mean()),pitch_abs_peak=float(np.abs(arrays['posture'][mask,0]).max()),wheel_target_mean=arrays['executed_wheel_target'][mask].mean(0).tolist(),contact_fraction=(arrays['state'][mask,10:12]>2).mean(0).tolist()))
            item=dict(policy=name,result=result,windows=windows);results.append(item);write(args.output_dir/(name+'_result.json'),item)
            if result['termination'] in ('RUNTIME_FAULT','NONFINITE_ACTION'):raise RuntimeError(result['termination'])
    except BaseException as exc:error=repr(exc);raise
    finally:
        unchanged=all(sha(Path(p))==h for p,h in manifest['inputs_sha256'].items())
        payload=dict(status='COMPLETED' if error is None and unchanged else 'FAILED',error=error,wall_s=time.monotonic()-start,PPO=0,optimizer=0,episodes=len(results),inputs_unchanged=unchanged,scope='single deterministic normal-pose diagnostic; common Recovery-style certificate, B250 command .40 versus student .34; no native capability claim',results=results)
        write(args.output_dir/'terminal_report.json',payload);print(json.dumps(payload,indent=2))
