"""Browser controls for stored trajectories; no movement commands or physics."""
import math
import threading

class ReplayPlan:
    def __init__(self, names, model, speed, paused):
        self.names=list(names)
        if model not in self.names:raise ValueError('unknown clip')
        self.lock=threading.Lock()
        self.state=dict(model=model,playback_speed=speed,paused=paused)
        self.restart=False
        self.telemetry={}

    def apply(self,payload):
        if not isinstance(payload,dict) or set(payload)-{'model','paused','reset','playback_speed'}:
            raise ValueError('replay accepts only model, paused, reset, playback_speed; no movement commands')
        update=dict(payload)
        if 'model' in update and update['model'] not in self.names:raise ValueError('unknown clip')
        for k in ('paused','reset'):
            if k in update and type(update[k]) is not bool:raise ValueError(k+' must be boolean')
        if 'playback_speed' in update:
            v=update['playback_speed']
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not .1<=v<=2:
                raise ValueError('playback_speed must be .1..2')
        with self.lock:
            self.restart |= update.get('reset',False) or update.get('model',self.state['model'])!=self.state['model']
            self.state.update({k:v for k,v in update.items() if k!='reset'})
        return update

    def take(self):
        with self.lock:
            result=dict(self.state,restart=self.restart);self.restart=False
            return result

    def publish(self,telemetry):
        with self.lock:self.telemetry=dict(telemetry)

    def snapshot(self):
        with self.lock:return dict(self.state,models=self.names,mode='Replay',backend='MuJoCo',pause_supported=True,telemetry=dict(self.telemetry))


class ReplayTiming:
    """Measure displayed trajectory seconds per wall second, excluding rewind jumps."""
    def __init__(self, wall):
        self.start=self.window_start=wall
        self.previous=None
        self.total=self.window_total=0.
        self.recent=0.

    def sample(self, model, clip_time, wall, restart=False):
        if self.previous is not None:
            old_model,old_time=self.previous
            if old_model==model and not restart and clip_time>=old_time:
                self.total+=clip_time-old_time
        self.previous=(model,clip_time)
        if wall-self.window_start>=.5:
            self.recent=(self.total-self.window_total)/(wall-self.window_start)
            self.window_start,self.window_total=wall,self.total
        return dict(realtime_factor=self.recent,realtime_factor_total=self.total/max(wall-self.start,1e-9))


def frame_telemetry(clip, frame):
    """Same named telemetry as live Motion; saved pre-action input / post-state explicit.

    Replay has an extra initial state at index zero. Row i describes input/action
    for the interval ending at replay frame i+1; never shift to the next label.
    """
    result=dict(t=float(clip['t'][frame]),steps=frame,task_mode='Replay',
                source=('CSV动作开环' if clip['name']=='recorded_actions_open_loop' else '保存的仿真闭环')+'；非实机姿态重建',
                sample_id=clip['name']+':'+str(frame),
                sample_semantics='pre-action input/mean; post-physics state/held torque')
    if frame==0:return result
    arrays=clip.get('telemetry',{});i=frame-1
    for name in ('mean','actor_input','requested','adopted','q','dq','torque','preclip','contact','state'):
        if name in arrays:result[name]=arrays[name][i].tolist()
    if 'state' in result:
        for k,v in zip(('vx','yaw','height','pitch','roll','closure_m'),result['state']):result[k]=v
    if 'actor_input' in result:
        a=result['actor_input']
        result.update(cmd_vx=a[22]*4,cmd_yaw=a[23]*4,cmd_height=a[24]*.5,actor_task_tail=a[25:31])
    for source,prefix in (('mean','act_'),):
        for j,v in enumerate(result.get(source,[])):result[prefix+str(j)]=v
    for k,j in (('l',2),('r',5)):
        if 'dq' in result:result['wheel_qdot_'+k]=result['dq'][j]
        if 'torque' in result:result['wheel_tau_'+k]=result['torque'][j]
        if 'contact' in result:result['contact_'+k]=result['contact'][0 if j==2 else 1]
        if 'adopted' in result:result['wheel_target_'+k]=result['adopted'][j]
    for n,j in enumerate((0,1,3,4)):
        if 'torque' in result:result['leg_tau_'+str(n)]=result['torque'][j]
    return result


def kinematic_telemetry(model, data, refs):
    """Reconstruct geometry from saved full pose; never re-solve saved contact/torque."""
    import numpy as np
    import torch
    import mujoco
    from native_stand.command_v6.domain.posture_geometry import measure
    root=refs['root'];rotation=data.xmat[root].reshape(3,3)
    hips=data.xanchor[np.asarray(refs['joints'])[[0,3]]]
    wheels=data.xpos[refs['wheels']]
    shape=measure(torch.from_numpy(hips.copy())[None],torch.from_numpy(wheels.copy())[None],
                  torch.from_numpy(rotation.copy())[None])
    vectors=(wheels-hips)@rotation
    swing=np.arctan2(-vectors[:,2],-vectors[:,1])
    difference=abs(np.arctan2(np.sin(swing[0]-swing[1]),np.cos(swing[0]-swing[1])))
    velocity=np.zeros(6)
    mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,root,velocity,0)
    return dict(swing_diff_rad=float(difference),swing_mean_rad=float(swing.mean()),
                leg_length_diff_m=float(shape['leg_length_difference_m'][0]),
                wheel_fore_aft_m=float(shape['wheel_heading_offset_m'][0]),vz=float(velocity[5]),
                geometry_source='saved full pose; shared posture_geometry.measure; no physics step')
