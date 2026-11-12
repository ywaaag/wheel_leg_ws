"""Offline ONNX sensitivity on recorded states, no simulator or weight updates.

Missing actual executed targets remain UNKNOWN. Proxy and recursive shadow cases
are counterfactuals, not corrected deployment inputs or closed-loop evidence.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from onnx.reference import ReferenceEvaluator
from replay_motion31_shadow_csv import read_csv,sha

SCALE=np.array([1.5,1.5,100.,1.5,1.5,100.]);LEGS=[0,1,3,4];WHEELS=[2,5]


def delayed_columns(x,t,cols,ms,method):
    out=x.copy();query=t-ms/1000.
    if method=='past_sample':
        ids=np.maximum(0,np.searchsorted(t,query,side='right')-1);out[:,cols]=x[ids][:,cols]
        age=t-t[ids]
    elif method=='linear_estimate':
        for c in cols:out[:,c]=np.interp(query,t,x[:,c])
        age=t-np.maximum(t[0],query)
        if cols==list(range(3,6)) or cols==list(range(6)):
            g=out[:,3:6];norm=np.linalg.norm(g,axis=1,keepdims=True)
            out[:,3:6]=np.divide(g,norm,out=g.copy(),where=norm>1e-8)
    else:raise ValueError('unknown reconstruction method')
    return out,age


def output_metrics(y,base,t):
    physical=y*SCALE;bp=base*SCALE;delta=np.diff(physical,axis=0)
    legdelta=delta[:,LEGS]
    opposite=(delta[1:]*delta[:-1]<0)&(abs(delta[1:])>1e-4)&(abs(delta[:-1])>1e-4)
    return {'samples':len(t),'physical_mean':physical.mean(0).tolist(),
        'physical_MAE_vs_logged_input_replay':abs(physical-bp).mean(0).tolist(),
        'physical_step_abs_p95':np.percentile(abs(delta),95,axis=0).tolist(),
        'physical_step_abs_max':abs(delta).max(0).tolist(),
        'step_direction_reversal_fraction':opposite.mean(0).tolist(),
        'leg_steps_over_point1_fraction':float((abs(legdelta)>.1).any(1).mean()),
        'interpretation':'output roughness on frozen recorded states; lower is NOT certified better feedback or stability'}


def replay_variants(x,t,infer):
    base=infer(x);yield 'logged_previous','as_logged',x,base,{}
    groups={'dq':list(range(10,16)),'gyro':list(range(3)),'gravity':list(range(3,6)),'imu':list(range(6))}
    for group,cols in groups.items():
        for ms in (5,10,20):
            for method in ('past_sample','linear_estimate'):
                v,age=delayed_columns(x,t,cols,ms,method)
                yield f'{group}_{ms}ms_{method}',method,v,infer(v),{'requested_delay_ms':ms,'effective_age_ms':{'min':float(age.min()*1000),'median':float(np.median(age)*1000),'max':float(age.max()*1000)},'reconstruction':'past stored sample only' if method=='past_sample' else 'interpolated historical estimate, not measured5ms sample'}
    v=x.copy();v[:,20:22]=x[:,[12,15]]*.2
    yield 'measured_wheels_proxy','counterfactual',v,infer(v),{'warning':'actual velocity is not actual executed speed target; proxy only'}
    v=x.copy();v[:,16:20]=x[:,6:10];v[:,20:22]=x[:,[12,15]]*.2
    yield 'measured_q_wheels_proxy','counterfactual',v,infer(v),{'warning':'actual q/dq are not actual executed targets; not validated handover initialization'}
    v=x.copy();v[:,16:22]=0.
    yield 'zero_previous','counterfactual',v,infer(v),{'warning':'zero targets may be physically inconsistent; not a startup recommendation'}
    v=x.copy();y=np.empty((len(x),6),np.float32)
    for i in range(len(v)):
        if i:v[i,16:22]=y[i-1,[0,1,3,4,2,5]]
        y[i]=infer(v[i:i+1])[0]
    yield 'recursive_shadow_previous','recursive_offline',v,y,{'warning':'predictions recursively feed previous, while real state sequence stays frozen; no physical controller or closed loop; seed is first recorded previous'}
    # Deterministic perturbations do not consume training RNG or inject simulator noise.
    for amplitude in (.1,.5):
        v=x.copy();sign=np.where(np.arange(len(x))%2==0,1.,-1.)
        v[:,10:16]+=sign[:,None]*amplitude*.05
        yield f'dq_alternating_{amplitude}rad_s','synthetic_noise',v,infer(v),{'physical_amplitude_rad_s':amplitude,'warning':'common signed alternating synthetic dq error, sensitivity only; not measured sensor noise'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--onnx',type=Path,default=Path('deployment_protocol/2026_10_19-19_57__motion3250_small31_dagger2_onnx/actor.onnx'))
    p.add_argument('--output-dir',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args()
    _,x,logged,t,columns=read_csv(a.csv)
    spec={'version':'motion31_log_sensitivity_v1','csv':str(a.csv.resolve()),'csv_sha256':sha(a.csv),'onnx':str(a.onnx.resolve()),'onnx_sha256':sha(a.onnx),
        'windows':{'pre_oscillation':[12.,17.06],'oscillation':[17.06,17.89]},'missing_observations_assumed':{str(i):float(x[0,i]) for i in range(25,31) if i not in columns['observation']},
        'actual_executed_target_comparison':'UNKNOWN: no actual target/owner/driver acknowledgement in CSV',
        'simulation_steps':0,'PPO_updates':0,'scope':'offline recorded-state response, no physical or dynamical root-cause identification'}
    if not a.run:print(json.dumps(spec,indent=2));return
    if a.output_dir is None:
        from native_stand.command_v6.domain.experiment_naming import new_experiment_name
        a.output_dir=Path('artifacts')/new_experiment_name('motion3250_log_sensitivity')
    out=a.output_dir;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    model=ReferenceEvaluator(str(a.onnx));infer=lambda v:model.run(None,{model.input_names[0]:np.asarray(v,np.float32)})[0]
    results={};traces={}
    for label,(start,end) in spec['windows'].items():
        # Include real pre-window samples so the first scored delayed inputs
        # are not artificially clamped to the window's first measurement.
        mask=(t>=start-.03)&(t<end);xx=x[mask];tt=t[mask];scored=tt>=start
        if len(tt)<3:raise ValueError('insufficient recorded window')
        base=infer(xx);err=abs(base-logged[mask]).max()
        if err>1e-5:raise ValueError('ONNX/log mismatch; confirm model/task fields before sensitivity')
        rows=[]
        for name,method,inputs,y,meta in replay_variants(xx,tt,infer):
            rows.append({'variant':name,'method':method,**meta,**output_metrics(y[scored],base[scored],tt[scored])})
            traces[f'{label}__{name}']=y[scored]
        results[label]={'time_s':[float(tt[scored][0]),float(tt[-1])],'history_start_s':float(tt[0]),'max_baseline_replay_error':float(err),'cases':rows}
        traces[label+'__time']=tt[scored]
    result={**spec,'status':'COMPLETED','wall_s':time.monotonic()-started,'results':results}
    with (out/'analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    np.savez_compressed(out/'outputs.npz',**traces)
    lines=['# Motion3250实机日志离线敏感性','',
        '不推进物理，不修改模型，不更新PPO。每种输入变体都使用同一段已录制的真实状态；不能据此预测闭环是否摔倒。',
        '', '实际执行目标没有记录，因此actual-target previous对照为UNKNOWN；logged previous不自动视为真实执行历史。实际q/dq、清零及递归shadow均为显式反事实，不是修正输入或部署建议。',
        '', '日志约100Hz：5ms不能由日志直接观测。past_sample严格取不晚于t-delay的最近记录，实际年龄可能约10/20ms；linear_estimate使用历史插值估计，不伪称真实5ms采样。gyro与gravity另列和联合延迟，不改变31D尺度。每个窗口包含约30ms真实前置历史，递归shadow以此前置历史第一帧previous为种子；前置帧不参与粗糙度评分。',
        '', '腿目标跳变指本帧目标减上一目标，非真实关节位移。输出更平滑可能只是引入相位滞后或减弱反馈，不能当作控制改进。']
    for label,data in results.items():
        lines+=['',f'## {label}', '', '|变体|腿目标逐帧最大跳变rad|腿目标逐帧P95最大rad|跳变>.1rad帧比例|', '|---|---:|---:|---:|']
        for row in data['cases']:
            lines.append(f"|{row['variant']}|{max(np.array(row['physical_step_abs_max'])[LEGS]):.4f}|{max(np.array(row['physical_step_abs_p95'])[LEGS]):.4f}|{row['leg_steps_over_point1_fraction']:.3f}|")
    lines+=['',f'原始CSV SHA：{spec["csv_sha256"]}；ONNX SHA：{spec["onnx_sha256"]}。实际命令、单位、各通道MAE和延迟年龄见analysis.json。']
    with (out/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(13,9))
    variants=['logged_previous','dq_10ms_past_sample','dq_20ms_past_sample','imu_10ms_past_sample','recursive_shadow_previous']
    for ax,label in zip(axes,spec['windows']):
        tt=traces[label+'__time']
        for name in variants:ax.plot(tt,traces[label+'__'+name][:,0]*1.5,label=name,alpha=.8)
        ax.set_title(label);ax.set_ylabel('ch0 requested position (rad)');ax.grid(alpha=.3);ax.legend(fontsize=8,ncol=2)
    axes[-1].set_xlabel('recorded time (s)');fig.suptitle('Offline input sensitivity on frozen real states; NOT closed-loop simulation');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    assert sha(a.csv)==spec['csv_sha256'] and sha(a.onnx)==spec['onnx_sha256']
    print(json.dumps({'output':str(out.resolve()),'status':'COMPLETED','cases_per_window':len(rows),'wall_s':result['wall_s'],'simulation_steps':0,'PPO_updates':0}))
if __name__=='__main__':main()
