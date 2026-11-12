"""Read completed real2sim sweep; compare log signals without fitting a root cause."""
import argparse,json,hashlib,importlib.util
from pathlib import Path
import numpy as np


def metrics(time,target,state,dq):
    legs=[0,1,3,4];delta=np.diff(target[:,legs],axis=0)
    peak=float(abs(delta).max()) if len(delta) else None
    onset=np.flatnonzero(abs(delta).max(1)>.1) if len(delta) else []
    return {'leg_step_p95_rad':np.percentile(abs(delta),95,axis=0).tolist() if len(delta) else None,
        'leg_step_max_rad':peak,'first_target_step_over_point1_s':float(time[onset[0]+1]) if len(onset) else None,
        'dq_abs_p95_rad_s':np.percentile(abs(dq[:,legs]),95,axis=0).tolist(),
        'pitch_abs_peak_rad':float(abs(state[:,4]).max()),'roll_abs_peak_rad':float(abs(state[:,5]).max())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--event-time',type=float,default=17.06,help='alignment hypothesis only; hardware owner not logged')
    a=p.parse_args();out=a.run_dir
    terminal=json.loads((out/'terminal_report.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    if terminal['status'] not in ('COMPLETED','STOPPED_BOUNDED'):raise ValueError('review completed diagnostic only')
    root=Path(__file__).resolve().parents[1]
    sp=importlib.util.spec_from_file_location('csv_replay',root/'scripts/replay_motion31_shadow_csv.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
    _,x,action,t,_=r.read_csv(a.csv);mask=(t>=a.event_time)&(t<a.event_time+1.)
    if mask.sum()<3:raise ValueError('missing real comparison window')
    target=action*np.array([1.5,1.5,100,1.5,1.5,100]);g=x[:,3:6]
    realstate=np.zeros((len(t),7));realstate[:,4]=np.arctan2(g[:,2],-g[:,1]);realstate[:,5]=np.arctan2(g[:,0],-g[:,1])
    real=metrics(t[mask]-a.event_time,target[mask],realstate[mask],x[mask,10:16]*20)
    comparisons=[]
    for summary in terminal['results']:
        pth=out/(summary['name']+'.npz')
        if not pth.exists():continue
        with np.load(pth) as z:
            m=z['time']<=1.+1e-8
            if m.sum()<3:continue
            row={'name':summary['name'],'one_second_after_change':metrics(z['time'][m],z['requested_target'][m],z['state'][m],z['joint_dq_pre'][m]),
                'termination':summary['termination'],'duration_s':summary['duration_s'],'both_contact_fraction':summary['both_contact_fraction']}
            comparisons.append(row)
    result={'version':'motion31_real2sim_delay_review_v1','CSV':str(a.csv.resolve()),'CSV_sha256':hashlib.sha256(a.csv.read_bytes()).hexdigest(),
        'event_time_assumption_s':a.event_time,'real_first_second':real,'cases':comparisons,
        'limitation':'Common simulated stable state, no LQR/real initial pose/identified spring residual. Similar oscillation is sufficiency, not hardware cause. No trajectory or capability equivalence.',
        'simulation':0,'PPO':0}
    with (out/'comparison.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    lines=['# Motion3250 real2sim延迟诊断','',
        '固定3250 dagger2小模型；MuJoCo CPU单环境、有界诊断，无Isaac/PPO。每组同样的8秒零命令预运行后，切入指定机制及vx=-.286（zero基线保持0）；比较窗口6秒。',
        '', '真实接管时间未记录，暂以CSV17.06秒对齐。实机初态、LQR、补偿残差未被复现，不能把匹配形态当作真实参数辨识。11/7ms运输与力矩建立时间常数是两个独立假设，不同时叠加J/B/Tc。',
        '', '|条件|诊断秒数|终止原因|接触比例|首秒腿目标跳变最大rad|首秒pitch峰值rad|首秒roll峰值rad|', '|---|---:|---|---:|---:|---:|---:|']
    for row in comparisons:
        v=row['one_second_after_change'];lines.append(f"|{row['name']}|{row['duration_s']:.2f}|{row['termination'] or '无'}|{row['both_contact_fraction']:.3f}|{v['leg_step_max_rad']:.4f}|{v['pitch_abs_peak_rad']:.4f}|{v['roll_abs_peak_rad']:.4f}|")
    lines+=['',f"实机对齐后首秒：腿目标最大逐帧跳变{real['leg_step_max_rad']:.4f}rad，pitch峰值{real['pitch_abs_peak_rad']:.4f}rad，roll峰值{real['roll_abs_peak_rad']:.4f}rad。",'',
        'requested_target为策略请求，delivered_target为经过1ms运输队列送达值，PD_adopted_target为5ms PD边界采用值；默认previous来自上一PD采用值。state/contact是100Hz物理后采样；actor_input、joint_q_pre/dq_pre来自该策略帧物理前。',
        '', 'shadow_cache_at_handover仅在切入首帧注入实机16.998秒的shadow目标，之后恢复实际PD目标缓存；它不是完整LQR重演。measured_wheels_previous是持续错误语义对照，不是部署建议。',
        '', f"模型未改变：{terminal['model_unchanged']}；总墙钟{terminal['wall_s']:.2f}秒；样本及原始输入SHA见manifest.json。"]
    with (out/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,11),sharex=True)
    selected=['baseline_reverse','leg_transport_10','leg_transport_20','leg_transport_40','leg_transport_60','leg_transport_80','dq_observation_10','dq_observation_20','wheel_transport_11_7']
    for name in selected:
        f=out/(name+'.npz')
        if not f.exists():continue
        with np.load(f) as z:
            tt=z['time'];axes[0].plot(tt,z['state'][:,4],label=name)
            axes[1].plot(tt,z['requested_target'][:,0],label=name)
            axes[2].plot(tt,z['joint_dq_pre'][:,0],label=name)
    axes[0].plot(t[mask]-a.event_time,realstate[mask,4],color='black',linestyle='--',label='real assumed alignment')
    axes[1].plot(t[mask]-a.event_time,target[mask,0],color='black',linestyle='--')
    axes[2].plot(t[mask]-a.event_time,x[mask,10]*20,color='black',linestyle='--')
    for ax,label in zip(axes,['native pitch (rad)','ch0 requested target (rad)','ch0 dq pre (rad/s)']):ax.set_ylabel(label);ax.grid(alpha=.3)
    axes[0].legend(ncol=2,fontsize=8);axes[-1].set_xlabel('seconds after assumed mechanism change');fig.suptitle('Delay sensitivity: common simulated state vs real log; not a fitted hardware model');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
