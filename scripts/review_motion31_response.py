"""Offline review of a completed tau-pair run; no simulator integration."""
import argparse
import ctypes as C
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
from scipy import signal
from compare_motion31_response import ROOT,VALIDATION,LOG,LEGS,sha,write
from replay_motion31_shadow_csv import read_csv


def verify_results(run,standing,logs):
    import torch
    from compare_motion31_response import BASE,PT_SHA
    from native_stand.students.model import load
    torch.set_num_threads(1)
    identity=json.loads((run/'identity.json').read_text())
    changed=[p for p,h in identity['input_sha256'].items() if sha(p)!=h]
    if changed:raise ValueError('completed-run input identity changed: '+repr(changed))
    checkpoint=json.loads(BASE.read_text())['source_checkpoint']
    if sha(checkpoint)!=PT_SHA:raise ValueError('source actor changed')
    actor,_=load(checkpoint,'MotionZeroStudent');actor.eval();mean_error=previous_error=target_error=0.;frames=0
    for row in standing:
        path=run/row['trace']['path']
        if sha(path)!=row['trace']['sha256']:raise ValueError('trace differs')
        a=np.load(path,allow_pickle=False);x=a['actor_input'].astype(np.float32)
        with torch.inference_mode():pred=actor(torch.from_numpy(x)).numpy()
        mean_error=max(mean_error,float(abs(pred-a['mean']).max()));frames+=len(x)
        previous=np.concatenate([a['adopted'][:-1,LEGS]/1.5,a['adopted'][:-1][:,[2,5]]/100.],-1)
        previous_error=max(previous_error,float(abs(x[1:,16:22]-previous).max()))
        target_error=max(target_error,float(abs(a['requested_target']-a['mean']*np.array([1.5,1.5,100.,1.5,1.5,100.])).max()))
        # t is suffix controlled-age: it is deliberately0 during frozen prefix.
        valid=a['controlled'].astype(bool)
        if not np.allclose(np.diff(a['t'][valid]),.01,atol=1e-12,rtol=0):raise ValueError('controlled timestamps invalid')
        if not np.all(a['t'][~valid]==0):raise ValueError('prefix clock semantics differ')
    if mean_error>1e-5 or previous_error>1e-6 or target_error>1e-8:raise ValueError('input/output/adoption alignment differs')
    force_error=0.
    for row in logs['results']:
        path=run/row['trace']['path']
        if sha(path)!=row['trace']['sha256']:raise ValueError('log-diagnostic trace differs')
        force_error=max(force_error,row['force_readback']['total_vs_requested_max_abs_nm'])
    if force_error>1e-5:raise ValueError('motor/spring force routing differs')
    pairs=json.loads((run/'paired_results.json').read_text())
    if not all(r['initial_actor_max_abs']==0 and r['initial_physical_identical'] for r in pairs):raise ValueError('initial pairing differs')
    return dict(unchanged_input_files=len(identity['input_sha256']),traces_checked=len(standing)+len(logs['results']),
        policy_frames_reinferred=frames,mean_max_abs=mean_error,previous_max_abs=previous_error,
        requested_target_max_abs=target_error,force_total_max_abs_nm=force_error,
        initial_pairs_identical=len(pairs),analysis_source_sha256=sha(__file__),simulation_steps=0,optimizer_steps=0)


def high_rate(run,logs):
    out={}
    for tau in (0.,2.):
        row=next(r for r in logs['results'] if r['asset']=='current' and r['profile']=='F40' and r['motor_tau_ms']==tau)
        a=np.load(run/row['trace']['path'],allow_pickle=False);fine=a['fine_1ms'];keep=(fine[:,0]>=2)&(fine[:,0]<6)
        values={}
        for name,col in [('load',1),('encoder',13)]:
            dq=fine[keep,col:col+6][:,LEGS];f,power=signal.welch(dq,fs=1000,nperseg=2048,axis=0);band=(f>=5)&(f<=200)
            values[name]=dict(std_rad_s=dq.std(0),peak_hz=f[band][power[band].argmax(0)])
        values['motor_abs_max_nm']=np.max(abs(fine[:,7:13]),axis=0)
        out[str(int(tau))]=values
    return out


def leg_geometry(library,q4):
    """Call the actual firmware coordinate mapper and five-bar C forward model."""
    lib=C.CDLL(str(library));fp=C.POINTER(C.c_float)
    lib.RLMotion31MapPolicyTargets.argtypes=[fp,fp,fp]
    lib.getPhi.argtypes=[C.c_float]*7+[fp]*4;lib.getPhi.restype=None
    header=Path('/home/kellen/combat-dm-balance-main/application/balance/leg_const.hpp').read_text()
    lengths=[float(re.search(r'^#define\s+L_'+str(i)+r'\s+([0-9.]+)f',header,re.M)[1]) for i in range(1,6)]
    result=[]
    for q in np.asarray(q4):
        target=np.zeros(6,np.float32);target[LEGS]=q
        joints=np.empty(4,np.float32);wheels=np.empty(2,np.float32)
        if lib.RLMotion31MapPolicyTargets(target.ctypes.data_as(fp),joints.ctypes.data_as(fp),wheels.ctypes.data_as(fp)):
            raise ValueError('firmware model/motor mapping rejected')
        row=[]
        for side in range(2):
            a,b,l,theta=[C.c_float() for _ in range(4)]
            lib.getPhi(float(joints[2*side]),float(joints[2*side+1]),*lengths,C.byref(a),C.byref(b),C.byref(l),C.byref(theta))
            row.extend([l.value,theta.value])
        result.append(row)
    return np.asarray(result)


def observed_response(t,targets,measured):
    """Descriptive closed-loop transfer estimates, NOT plant identification."""
    grid=np.arange(t[0],t[-1]+1e-7,.01)
    u=np.stack([np.interp(grid,t,targets[:,j]) for j in range(4)],-1)
    y=np.stack([np.interp(grid,t,measured[:,j]) for j in range(4)],-1)
    rows=[]
    for j in range(4):
        f,px=signal.welch(u[:,j],fs=100,nperseg=min(256,len(grid)))
        _,py=signal.welch(y[:,j],fs=100,nperseg=min(256,len(grid)))
        _,xy=signal.csd(u[:,j],y[:,j],fs=100,nperseg=min(256,len(grid)))
        eligible=(f>=5)&(f<=40);i=np.flatnonzero(eligible)[np.argmax(py[eligible])]
        transfer=xy[i]/max(px[i],1e-30)
        coherence=abs(xy[i])**2/max(px[i]*py[i],1e-30)
        rows.append(dict(channel=j,measured_peak_hz=float(f[i]),apparent_gain=float(abs(transfer)),
            apparent_phase_deg=float(np.angle(transfer,deg=True)),coherence=float(coherence),
            measured_std=float(y[:,j].std()),requested_std=float(u[:,j].std())))
    return rows


def leg_rate(library,q4,dq4,epsilon=1e-3):
    columns=[]
    for j in range(4):
        delta=np.zeros_like(q4);delta[:,j]=epsilon
        difference=leg_geometry(library,q4+delta)-leg_geometry(library,q4-delta)
        difference[:,[1,3]]=np.arctan2(np.sin(difference[:,[1,3]]),np.cos(difference[:,[1,3]]))
        columns.append(difference/(2*epsilon))
    jacobian=np.stack(columns,-1)
    return np.einsum('nij,nj->ni',jacobian,dq4)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);a=parser.parse_args()
    a.run_dir=a.run_dir.resolve();out=a.output_dir.resolve()
    if not out.is_dir():raise ValueError('existing independent audit directory required')
    ledger=json.loads((a.run_dir/'ledger.json').read_text())
    plan=json.loads((a.run_dir/'spec.json').read_text());parent=plan.get('continuation',{})
    standing=json.loads((a.run_dir/'combined_standing_results.json').read_text());logs=json.loads((a.run_dir/'log_results.json').read_text())
    total_physics=ledger['physics_steps']+parent.get('used_physics_steps',0)
    total_wall=ledger['wall_s']+parent.get('used_wall_s',0.)
    total_episodes=ledger['scored_episodes']+parent.get('used_scored_episodes',0)
    verification=verify_results(a.run_dir,standing,logs);write(out/'verification.json',verification)
    spectra=high_rate(a.run_dir,logs);write(out/'one_khz_frequency_check.json',spectra)
    _,x,y,t,_=read_csv(LOG)
    with LOG.open(newline='') as stream:raw=list(csv.DictReader(stream))
    name=next(k for k in raw[0] if k.endswith('.rl_active'));enabled=np.array([float(r[name])!=0 for r in raw])
    start=np.flatnonzero(enabled)[0];relative=t-t[start];keep=(relative>=.5)&(relative<6.)
    library=json.loads(VALIDATION.read_text())['library']
    q=x[keep,6:10]*1.5;target=y[keep][:,LEGS]*1.5
    measured=leg_geometry(library,q);requested=leg_geometry(library,target)
    if not np.isfinite(measured).all() or not np.isfinite(requested).all():raise ValueError('task-space FK invalid; cannot fabricate response')
    for col in (1,3):measured[:,col]=np.unwrap(measured[:,col]);requested[:,col]=np.unwrap(requested[:,col])
    response=observed_response(relative[keep],requested,measured)
    rates=leg_rate(library,q,x[keep,10:16][:,LEGS]*20.)
    rates_half=leg_rate(library,q,x[keep,10:16][:,LEGS]*20.,epsilon=5e-4)
    numerical_rates=np.gradient(measured,relative[keep],axis=0)
    task=dict(source_log_sha256=sha(LOG),C_library_sha256=sha(library),window_s=[.5,6.],
        columns=['firmware_leg0_L_m','firmware_leg0_theta_rad','firmware_leg1_L_m','firmware_leg1_theta_rad'],
        values=response,encoded_q_mean=measured.mean(0),encoded_q_std=measured.std(0),
        encoded_task_rate_std=rates.std(0),rate_Jacobian_step_rad=1e-3,
        rate_two_steps_abs_P95=np.quantile(abs(rates-rates_half),.95,axis=0),
        velocity_semantics='C forward-model numerical Jacobian times logged encoder dq; not independent load velocity',
        command_semantics='FK of requested simultaneous motor targets; not an independent excitation or a measured adopted target',
        measurement_semantics='FK derived from motor encoders; not an independent measurement of flexible load position',
        sysid_eligible=False,reason='closed-loop correlated signals, no actual torque/receipt timestamps or independent load measurement')
    write(out/'logged_whole_leg_response.json',task)
    with (out/'logged_whole_leg_response.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['time_s']+['requested_'+s for s in task['columns']]+['encoder_FK_'+s for s in task['columns']]+
            [s+'_per_s_from_dq' for s in task['columns']]+[s+'_per_s_from_sampled_q' for s in task['columns']])
        writer.writerows(np.column_stack([relative[keep],requested,measured,rates,numerical_rates]))
    group=[]
    for asset,profile in [('current','R'),('current','F40'),('colleague','R')]:
        for tau in (0.,2.):
            rows=[r for r in standing if r['case']['asset']==asset and r['case']['profile']==profile and r['case']['motor_tau_ms']==tau]
            metrics=[r['metrics'] for r in rows if r.get('metrics')]
            group.append(dict(asset=asset,profile=profile,tau_ms=tau,episodes=len(rows),survival30=sum(r['survival30'] for r in rows),
                accepted=sum(r['accepted'] for r in rows),
                mean_vx_abs=float(np.mean([r['vx_abs_mean'] for r in metrics])) if metrics else None,
                mean_height_MAE=float(np.mean([r['height_MAE'] for r in metrics])) if metrics else None,
                max_encoder_dq_std=max(max(r['encoder_dq_std']) for r in metrics) if metrics else None,
                max_motor_saturation=max(max(r['saturation_fraction']) for r in metrics) if metrics else None,
                min_contact_fraction=min(min(r['contact_fraction']) for r in metrics) if metrics else None))
    write(out/'result.json',dict(terminal=ledger,cumulative=dict(physics_steps=total_physics,wall_s=total_wall,
        scored_episodes=total_episodes,attempts=1+parent.get('prior_attempts',0)),grouped=group,log_diagnostics=logs,whole_leg_response=task,
        verification=verification,high_rate_frequency=spectra,
        evidence_scope='simulation robustness only; hardware parameters and real-world stability NOT_VERIFIED'))
    lines=['# 原4500：出力τ配对、同事MJCF与日志初态闭环对照','',
        '固定小Actor SHA `efad012efcced3a9ac58dc51839dea98b81598de289dea8ef0986a0813d139a1`；ONNX SHA `776201d861dbf1f00b6824f2c2615068a3206eb45b38529e65e7591bc150e617`。',
        '同一31D/6D、PD20/2/轮Kv.15、1/5/10ms、腿40Nm/轮2.44695Nm；BC/PPO/optimizer均0。τ只作用于四腿post-PD电机出力，每1ms更新一次。','',
        f'补缺run：`{a.run_dir.relative_to(ROOT)}`。终态：**{ledger["status"]}**；父子累计评分episode {total_episodes}/30，实际1ms物理步{total_physics}，墙钟{total_wall:.3f}s。',
        '原16条件完成后，同事分支因构造器误用训练F20默认槽位在积分前失败。修复并通过四worker实际C零积分启动后，只补缺失14条件；旧FAILED/spec/ledger/raw不改，独占claim禁止重复补缺，原物理/墙钟/产物全额保留。',
        '','## 已得到的结论','',
        '1. τ2在刚性Plant没有明显收益；在F40柔顺假设下明显放大高频振荡，不能直接写成训练默认或认为“同事用2ms，我们也会更稳”。',
        '2. 同事MJCF可接入，但我方策略在该桥接条件下仍漂移，未复现实机23Hz。换用对方XML不足以直接解决问题；本轮不是其D5完整运行链。',
        '3. 同一固定Actor在执行响应与柔顺组合变化时可以从平静变强振荡。这支持继续核实整腿动态，尚不能证明本次K/B/τ、反馈拓扑就是真机。',
        '','## 30秒zero保持：包含直接初态及冻结原4500前控1秒的交接','',
        '|Plant|τ ms|生存30秒|完整指标通过|平均abs(vx) m/s|高度MAE m|最大编码器dq std rad/s|最大电机触限|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in group:
        fmt=lambda k:'N/A' if r[k] is None else f'{r[k]:.6f}'
        lines.append(f'|{r["asset"]}/{r["profile"]}|{r["tau_ms"]:g}|{r["survival30"]}/{r["episodes"]}|{r["accepted"]}/{r["episodes"]}|{fmt("mean_vx_abs")}|{fmt("mean_height_MAE")}|{fmt("max_encoder_dq_std")}|{fmt("max_motor_saturation")}|')
    lines+=['','不能将30秒未摔倒等同于站高、零速、对称、接触和触限同时合格。完整逐条件失败项与pre/post状态见两个run的standing_results.json及补缺run的combined_standing_results.json/NPZ；无失败分母删除。',
        '前控和后控使用相同4500权重，因此此处只检查连续状态/previous与后缀计时；它不是不同控制器的LQR→RL接管验证。12个τ配对的首帧31D及完整qpos/qvel均逐值相同。',
        '','## 日志初态闭环：固定0.5–2秒及2–6秒窗口','',
        '|Plant|τ ms|实际仿真时长 s|终止|2–6秒覆盖|四腿编码器dq std rad/s|峰值 Hz|严格23Hz匹配|',
        '|---|---:|---:|---|---:|---|---|---|']
    for r in logs['results']:
        f=r['features_holdout']
        lines.append(f'|{r["asset"]}/{r["profile"]}|{r["motor_tau_ms"]:g}|{r["physics_steps"]*.001:.3f}|{r["termination"] or "6秒完成"}|{f.get("coverage",0):.3f}|{f.get("dq_std_rad_s",[])}|{f.get("velocity_peak_hz",[])}|{r["match_holdout"]["matched"]}|')
    lines+=['','F40 τ2的1kHz全2–6秒编码器和负载峰值均35.6445Hz；编码器std约3.16/4.81/5.75/4.35rad/s，负载仅.278/.299/.257/.231rad/s。这排除了把100Hz的35.55Hz直接误称23Hz或仅凭混叠解释本案例，但未经确认的上游编码器模型仍是关键假设。低振幅R中的频谱峰不能单独称为共振。',
        '沿用真实日志的命令，测量和后续previous来自仿真/C实际采用；不回放真实obs强迫Actor抖动。初始root vx=0、被动闭链分支/轮支撑、初始LQR debug力矩代理和空执行预史均是假设，不是已复原实机全部状态。',
        '','## 同事环境接入边界','',
        '只读加载同事rmuc_peak489_v9.xml及其mesh，保留几何、质量/惯量、.01腿armature、原关节阻尼/摩擦、限位、原接触和闭链solver配置；显式改为我方1ms物理步及40/2.44695Nm执行上限，使用我方C控制链、独立反补偿弹簧假设和31D Actor。',
        '这不是D5原样环境：未使用其25D输入、轮直接力矩、60/2腿PD、8ms策略、4.89Nm轮能力及4.31Nm弹簧模型。因此该对照检验MJCF机械/接触配置改变，不能把失败归于D5训练错误，或用它证明实机哪个参数是真值。',
        '','## 真实整腿命令→角度/速度能核实到什么程度','',
        '已用真实固件RLMotion31MapPolicyTargets/getPhi，将日志两电机目标与编码器同时换算到两侧L/theta，保留交叉耦合，输出logged_whole_leg_response.csv/json。未锁住另一个关节或把同号/反号当解耦。',
        '同时用C正解的数值Jacobian×日志dq计算两侧Ldot/thetadot，并保存100Hz角度有限差分作参考，两种量不强行等同。两步长误差和各量单位写入JSON；无法由100Hz记录独立恢复高频真实负载动态。',
        '现有记录只能提供闭环相关响应；目标是策略请求，不是独立激励；FK来自电机编码器，不是独立负载测量。缺实际电流/矩与真实发送/接收时间，不能由这些数据唯一辨识整腿惯量、刚度、阻尼或电机建立时间。表观相位不可直接当电机纯延迟。实机整腿辨识仍NOT_RUN。',
        '','## 可追溯与边界','',
        '独立spec/identity/ledger/terminal保留全部预算和源SHA；输入在运行后逐SHA复验，未复制源码树/模型。全100Hz轨迹保留，1kHz只保留站立尾窗；日志6秒诊断额外保留分力读回。没有改动外部仓库、旧run/模型/账本；不启动新PPO、GUI或真机。',
        f'离线对齐检查：{verification["policy_frames_reinferred"]}帧重新推理，mean最大差{verification["mean_max_abs"]:.3g}，previous对上一实际采用目标最大差{verification["previous_max_abs"]:.3g}；全部30个trace SHA、12组初态配对和电机/弹簧分力校验通过。',
        'CPU：初始58项针对性通过，修复后23项通过；最终PORTABLE1317通过/2项与历史基线逐ID及缺trace路径相同的失败/9跳过，未弱化SHA或伪造缺失数据。详见cpu_validation.json和原始测试输出。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    plot(a.run_dir,out,logs,data=(relative,x,y))
    print(json.dumps(dict(grouped=group,matched=sum(r['match_holdout']['matched'] for r in logs['results'])),indent=2))


def plot(run,out,logs,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t,x,y=data;fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True)
    m=(t>=0)&(t<=6.)
    for i,ch in enumerate(LEGS):axes[0,0].plot(t[m],x[m,10+ch]*20,label=f'ch{ch}',lw=.8)
    axes[0,0].set_title('Real log: encoder dq (100 Hz samples)')
    for i,ch in enumerate(LEGS):axes[0,1].plot(t[m],y[m,ch]*1.5,lw=.6,label=f'ch{ch}')
    axes[0,1].set_title('Real log: requested leg targets')
    for row,(asset,profile) in enumerate([('current','R'),('current','F40'),('colleague','R')],1):
        for tau,ls in [(0.,'-'),(2.,'--')]:
            case=next(r for r in logs['results'] if r['asset']==asset and r['profile']==profile and r['motor_tau_ms']==tau)
            a=np.load(run/case['trace']['path'],allow_pickle=False)
            if 'time' not in a:continue
            rms=np.sqrt(np.mean(a['dq'][:,LEGS]**2,axis=1));axes[row,0].plot(a['time'],rms,ls,label=f'tau {tau:g} ms')
            state=a['post_state'];axes[row,1].plot(a['time'],state[:,2],ls,label=f'tau {tau:g} ms')
        axes[row,0].set_title(f'{asset}/{profile}: encoder dq RMS');axes[row,1].set_title(f'{asset}/{profile}: root height')
    for i,ax in enumerate(axes.ravel()):
        ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_xlim(0,6);ax.set_ylabel('rad/s' if i%2==0 else ('rad' if i<2 else 'm'))
    for ax in axes[-1]:ax.set_xlabel('seconds after takeover')
    fig.tight_layout();fig.savefig(out/'log_comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
