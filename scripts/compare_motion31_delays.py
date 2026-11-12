"""Fixed4500 closed-loop observation/target/post-PD delay diagnostic; CPU only."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from compare_motion31_response import ROOT,LOG,ONNX,prepare,sha,write
from native_stand.students.motion_policy_delay import PolicyDelay,cases,VERSION


def plan(identity):
    return dict(version=VERSION,source_actor=str(ONNX),actor_sha256=sha(ONNX),source_log=str(LOG),source_log_sha256=sha(LOG),
        conditions=cases(),duration_s=6.,windows_s=dict(fit=[.5,2.],holdout=[2.,6.]),
        budget=dict(attempts=1,episodes=36,physics_steps=216000,wall_s=900.,output_bytes=256*2**20),
        resources=dict(GPU=False,CPU_threads=1,viewer=False),online_updates=0,optimizer_steps=0,
        observation_scope='Actor sensor slots0:16 delayed; command/task and actual adopted previous remain current; PD uses live source packets',
        target_scope='hypothetical upstream mean queue before live C publication, not a delayed MCU packet; generated/delivered stamps separate',
        post_PD_scope='additional motor torque transport only in named postPD cases; target relay disabled there',
        prehistory='constant first sensor sample; target held at first logged previous until queued mean arrives; no teacher rescue',
        unchanged='Actor31D,1/5/10ms,USD-derived currentPlant,tight solver,motor40Nm,wheel2.44695Nm,source C decoding/filters/adoption/compensation',
        motor_tau_ms=[0.]*6,PD_feedback_extra_age_ms=0,
        initial_state='same log-constrained q/dq/gravity; root vx0, support/passive branch and LQR motor proxy remain hypotheses',
        randomization=False,selection='fixed matrix only; no adaptive parameter search or trained model selection',
        scoring='existing strict full2-6s reproduction, per-leg frequency±4Hz, amplitude ratio.5..2,band20-27 and tilt',
        hardware_parameters_identified=False,certificate_eligible=False)


def observation_features(data,window):
    from native_stand.students.oscillation_search import features
    return features(data['t']-data['t'][data['ix']],data['x'][:,6:10]*1.5,
        data['x'][:,10:16]*20,data['y'],data['tau'],data['x'][:,3:6],window=window)


def report(output,identity):
    """Offline numerical/causal verification and compact human report."""
    from scipy.signal import welch
    d=json.loads((output/'results.json').read_text());ledger=json.loads((output/'ledger.json').read_text())
    request=json.loads((output/'spec.json').read_text());maximum_mean_error=0.;frames=0
    changed=[p for p,h in identity['input_sha256'].items() if sha(p)!=h]
    if changed:raise ValueError('source identity differs at review')
    lines=['# 固定4500：增大观测/目标/出力运输延迟的闭环诊断','',
        '**所有延迟均为仿真假设，不是实机已测参数。** Actor固定为日志匹配4500小模型，PPO/optimizer均0。',
        '比较腿PD40/2/轮Kv.2与20/2/.15；物理1ms、PD5ms、policy10ms，电机40Nm/轮2.44695Nm，原C映射/补偿/限幅/报文处理保留。',
        'Actor观测只延迟0:16传感字段，previous和命令/任务保持当前值。目标延迟是本地C发布前的上游队列假设，不是直接复刻原固件带陈旧snapshot的消息；Actor生成与队列交付的时间分别保存，不放宽C本地新鲜度/因果检查。',
        'postPD条件仅延迟电机命令运输，不叠目标队列；PD反馈保持源报文当前值，出力τ均0，无新噪声。',
        '',f'run `{output.relative_to(ROOT)}`；终态{ledger["status"]}，完成{ledger["scored_episodes"]}/36条件，实际{ledger["physics_steps"]}个1ms步，墙钟{ledger["wall_s"]:.3f}s。',
        '','|条件|运行s|终止|2–6s覆盖|四腿dq std rad/s|100Hz采样主峰Hz|1kHz编码器主峰Hz|严格日志匹配|',
        '|---|---:|---|---:|---|---|---|---|']
    oscillatory=[]
    for row in d['results']:
        path=output/row['trace']['path']
        if sha(path)!=row['trace']['sha256']:raise ValueError('trace identity differs')
        a=np.load(path,allow_pickle=False)
        if 'relay_actor_input' in a:
            x=a['relay_actor_input'];current=a['relay_current_input'];t=a['relay_tick_ms']
            if not np.array_equal(x[:,16:],current[:,16:]):raise ValueError('stale previous or command in Actor')
            for i,stamp in enumerate(a['relay_sensor_source_policy_tick_ms']):
                if stamp>t[i]:raise ValueError('future measurement leaked')
                j=np.flatnonzero(t==stamp)
                if len(j)!=1 or not np.array_equal(x[i,:16],current[j[0],:16]):raise ValueError('sensor age differs')
            for i,stamp in enumerate(a['relay_target_generation_tick_ms']):
                if stamp>=0:
                    j=np.flatnonzero(t==stamp)
                    error=float(abs(a['relay_delivered_mean'][i]-a['relay_requested_mean'][j[0]]).max())
                    maximum_mean_error=max(maximum_mean_error,error)
            frames+=len(x)
        f=row['features_holdout'];v=f.get('dq_std_rad_s',[]);freq=f.get('velocity_peak_hz',[])
        high=row.get('encoder_1kHz_holdout',{}).get('peak_hz',[])
        lines.append(f'|{row["name"]}|{row["physics_steps"]*.001:.3f}|{row["termination"] or "6秒完成"}|{f.get("coverage",0):.3f}|'+
            '/'.join(f'{z:.2f}' for z in v)+'|'+ '/'.join(f'{z:.1f}' for z in freq)+'|'+ '/'.join(f'{z:.1f}' for z in high)+f'|{row["match_holdout"]["matched"]}|')
        if f.get('coverage',0)>=.98 and v and np.median(v)>=1.:
            oscillatory.append(dict(name=row['name'],encoder_dq_std=v,sampled_peak_hz=freq,one_kHz_peak_hz=high,
                strict_match=row['match_holdout']['matched']))
    if maximum_mean_error!=0.:raise ValueError('target queue delivered a different source mean')
    real=d['real']['holdout']
    lines+=['','## 判断范围','',
        '实机2–6秒四腿dq std约'+ '/'.join(f'{v:.2f}' for v in real['dq_std_rad_s'])+'rad/s，100Hz采样主峰约23.44Hz。',
        '已有严格判据同时检查完整2–6秒、四腿频率误差≤4Hz、幅度比.5–2、20–27Hz能量及姿态均值。提前跌倒不算持续复现；低幅信号的频谱峰不能单独称为共振，100Hz与1kHz频谱分开列出。',
        f'完整强振荡候选{len(oscillatory)}项；严格匹配{sum(r["match_holdout"]["matched"] for r in d["results"])}/{len(d["results"])}。',
        '相同Actor在某个延迟假设下振荡，只证明其对该执行条件敏感；不能证明实机正好具有该延迟，也不能单凭相似频率排除柔顺、内部速度测量或负载差异。',
        '初态按日志q/dq/gravity约束重建，root vx0、支撑/被动分支、前一LQR debug矩代理及未知执行队列均为假设；命令沿原日志，不伪称全程zero。后续观测来自仿真，未强制喂真实obs或真实动作来制造振荡。',
        '',f'队列因果核验{frames}帧；命令/任务/previous均保持当前，测量只来自声明的过去帧，交付目标与声明源mean最大差{maximum_mean_error:.1g}；所有输入与trace SHA复核通过。',
        '','## 工程范围','',
        '本轮新增motion_policy_delay.py、compare_motion31_delays.py及两份针对性测试；不修改原执行/训练默认、外部仓库、旧run或模型。新spec/identity/ledger/terminal独立。测试和CPU预算见本轮artifact目录。未启动Isaac/GUI/硬件/PPO/通知；未自动追加扫描。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    summary=dict(terminal=ledger,strong_oscillation_candidates=oscillatory,strict_matches=sum(r['match_holdout']['matched'] for r in d['results']),
        conditions=len(d['results']),causality_frames=frames,target_queue_max_abs=maximum_mean_error,
        report_source_sha256=sha(__file__),hardware_identified=False)
    write(output/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['R_original_obs0_target0','R_original_obs20_target0','R_original_obs0_target20','R_original_obs20_target20',
           'R_lower_obs20_target20','F40_lower_obs20_target20']
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    for name,ax in zip(names,axes.ravel()):
        row=next((r for r in d['results'] if r['name']==name),None)
        if row is not None:
            a=np.load(output/row['trace']['path'],allow_pickle=False)
            if 'time' in a:
                for ch in (0,1,3,4):ax.plot(a['time'],a['dq'][:,ch],lw=.7,label=f'ch{ch}')
            ax.set_title(name);ax.grid(alpha=.2);ax.set_ylabel('encoder rad/s');ax.set_xlim(0,6)
            if 'time' in a:ax.legend(fontsize=7,ncol=4)
    for ax in axes[-1]:ax.set_xlabel('seconds after takeover')
    fig.tight_layout();fig.savefig(output/'delay_comparison.png',dpi=140);plt.close(fig)
    return summary


def evaluate(output,source,validation,data,actor,identity,models,request):
    from native_stand.students.firmware_sampling import profile,source_parameters
    from native_stand.students.firmware_mechanics import reproduction
    from native_stand.students.oscillation_search import features
    from search_motion31_oscillation import run_sampled_case
    from scipy.signal import welch
    start=time.monotonic();deadline=start+request['budget']['wall_s'];results=[];used=0
    ledger=dict(status='RUNNING',attempts=1,scored_episodes=0,physics_steps=0,wall_s=0.,optimizer_steps=0,online_updates=0,failure=None)
    write(output/'spec.json',request);write(output/'identity.json',identity);write(output/'ledger.json',ledger)
    real={k:observation_features(data,w) for k,w in [('fit',(.5,2.)),('holdout',(2.,6.))]}
    model,refs,base=models
    def mean(x):
        with torch.inference_mode():return actor(torch.from_numpy(x)).numpy()
    try:
        for case in request['conditions']:
            before=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if before+16*2**20>request['budget']['output_bytes']:raise RuntimeError('bounded output reserve reached')
            if shutil.disk_usage(output).free<256*2**20:raise RuntimeError('disk reserve reached')
            if used+6000>request['budget']['physics_steps'] or time.monotonic()>=deadline:raise RuntimeError('bounded runtime/physics reached')
            p=source_parameters();p.update(kp=case['kp'],kd=case['kd'],kv=case['kv'],motor_delay_ms=[case['motor_delay_ms']]*6)
            s=profile()
            if case['profile']=='F40':s.update(coupling_stiffness_nm_rad=40.,coupling_damping_nm_s_rad=.4,rotor_inertia=.009)
            wrapper=PolicyDelay(mean,case['observation_ms'],case['target_ms'],data['x'][data['ix'],16:22])
            def queued(x):return wrapper.step(len(wrapper.records)*10,x)
            entry=dict(name=case['name'],parameters=p,sampling=s,force_route='separated',preload_fraction=0.,mechanical_envelope_nm=200.)
            row,arrays=run_sampled_case(model,refs,base,queued,data,validation['library'],entry,6.,deadline)
            used+=row['physics_steps'];row.update(case=case)
            records=wrapper.records
            if records:
                arrays.update({'relay_'+k:np.stack([r[k] for r in records]) for k in records[0]})
            for label,window in [('fit',(.5,2.)),('holdout',(2.,6.))]:
                f=features(arrays['time'],arrays['q'][:,[0,1,3,4]],arrays['dq'],
                    arrays['relay_requested_mean'][:len(arrays['time'])],arrays['command'],arrays['gravity'],window=window) if 'time' in arrays else dict(valid=False)
                row['features_'+label]=f;row['match_'+label]=reproduction(real[label],f)
            fine=arrays.get('fine_1ms')
            if fine is not None:
                mask=(fine[:,0]>=2)&(fine[:,0]<6);v=fine[mask,13:19][:,[0,1,3,4]]
                if len(v)>=100:
                    f,power=welch(v,fs=1000,nperseg=min(2048,len(v)),axis=0);band=(f>=5)&(f<=200)
                    row['encoder_1kHz_holdout']=dict(samples=len(v),std_rad_s=v.std(0).tolist(),peak_hz=f[band][power[band].argmax(0)].tolist())
            trace=output/(case['name']+'.npz');np.savez_compressed(trace,**arrays)
            row['trace']=dict(path=trace.name,sha256=sha(trace));results.append(row)
            ledger.update(scored_episodes=len(results),physics_steps=used,wall_s=time.monotonic()-start)
            write(output/'results.json',dict(real=real,results=results));write(output/'ledger.json',ledger)
            print(json.dumps(dict(completed=len(results),case=case['name'],duration_s=row['physics_steps']*.001,
                termination=row['termination'],matched=row['match_holdout']['matched'])),flush=True)
            if row['termination'] in ('RUNTIME_FAULT','WALL_BUDGET'):raise RuntimeError(row.get('runtime_fault',row['termination']))
        changed=[p for p,h in identity['input_sha256'].items() if sha(p)!=h]
        if changed:raise RuntimeError('input changed during fixed diagnostic: '+repr(changed))
        ledger['status']='COMPLETED'
    except BaseException as exc:
        ledger.update(status='FAILED',failure=type(exc).__name__+': '+str(exc));raise
    finally:
        ledger.update(physics_steps=used,scored_episodes=len(results),wall_s=time.monotonic()-start,
            output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        write(output/'ledger.json',ledger);write(output/'terminal.json',ledger)
    return ledger


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--check',action='store_true');parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--review',type=Path,help='Offline-only review of an existing completed attempt')
    a=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.review:
        p=a.review.resolve();identity=json.loads((p/'identity.json').read_text());print(json.dumps(report(p,identity),indent=2));return
    spec,source,validation,data,actor,identity,models=prepare()
    for p in (Path(__file__).resolve(),ROOT/'src/native_stand/students/motion_policy_delay.py'):
        identity['input_sha256'][str(p)]=sha(p)
    request=plan(identity)
    if not a.run:
        print(json.dumps(dict(CPU_preflight='PASS',conditions=len(cases()),budget=request['budget'],
            source=request['source_actor'],PT_ONNX_max_abs=identity['PT_ONNX_max_abs'],physics_steps=0),indent=2));return
    from native_stand.command_v6.domain.experiment_naming import new_experiment_name,validate_new_experiment_name
    output=(a.output_dir or ROOT/'sim2sim'/new_experiment_name('motion4500_delay_matrix')).resolve()
    if output.parent!=ROOT/'sim2sim':raise ValueError('new diagnostic must reside under project sim2sim')
    validate_new_experiment_name(output.name)
    if shutil.disk_usage(output.parent).free<512*2**20:raise ValueError('insufficient storage')
    output.mkdir(exist_ok=False)
    ledger=evaluate(output,source,validation,data,actor,identity,models,request)
    report(output,identity)
    print(json.dumps(dict(output=str(output),**ledger),indent=2))


if __name__=='__main__':main()
