"""One-shot review of completed log-constrained handover sweeps; no simulation."""
import argparse
import json
from pathlib import Path
import numpy as np


def discrepancy(real, sim):
    """Descriptive equal-weight relative errors, never an identification likelihood."""
    floors={'dominant_leg_target_hz':2., 'leg_target_p95_minus_p5_rad':.05,
            'leg_dq_p95_rad_s':.5, 'pitch_abs_peak_rad':.02, 'roll_abs_peak_rad':.01}
    errors={k:float(np.mean(abs(np.asarray(sim[k])-np.asarray(real[k]))/
                            np.maximum(abs(np.asarray(real[k])), floor))) for k,floor in floors.items()}
    return dict(mean_relative_discrepancy=float(np.mean(list(errors.values()))), per_observable=errors)


def review_search(out, report_dir, terminal, manifest):
    """Review the fit-frozen shortlist; never rerank using holdout."""
    shortlist=json.loads((out/'frozen_shortlist.json').read_text())['candidates']
    validated=[]
    for row in shortlist:
        others=[r for r in terminal['validation'] if r['frozen_candidate_name']==row['name']]
        validated.append(dict(candidate=row['name'],configuration=row['configuration'],cache=row['cache_hypothesis'],
            fit=row['fit'],holdout=row['holdout'],termination=row['termination'],
            duration_s=row['duration_s'],upright_failure_time_s=row['upright_failure_time_s'],
            contact_fraction=row.get('both_contact_fraction'),
            context_fit_scores=[r['fit']['score'] for r in others],
            context_holdout_scores=[r['holdout']['score'] for r in others],
            context_terminations=[r['termination'] for r in others]))
    result=dict(version='motion31_handover_search_review_v1',selection='fit-only frozen order; no holdout reranking',
        candidates=validated,hardware_identified=False,training_export_ready=False,
        limitation='same real log and overlapping initial times; temporal holdout is not independent hardware validation')
    with (report_dir/'comparison.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    budget=json.loads((out/'budget.json').read_text())
    lines=['# MuJoCo交接故障：有约束粗搜索与局部细化', '',
        f"完成{len(terminal['search'])}组搜索＋{len(terminal['validation'])}组初态敏感性检查，物理步{budget['physics_steps_used']}，墙钟{budget['wall_used_s']:.2f}秒，PPO0、权重未变。", '',
        '固定实机3250 dagger2小模型。只搜索腿目标延迟、反馈延迟、一阶响应、dq观测延迟及轮7/11ms两种分配；质量/惯量/armature/PD/摩擦/reward不变。所有腿参数与实际previous来源均未辨识。', '',
        '前0–.8秒用于拟合和冻结六候选，.8–1.6秒留出不参与选模；随后改变未知root车速及接管初始时刻检查敏感性。这些仍来自同一份、且有重叠窗口的日志，不是独立实机验证。', '',
        '评分包含频率、幅值、dq、目标跳变、有符号pitch/roll、目标-dq相关性和轮速。无时间平移/轨迹对齐优化；缺少模拟窗口单独惩罚。接触与真实倒地时刻缺实测，不纳入匹配评分。', '',
        '|候选（拟合排序）|目标/反馈/响应/dq延迟ms|轮模型/缓存假设|拟合误差|留出误差|其他初态留出误差|',
        '|---|---|---|---:|---:|---|']
    for r in validated:
        v=r['configuration']['search_values'];ctx='/'.join(f'{x:.3f}' for x in r['context_holdout_scores'])
        lines.append(f"|{r['candidate']}|{v['target']}/{v['feedback']}/{v['rise']}/{v['dq']}|{v['wheel']}/{r['cache']}|{r['fit']['score']:.3f}|{r['holdout']['score']:.3f}|{ctx}|")
    best=validated[0]; fit=best['fit'];hold=best['holdout']
    lines+=['', '## 第一候选逐项证据（不等于真实参数）', '',
            '|拟合窗口指标|实机|仿真|', '|---|---|---|']
    for field in ['dominant_leg_target_hz','leg_target_p95_minus_p5_rad','leg_dq_p95_rad_s','leg_target_step_max_rad','pitch_abs_peak_rad','roll_abs_peak_rad']:
        lines.append(f"|{field}|{fit['real'][field]}|{fit['simulation'][field]}|")
    lines+=['',f"拟合分项误差：`{json.dumps(fit['per_observable'])}`。",'',
        f"留出分项误差：`{json.dumps(hold.get('per_observable',{}))}`。",'',
        '## 使用边界', '',
        '- 不把最像的一组认作电机实际参数；尤其腿延迟均是假设，轮7/11ms为未拆分的空载响应起始量。',
        '- 三种previous初始假设并不确认真实控制权；初始LQR力矩假设0、支撑高度/车速未知可能改变拟合结果。',
        '- 单段可拟合不意味着可泛化；需要真实执行目标/owner/采样时间戳、腿承重响应与独立日志。',
        '- 尚不生成Isaac训练实参，不启动PPO。原日志、权重和外部资产只读。',
        '- 只保存六个候选与18个初态检查轨迹，其余保存小型指标/配置；不保存重复源码或checkpoint。']
    with (report_dir/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from replay_motion31_shadow_csv import read_csv
    _,obs,actions,t,_=read_csv(manifest['input_sha256'] and next(k for k in manifest['input_sha256'] if k.endswith('.csv')))
    from native_stand.students.real2sim_handover import SCALE
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for row in shortlist[:3]:
        with np.load(out/(row['name']+'.npz')) as z:
            axes[0].plot(z['time'],z['target'][:,0],label=row['name'])
            axes[1].plot(z['time'],z['dq_pre'][:,0])
            g=z['actual_gravity_post'];axes[2].plot(z['state_time'],np.arctan2(g[:,2],-g[:,1]))
    at=shortlist[0]['event_sample_time_s'];m=(t>=at)&(t<at+2.5)
    axes[0].plot(t[m]-at,(actions*SCALE)[m,0],'k--',label='real log; owner UNKNOWN')
    axes[1].plot(t[m]-at,obs[m,10]*20,'k--')
    axes[2].plot(t[m]-at,np.arctan2(obs[m,5],-obs[m,4]),'k--')
    for ax,label in zip(axes,['ch0 target (rad)','ch0 observed dq (rad/s)','native pitch (rad)']):
        ax.set_ylabel(label);ax.axvline(.8,color='gray',linestyle=':');ax.axvspan(.8,1.6,alpha=.08,color='green');ax.grid(alpha=.3)
    axes[0].legend();axes[-1].set_xlabel('seconds after assumed takeover; shaded=.8..1.6s holdout')
    fig.suptitle('Constrained hypothesis search; fit shortlist frozen before temporal holdout')
    fig.tight_layout();fig.savefig(report_dir/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(output=str(report_dir.resolve()),best_fit=best['candidate'],fit_score=fit['score'],holdout_score=hold['score'],training_export_ready=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,help='new exclusive corrected analysis directory; raw run stays read-only')
    a=p.parse_args()
    out=a.run_dir
    report_dir=a.output_dir or out
    if a.output_dir:report_dir.mkdir(parents=True,exist_ok=False)
    terminal=json.loads((out/'terminal_report.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    if terminal['status']!='COMPLETED':raise ValueError('completed diagnostic required')
    if manifest.get('version')=='motion31_handover_coarse_local_search_v1':
        review_search(out,report_dir,terminal,manifest);return
    real={r['event_sample_time_s']:r['metrics'] for r in terminal['real']}
    ranked=[]
    for row in terminal['results']:
        if 'first_second' not in row:continue
        with np.load(out/(row['name']+'.npz')) as z:
            g=z['actual_gravity_post']; pitch=np.arctan2(g[:,2],-g[:,1])
            signed=dict(native_pitch_min_rad=float(pitch.min()),native_pitch_max_rad=float(pitch.max()),
                        terminal_native_pitch_rad=float(pitch[-1]),
                        tilt_failure_direction='positive_pitch' if pitch[-1]>0 else 'negative_pitch')
        ranked.append({**row, **discrepancy(real[row['event_sample_time_s']],row['first_second']),
                       'signed_posture_full_trace':signed})
    ranked.sort(key=lambda r:r['mean_relative_discrepancy'])
    comparison=dict(version='motion31_handover_observable_comparison_v1',ranking=ranked,
                    real=terminal['real'], initializations=terminal['initializations'],
                    training_export_ready=False, hardware_cause_identified=False,
                    limitations=['frequency sampled100Hz/aliasing; initial root velocity and support unknown',
                                 'actual control-owner/previous/execution not logged',
                                 'contact has no real quantitative target; real fall time not observed by upright threshold',
                                 'ranking is equally weighted relative observable error; not statistical parameter identification'])
    with (report_dir/'comparison.json').open('x') as f:json.dump(comparison,f,indent=2,allow_nan=False)
    lines=['# 日志约束MuJoCo交接复现', '',
           '固定实机部署3250 dagger2 Student，非4995；不更新权重、不运行Isaac/PPO。', '',
           '日志固定q/dq/gyro/gravity；初始被动关节只求解闭链一致性，运行中无姿态纠正。root高度由较低圆柱轮触地假设获得；root线速度、previous及控制权仍是假设。没有伪造LQR或实际执行目标。', '',
           f"完成{len(ranked)}组，{terminal['model_unchanged']=}。轮子7/11ms为合并响应起始数据：运输与一阶响应分开，未直接使用未确认单位/轴侧的J/B/Tc。腿/反馈时间是敏感性假设，不是实测参数。", '',
           '## 实机可比较指标', '', '|假设接管样本s|腿目标主频Hz（四通道）|P95−P5目标幅值rad|最大目标跳变rad|pitch/roll峰值rad|', '|---|---|---|---:|---|']
    for at,r in real.items():
        fmt=lambda x:'/'.join(f'{v:.3f}' for v in x)
        lines.append(f"|{at:.6f}|{fmt(r['dominant_leg_target_hz'])}|{fmt(r['leg_target_p95_minus_p5_rad'])}|{r['leg_target_step_max_rad']:.3f}|{r['pitch_abs_peak_rad']:.3f}/{r['roll_abs_peak_rad']:.3f}|")
    lines+=['', '日志未记录真实接触及明确控制权；用户约2秒倒地是观察描述，日志未由up<.5证实该时刻。不能声称失触率或倒地时间已匹配。', '',
            '## 描述性误差最小的前十组', '',
            '按频率、幅值、dqP95、pitch、roll五类相对误差等权排序；不是系统辨识或参数可信区间。', '',
            '|case|初态s/vx假设|profile/cache假设|相对误差|首秒目标跳变rad|全程接触比例|倾倒s|', '|---|---|---|---:|---:|---:|---|']
    for r in ranked[:10]:
        lines.append(f"|{r['name']}|{r['event_sample_time_s']:.3f}/{r['assumed_root_vx_mps']}|{r['configuration']['name']}/{r['cache_hypothesis']}|{r['mean_relative_discrepancy']:.3f}|{r['first_second']['leg_target_step_max_rad']:.3f}|{r['both_contact_fraction']:.3f}|{r['upright_failure_time_s']}|")
    best=ranked[0]; signed=best['signed_posture_full_trace']
    lines+=['', '**方向核查不能省略**：上述排序只描述首秒的绝对量误差，不能代替有符号轨迹及倒地方向匹配。',
            f"排序第一组全程native pitch范围[{signed['native_pitch_min_rad']:.3f},{signed['native_pitch_max_rad']:.3f}]rad，终点{signed['terminal_native_pitch_rad']:.3f}rad；实机最初主变化为正pitch（约+.556rad）。不能只凭倒地时间接近约2秒而称后翻复现成功。"]
    lines+=['', '## 三种previous假设', '',
            '- bridge_actual：初始腿目标=实测q、轮目标=实测dq，之后缓存PD采用目标。这是未验证桥接对照，不是部署建议。',
            '- logged_seed：初始执行桥接同上，但首帧Actor previous来自日志；随后仅采用仿真实际目标。用于分离首帧缓存不一致。',
            '- logged_as_executed：假设日志previous就是初始执行目标；其后也只采用PD目标。不能据此确认真实LQR执行过这些目标。', '',
            '## 迁入Isaac的边界', '',
            '本次不生成已辨识执行器配置，不自动写入训练。只有现象相似不足以确认真实参数；必须补齐执行目标/时序、腿承重响应和至少一条独立验证轨迹。可复用延迟/一阶响应接口，但数值保持诊断假设，不得写成实测常数。', '',
            f"输入路径及SHA见manifest.json；预算见budget.json。原CSV、权重、外部MJCF/USD与历史证据只读。结果目录：{out.resolve()}"]
    with (report_dir/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from replay_motion31_shadow_csv import read_csv
    from native_stand.students.real2sim_handover import SCALE
    _,obs,actions,t,_=read_csv(manifest['csv'])
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    best=ranked[0]
    chosen=[best]+[r for r in ranked if r['configuration']['name']=='ideal' and r['cache_hypothesis']==best['cache_hypothesis'] and r['event_sample_time_s']==best['event_sample_time_s'] and r['assumed_root_vx_mps']==best['assumed_root_vx_mps']][:1]
    for r in chosen:
        with np.load(out/(r['name']+'.npz')) as z:
            label=r['name']+' '+r['configuration']['name']
            axes[0].plot(z['time'],z['target'][:,0],label=label)
            axes[1].plot(z['time'],z['dq_pre'][:,0],label=label)
            g=z['actual_gravity_post'];axes[2].plot(z['state_time'],np.arctan2(g[:,2],-g[:,1]),label=label)
    at=best['event_sample_time_s'];m=(t>=at)&(t<at+2)
    axes[0].plot(t[m]-at,(actions*SCALE)[m,0],'k--',label='real log; owner time unknown')
    axes[1].plot(t[m]-at,obs[m,10]*20,'k--')
    axes[2].plot(t[m]-at,np.arctan2(obs[m,5],-obs[m,4]),'k--')
    for ax,label in zip(axes,['ch0 target (rad)','ch0 observed dq (rad/s)','native pitch (rad)']):
        ax.set_ylabel(label);ax.grid(alpha=.3);ax.set_xlim(0,2)
    axes[0].legend(fontsize=8);axes[-1].set_xlabel('seconds after assumed handover')
    fig.suptitle('Log-constrained initialization; hypothesis comparison, not identified hardware dynamics')
    fig.tight_layout();fig.savefig(report_dir/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(output=str(report_dir.resolve()),best_descriptive_case=best['name'],
                         minimum_discrepancy=best['mean_relative_discrepancy'],training_export_ready=False)))


if __name__=='__main__':main()
