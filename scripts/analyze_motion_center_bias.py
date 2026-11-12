"""多份实机日志的支撑目标中心分析。只读模型／CSV，不改部署或训练。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

import analyze_motion_standing_log as log_tools
from native_stand.artifacts.io import file_sha256, new_dir, read_json, write_json


LEGS = [0, 1, 3, 4]


def symmetric_center(targets):
    """先按每帧投影镜像髋／共同active，再取中位数；不改变任何动作。"""
    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.isfinite(values).all():
        raise ValueError("需要非空、有限的N×4物理目标")
    hip = np.median((values[:, 0] - values[:, 2]) / 2)
    active = np.median((values[:, 1] + values[:, 3]) / 2)
    return np.array([hip, active, -hip, active])


def coverage(targets, center, halfwidth):
    """原物理目标的表示率，不是新策略成功率。"""
    return float((abs(targets - center) <= halfwidth + 1e-9).all(1).mean())


def quiet_zero_segments(log, matched):
    t, obs = log['t'], log['obs']
    near_zero = (log['active'] & matched & (abs(obs[:, 22]*4) < .010001)
                 & (abs(obs[:, 23]*4) < .010001))
    selected, segments = [], []
    for block in log_tools.spans(near_zero, t):
        if block['span_s'] < .3:
            continue
        ids = np.arange(block['first_row'], block['last_row'] + 1)
        ids = ids[t[ids] >= t[ids[0]] + .1]
        if (len(ids) < 20 or np.diff(t[ids]).max() > .02
                or abs(log['angles'][ids]).max() > 5
                or log['q'][ids].std(0).max() > .01
                or log['dq'][ids].std(0).max() > .3):
            continue
        selected.extend(ids.tolist())
        segments.append(dict(first_row=int(ids[0]), last_row=int(ids[-1]), rows=len(ids),
                             first_s=float(t[ids[0]]), last_s=float(t[ids[-1]])))
    return np.array(selected, dtype=int), segments


def summary(log, target, indices, old_center):
    if not len(indices):
        return None
    mean = log['mean'][indices][:, LEGS]
    return dict(samples=len(indices), mean_median=np.median(mean, 0).tolist(),
                q_median_rad=np.median(log['q'][indices], 0).tolist(),
                target_median_rad=np.median(target[indices], 0).tolist(),
                target_P05_P95_rad=np.percentile(target[indices], [5, 95], axis=0).tolist(),
                old_center_rad=old_center.tolist(),
                residual_median_rad=np.median(target[indices] - old_center, 0).tolist(),
                zero_mean_same_state_P_change_median_Nm=np.median(40*(old_center-target[indices]), 0).tolist(),
                measured_height='UNKNOWN', measured_contact='UNKNOWN')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, action='append', required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--fit-log', required=True, help='仅这一份日志用于提出中心，其余只验证')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    protected = [*args.csv, Path(__file__), Path(log_tools.__file__),
                 args.package/'actor.onnx', args.package/'manifest.json',
                 args.package/'reference_adapter.py', args.package/'output_contract.json']
    hashes = {str(p.resolve()): file_sha256(p) for p in protected}
    manifest = read_json(args.package/'manifest.json')
    if file_sha256(args.package/'actor.onnx') != manifest['actor_onnx_sha256']:
        raise ValueError('ONNX与部署manifest不一致')
    adapter = log_tools.load_adapter(args.package)
    contract = adapter.contract()
    if contract['leg_halfwidth_rad'] != .15 or contract['wheel_halfwidth_rad_s'] != 80:
        raise ValueError('本分析只适用声明的015/80物理解码')
    center = adapter.centers(.34).astype(float)
    graph = onnx.load(args.package/'actor.onnx')
    onnx.checker.check_model(graph, full_check=True)
    actor = ReferenceEvaluator(graph)
    records, data = [], {}
    for path in args.csv:
        log = log_tools.read_log(path)
        obs, mean = log['obs'], log['mean']
        if not log['active'].any():
            raise ValueError('没有可分析的非零RL调试命令：' + path.name)
        if (abs(obs[log['active'], 24]*.5 - .34) > 1e-6).any():
            raise ValueError('本次只分析0.34m命令，不外推其他高度')
        x = np.c_[obs, np.tile([1, 0, 0, 1, 0, 0], (len(obs), 1))].astype(np.float32)
        pred = actor.run(None, {graph.graph.input[0].name: x})[0]
        errors = abs(pred - mean).max(1)
        matched = errors < 1e-5
        valid = matched & log['active']
        target = center + .15*mean[:, LEGS]
        ids, blocks = quiet_zero_segments(log, matched)
        valid_ids = np.flatnonzero(valid)
        row = dict(file=path.name, source=str(path.resolve()), source_SHA256=file_sha256(path),
                   raw_rows=len(obs), debug_segments=log_tools.spans(log['active'], log['t']),
                   matched_active_rows=int(valid.sum()), active_rows=int(log['active'].sum()),
                   matched_max_error=float(errors[valid].max()),
                   mismatched_active_spans=log_tools.spans(log['active'] & ~matched, log['t']),
                   all_matched=summary(log, target, valid_ids, center),
                   quiet_near_zero=summary(log, target, ids, center), quiet_segments=blocks)
        if len(ids):
            row['separately_estimated_symmetric_center_rad'] = symmetric_center(target[ids]).tolist()
        records.append(row)
        data[path.name] = (log, target, valid, ids)
    if args.fit_log not in data or not len(data[args.fit_log][3]):
        raise ValueError('拟合日志没有合格的短时近零命令片段')
    candidate = symmetric_center(data[args.fit_log][1][data[args.fit_log][3]])
    evaluations = []
    for path in args.csv:
        log, target, valid, ids = data[path.name]
        selections = [('all_matched', np.flatnonzero(valid)), ('quiet_near_zero', ids)]
        if 'normal3' in path.name:
            selections += [('later_quiet_time_window', np.flatnonzero(valid & (log['t'] >= 45) & (log['t'] < 155))),
                           ('oscillation_time_window', np.flatnonzero(valid & (log['t'] >= 30) & (log['t'] < 40)))]
        for label, ix in selections:
            if not len(ix):
                continue
            current = target[ix]
            evaluations.append(dict(file=path.name, selection=label, samples=len(ix),
                role='FIT' if path.name == args.fit_log else 'HELD_OUT_FILE',
                first_s=float(log['t'][ix[0]]), last_s=float(log['t'][ix[-1]]),
                candidate_residual_abs_P95_rad=np.percentile(abs(current-candidate),95,axis=0).tolist(),
                maximum_required_halfwidth_rad=float(abs(current-candidate).max()),
                new_mean_if_preserve_physical_targets_median=np.median((current-candidate)/.15,axis=0).tolist(),
                coverage=[dict(halfwidth_rad=w, old_center_fraction=coverage(current,center,w),
                               candidate_center_fraction=coverage(current,candidate,w))
                          for w in (.15,.13,.12,.11,.10,.08,.06)]))
    result = dict(version='motion2320_three_logs_center_bias_v1',
        model_package=str(args.package.resolve()), model_sha256=file_sha256(args.package/'actor.onnx'),
        unrecorded_task_inputs_assumed=[1,0,0,1,0,0], source_sha256=hashes,
        channels=['geometric_right_hip/ch0','geometric_right_active/ch1','geometric_left_hip/ch3','geometric_left_active/ch4'],
        current_center_rad=center.tolist(), records=records, candidate_center_rad=candidate.tolist(),
        fit_file=args.fit_log, candidate_symmetry='hip centers opposite; active centers equal; no runtime symmetry projection',
        height_scope_m=.34, other_heights='NOT_IDENTIFIED_BY_THIS_DATA',
        candidate_shift_rad=(candidate-center).tolist(),
        unchanged_mean_center_only_prelimit_P_change_Nm=(40*(candidate-center)).tolist(),
        exact_target_remap='new_mean=(old_center+.15*old_mean-new_center)/new_scale; outside[-1,1] cannot be exactly represented',
        evaluations=evaluations,
        deployment_change=False, training_change=False, firmware_change=False,
        source_sha_unchanged=all(file_sha256(Path(p))==h for p,h in hashes.items()),
        physical_steps=0, PPO_updates=0, supervised_updates=0)
    if not result['source_sha_unchanged']:
        raise RuntimeError('分析期间原始输入发生变化')
    out = new_dir(args.output_dir)
    write_json(out/'input_manifest.json', dict(
        model_package=str(args.package.resolve()),
        source_sha256=hashes,
        fit_file=args.fit_log,
        validation_files=[p.name for p in args.csv if p.name != args.fit_log],
        limits='decoded requested targets; adopted targets, contact and measured root height are absent',
        physical_steps=0, PPO_updates=0))
    write_json(out/'analysis.json',result)
    draw(out,result,data,candidate,center)
    report(out,result)
    print('COMPLETED CPU center analysis:',out)
    print('candidate only:',candidate.tolist())


def draw(out,result,data,candidate,center):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.8))
    x = np.arange(4)
    axes[0].plot(x,center,'ko--',label='Current output center')
    axes[0].plot(x,candidate,'D-',color='#2b8c5a',label='Candidate fit from normal2')
    for color,row in zip(['#cc6c29','#316ab2','#8d4a9a'],result['records']):
        s = row['quiet_near_zero']
        if s is None:
            continue
        values=np.asarray(s['target_median_rad'])
        axes[0].plot(x,values,'o-',color=color,label=row['file'].split('261106')[-1][:-4])
    ix=np.flatnonzero((data['Ozone_DataSampling_261106normal3.csv'][0]['t']>=45)
                     &(data['Ozone_DataSampling_261106normal3.csv'][0]['t']<155)
                     &data['Ozone_DataSampling_261106normal3.csv'][2])
    target=data['Ozone_DataSampling_261106normal3.csv'][1][ix]
    axes[1].boxplot(target-center,positions=x-.17,widths=.26,showfliers=False,
                    patch_artist=True,boxprops=dict(facecolor='#b7cfe9'))
    axes[1].boxplot(target-candidate,positions=x+.17,widths=.26,showfliers=False,
                    patch_artist=True,boxprops=dict(facecolor='#b7dfc6'))
    axes[1].axhline(0,color='black',lw=.7)
    axes[1].set_title('normal3 45-155s: target minus center\nBlue: current; green: candidate',fontsize=11)
    axes[0].set_title('Requested leg targets and output centers\nHeight command: 0.34m',fontsize=11)
    axes[0].set_ylabel('Leg target (rad)');axes[1].set_ylabel('Target minus center (rad)')
    for ax in axes:
        ax.set_xticks(x,['Right hip','Right active','Left hip','Left active'])
        ax.grid(axis='y',alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'center_comparison.png',dpi=160);plt.close(fig)


def report(out,r):
    fmt=lambda values:' / '.join(f'{x:+.4f}' for x in values)
    lines=['|日志|近零命令片段／样本|四腿mean中位数|四腿物理目标中位数rad|',
           '|---|---:|---|---|']
    for row in r['records']:
        s=row['quiet_near_zero']
        if s:
            lines.append(f"|{row['file']}|{len(row['quiet_segments'])} / {s['samples']}|{fmt(s['mean_median'])}|{fmt(s['target_median_rad'])}|")
        else:
            lines.append(f"|{row['file']}|没有通过筛选的片段|不据振荡均值拟合中心|—|")
    held=next(e for e in r['evaluations'] if e['file'].endswith('normal3.csv') and e['selection']=='later_quiet_time_window')
    osc=next(e for e in r['evaluations'] if e['file'].endswith('normal3.csv') and e['selection']=='oscillation_time_window')
    fit=next(row for row in r['records'] if row['file']==r['fit_file'])
    heldzero=next(e for e in r['evaluations'] if e['file'].endswith('normal3.csv') and e['selection']=='quiet_near_zero')
    c12=next(c for c in held['coverage'] if c['halfwidth_rad']==.12)
    osc15=next(c for c in osc['coverage'] if c['halfwidth_rad']==.15)
    text=f'''# 三份11月6日日志：输出中心是否需要重定

**两个髋关节有可重复的输出偏移，值得测试更接近平静支撑时请求目标的新中心。不能直接移动现有电控中心，也不能由本次静态表示率宣布更小scale可部署。**

## 范围和筛选

normal、normal2原样保存在logs，normal3沿用已有原件；SHA见analysis.json。三份都是同一个2320全物理观测20ms小模型：normal和normal2全部非零RL调试力矩行逐帧复现，normal3早期不同步行排除，仅用同输入误差低于1e−5的记录。日志没有owner、独立实测高度、接触力和实测轴矩。下文物理目标由网络输出解码，是请求值，没有独立的实际采用回读。本次仅CPU离线推理、统计与代数，无新仿真、训练、固件或硬件修改。

near-zero片段要求vx／yaw绝对值不超过约0.01，连续至少0.3秒，去掉前0.1秒后至少20帧，四腿位置标准差不超过0.01rad、速度标准差不超过0.3rad/s、pitch／roll绝对值不超过5°。这些是短时平静片段，不是长期零命令认证。

normal全段约12秒、normal2约40秒，均包含明显抖动；因此不拿整段mean均值拟合站立中心。

## 偏移确实重复出现

四腿顺序为几何右hip、右active、左hip、左active。当前0.34m输出中心：

`{fmt(r['current_center_rad'])}` rad。

{chr(10).join(lines)}

两个髋的mean反复接近−0.91／+0.68～+0.72；active的mean更靠近0。说明主要是髋目标中心与这些短时支撑目标相差约0.10～0.14rad，而不是四个通道都需要同样的偏移。

## 为什么静止时网络输出不必为0

中心是网络动作的参考，不是编码器零位，也不是无需电机出力的姿态。支撑重力、弹簧补偿、原PD与实际构型共同决定q_target。静止时仍满足`tau_PD=Kp*(q_target-q)`；q_target与实测q不同很正常。

本次normal2片段中，若把mean强制改0而保留当前中心，同状态P项中位变化约`{fmt(fit['quiet_near_zero']['zero_mean_same_state_P_change_median_Nm'])}` Nm。这些偏移正在改变实际控制量，不能视为无效输出删掉。

但小包络中长期偏向一侧，会浪费该侧可用余量。因此现在有理由研究更合适的中心，让网络把更多范围用于动态纠偏。

## 一个明确的0.34m候选

只用normal2的{len(fit['quiet_segments'])}个片段、{fit['quiet_near_zero']['samples']}条记录拟合。每帧先取hip镜像中心`(right_hip_target-left_hip_target)/2`、active共同中心`(right_active_target+left_active_target)/2`，再取中位数：

`{fmt(r['candidate_center_rad'])}` rad。

这是固定输出参考，未对运行中动作做对称投影。用未参与拟合的normal3检查：独立估计中心与其接近，短时近零片段的{heldzero['samples']}条旧目标都能放入新中心±0.10rad。

在normal3 45～155秒11000条记录中，原目标放入中心±0.12rad的比例，由旧中心{c12['old_center_fraction']*100:.1f}%提高到候选中心{c12['candidate_center_fraction']*100:.1f}%。这是旧目标的静态可表示比例，不是新模型成功率；该长窗口仍有非零速度命令。

## 动态余量仍必须保留

normal3 30～40秒振荡片段中，候选中心±0.15rad只能表示{osc15['candidate_center_fraction']*100:.1f}%的旧四腿目标。最大所需偏移约{osc['maximum_required_halfwidth_rad']:.3f}rad。部分旧大动作可能不理想，但必须由新策略学会替代，不能静默裁掉后称能力不变。

如果只换中心、保留现有网络mean，目标会整体平移。Kp40下，限幅前P项改变为`{fmt(r['unchanged_mean_center_only_prelimit_P_change_Nm'])}` Nm，两个髋约4.8Nm。这会立即改变当前已较平稳的控制。

若要在离线保留相同物理目标，标签必须改为`new_mean=(old_physical_target-new_center)/new_scale`。越出±1的标签如实记录，不能裁剪成“准确复刻”；精确保留同一个物理函数也不会自动减小dq反馈增益。

## 推荐的新定义与验证顺序

1. 继续区分几何q、承重PD目标中心、策略修正和编码器零位。建议命名为“名义承重目标中心”，不是“站立零位”。本次候选只限0.34m，不能用它拟合0.30～0.40m全高度。
2. 候选中心可从仿真静态平衡或冻结策略的零命令目标提出，再用这批实机短时目标约束。保留左右镜像参考，未确认的实体零偏仍留给独立标定，不直接吸收到四个不对称常量。
3. 下一版先只更改中心定义，腿0.15／轮80、31D输入、PD、时序和限矩保持；重新补教／训练并独立验证。不要同时缩scale、改dq处理和PD。
4. 先验证重复接管、30～60秒零命令、低速前后／转向、小扰动恢复。检查目标触边、实体25Hz波动及双轮支撑；通过后再研究缩小scale。
5. 现有2320实机基线保持不变。新中心只作为设计证据，不生成部署包、不重启当前PPO。

## 新增文件

- scripts/analyze_motion_center_bias.py：复用现有日志读入／字段映射，逐帧推理及中心分析。
- tests/test_motion_center_bias.py：镜像投影、物理目标重编码、越界及P项变化检查。
- 本目录analysis.json、REPORT.md、center_comparison.png、input_manifest.json：小型结果和输入记录，不复制模型或源码树。
'''
    (out/'REPORT.md').write_text(text)


if __name__ == '__main__':
    main()
