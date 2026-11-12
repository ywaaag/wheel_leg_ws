"""CPU-only, read-only checkpoint inventory and trace-backed capability ranking.

No Isaac, optimizer, new physics trajectories, or writes into the source run.
Unevaluated weights are never promoted using predictions on another policy's trace.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from native_stand.command_v6.domain.trace_schema import STATE, CONTROL, VERSION
from native_stand.students.model import load
from native_stand.students.protocol import score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranking(row):
    # Same ordering as the public student selector; no reward/height proxy.
    return (row['accepted_envs'], row['entry_envs'],
            row['physical_hold_prefix_sum_s'], -row['update'])


def physical_prefix(s, p, entry):
    """Same first-window per-sample physical prefix as public training evaluation.

    Mean velocity/attitude and cumulative limit checks belong to the certificate.
    """
    if entry is None:
        return 0.
    count=0
    for k in range(entry+1,min(entry+1001,len(s))):
        if (s[k,STATE['height_m']]<.30 or (np.abs(p[k,:2])>.15).any() or
            s[k,STATE['tilt_rad']]>.18 or (s[k,10:12]<=2).any() or
            s[k,STATE['clearance_m']]<=0 or s[k,STATE['closure_m']]>.005):
            break
        count+=1
    return count*.01


def window_metrics(a, env, ids):
    if not len(ids):
        return None
    s = a['state'][ids, env]
    p = a['posture'][ids, env, :2]
    c = a['control'][ids, env]
    q = a['joint_q_pre'][ids, env]
    active = a['_active_indices']
    mirror = np.stack([q[:, active[0]] + q[:, active[3]],
                       q[:, active[1]] - q[:, active[4]]], -1)
    root = a['root_post'][ids, env]
    target = np.zeros((len(ids), 6))
    target[:, [0, 1, 3, 4]] = c[:, CONTROL['leg_target_after_limit']]
    target[:, [2, 5]] = c[:, CONTROL['clipped_action']][:, [2, 5]] * 100
    torque = c[:, CONTROL['applied_torque']]
    return {
        'samples': len(ids), 'duration_s': len(ids) * .01,
        'height_min_mean_max_m': [float(s[:, STATE['height_m']].min()),
                                  float(s[:, STATE['height_m']].mean()),
                                  float(s[:, STATE['height_m']].max())],
        'pitch_roll_abs_mean_rad': np.abs(p).mean(0).tolist(),
        'pitch_roll_abs_peak_rad': np.abs(p).max(0).tolist(),
        'tilt_peak_rad': float(s[:, STATE['tilt_rad']].max()),
        'native_velocity_abs_mean_vx_vy_yaw': np.abs(s[:, 7:10]).mean(0).tolist(),
        'native_velocity_signed_mean_vx_vy_yaw': s[:, 7:10].mean(0).tolist(),
        'root_world_displacement_xyz_m': (root[-1, :3] - a['root_pre'][ids[0], env, :3]).tolist(),
        'root_world_horizontal_path_m': float(np.linalg.norm(np.diff(
            np.concatenate([a['root_pre'][ids[:1], env, :2], root[:, :2]]), axis=0), axis=1).sum()),
        'dual_contact_fraction': float((s[:, 10:12] > 2).all(-1).mean()),
        'closure_peak_m': float(s[:, STATE['closure_m']].max()),
        'policy_saturation_fraction': float(s[:, STATE['saturated']].mean()),
        'hip_active_mirror_abs_mean_rad': np.abs(mirror).mean(0).tolist(),
        'hip_active_mirror_abs_p95_rad': np.quantile(np.abs(mirror), .95, axis=0).tolist(),
        'wheel_target_delta_rms_rad_s': (np.sqrt(np.mean(np.diff(target[:, [2, 5]], axis=0)**2, axis=0)).tolist()
                                          if len(ids) > 1 else None),
        'applied_torque_rms_nm': np.sqrt(np.mean(torque**2, axis=0)).tolist(),
        'requested_torque_rms_nm': np.sqrt(np.mean(c[:, CONTROL['requested_torque']]**2, axis=0)).tolist(),
        'physical_clipped_torque_rms_nm': np.sqrt(np.mean(c[:, CONTROL['physical_clipped_torque']]**2, axis=0)).tolist(),
        'channel_torque_limit_fraction': (np.abs(torque) >= np.array([40,40,5.5,40,40,5.5])*.99).mean(0).tolist(),
        'wheel_torque_sign_changes': (np.diff(np.sign(torque[:, [2,5]]), axis=0) != 0).sum(0).tolist(),
        'sampling_note': '100Hz latest held torque; not full PD/physics recording',
    }


def review(run, output, additional_evaluations=()):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(run):
        raise ValueError('Fresh output outside the source run is required')
    torch.set_num_threads(4)
    phase = run/'formal'
    checkpoints = sorted(phase.glob('model_update_*.pt'), key=lambda p:int(p.stem.split('_')[-1]))
    if not checkpoints:
        raise ValueError('No saved formal checkpoints')
    sources, episodes, groups = {}, [], {}
    paths=list(phase.glob('evaluation_*.json'))
    for extra in additional_evaluations:
        paths.extend(Path(extra).resolve().glob('evaluation_*.json'))
    seen_cases=set()
    for path in sorted(paths):
        if not path.stem.removeprefix('evaluation_').isdigit():
            continue
        d = json.loads(path.read_text())
        if d['field_schema']['state'] != VERSION or d['case']['kind'] != 'student':
            raise ValueError('Unsupported evaluation schema/controller')
        checkpoint = Path(d['case']['checkpoint'])
        if sha(checkpoint) != d['checkpoint_sha256'] or d['case']['sha256'] != d['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA differs: '+str(path))
        trace = path.parent/d['trace']['path']
        if sha(trace) != d['trace']['sha256']:
            raise ValueError('Trace SHA differs: '+str(path))
        for f in (path, trace, checkpoint): sources[str(f)] = sha(f)
        with np.load(trace, allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        a['_active_indices'] = d['field_schema']['active_indices']
        times = a['policy_time_s']
        if not np.isfinite(times).all() or not np.allclose(np.diff(times), .01, atol=2e-6, rtol=0):
            raise ValueError('Invalid trace time axis')
        actor, _ = load(checkpoint, 'RecoveryStudent')
        update = int(checkpoint.stem.split('_')[-1])
        if checkpoint.resolve().parent != phase:
            raise ValueError('Evaluation checkpoint is outside the reviewed training phase')
        case_id=re.sub(r'^u\d+_', '', d['case']['case_id'])
        if (update,case_id) in seen_cases:
            raise ValueError('Duplicate checkpoint/condition evidence; no cherry-picking')
        seen_cases.add((update,case_id))
        group = groups.setdefault(update, {'update':update,'checkpoint':str(checkpoint),
                     'sha256':d['checkpoint_sha256'],'cases':[],'episodes':[]})
        group['cases'].append(case_id)
        for e, old in enumerate(d['per_env']):
            mask = a['first_episode_valid'][:, e].astype(bool)
            ids = np.flatnonzero(mask)
            if not len(ids) or not np.array_equal(ids, np.arange(len(ids))):
                raise ValueError('First episode is empty or discontinuous')
            s, p = a['state'][ids,e], a['posture'][ids,e]
            if len(np.unique(s[:,STATE['episode_id']])) != 1:
                raise ValueError('Multiple episodes stitched')
            owned = a['controller'][ids,e,0] == 3
            result = score(torch.tensor(s), torch.tensor(p), torch.tensor(owned),
                           policy_time_s=torch.tensor(times[ids], dtype=torch.float32))
            for key in ('accepted','entry_index','hold_complete','failed_checks'):
                if result.get(key) != old.get(key):
                    raise ValueError(f'Offline scoring differs: {path} env{e} {key}')
            if bool(a['student_task_events'][ids,e,2].any()) != result['accepted']:
                raise ValueError('Online certificate differs from offline result')
            controlled = ids[owned]
            with torch.no_grad():
                predicted = actor.mean(torch.tensor(a['actor_input'][controlled,e]))
            error = float(np.max(np.abs(predicted.numpy() - a['mean'][controlled,e])))
            if error > 1e-5:
                raise ValueError('Saved actor mean differs on own trajectory')
            entry = result.get('entry_index')
            prefix=physical_prefix(s,p,entry)
            if 'valid_hold_prefix_s' in old and abs(prefix-old['valid_hold_prefix_s'])>1e-6:
                raise ValueError('Recomputed first-window prefix differs')
            hold = ids[entry+1:min(entry+1001,len(ids))] if entry is not None else np.array([], dtype=int)
            before = controlled[controlled <= ids[entry]] if entry is not None else controlled
            ep = {'update':update,'case_id':d['case']['case_id'],'seed':d['case']['seed'],
                  'yaw_rad':d['case']['yaw_rad'],'env':e,'evaluation':str(path),
                  'rescored':result,'physical_hold_prefix_s':prefix,
                  'actor_mean_reproduction_max_abs_error':error,
                  'controlled':window_metrics(a,e,controlled),
                  'pre_entry':window_metrics(a,e,before),'first_hold_window_observed':window_metrics(a,e,hold)}
            episodes.append(ep); group['episodes'].append(ep)
    candidates = []
    if not groups:
        raise ValueError('No complete trace-backed evaluations; cannot rank ability')
    expected = {'student_native','student_yaw10','student_yaw11','student_yaw12'}
    for u,g in sorted(groups.items()):
        rows = g['episodes']
        if len(rows) != 32 or len(g['cases']) != 4 or set(g['cases']) != expected:
            raise ValueError('Incomplete/duplicate comparison matrix')
        candidates.append({k:v for k,v in g.items() if k!='episodes'} | {
            'scored_envs':len(rows),'entry_envs':sum(r['rescored']['entry_index'] is not None for r in rows),
            'complete_windows':sum(r['rescored']['hold_complete'] for r in rows),
            'accepted_envs':sum(r['rescored']['accepted'] for r in rows),
            'physical_hold_prefix_sum_s':sum(r['physical_hold_prefix_s'] for r in rows),
            'physical_hold_prefix_range_s':[min(r['physical_hold_prefix_s'] for r in rows),max(r['physical_hold_prefix_s'] for r in rows)],
            'failed_checks':dict(Counter(c for r in rows for c in r['rescored']['failed_checks'])),
            'entry_time_mean_s':float(np.mean([r['rescored']['settle_s'] for r in rows if r['rescored'].get('settle_s') is not None])) if any(r['rescored'].get('settle_s') is not None for r in rows) else None,
        })
    inventory = []
    for path in checkpoints:
        actor, meta = load(path,'RecoveryStudent')
        if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
            raise ValueError('Nonfinite checkpoint')
        u = int(path.stem.split('_')[-1])
        inventory.append({'update':u,'checkpoint':str(path),'sha256':sha(path),
                          'interface':meta['interface_version'],'load_and_finite_check':'PASS',
                          'closed_loop_status':'TRACE_RESCORED' if u in groups else 'NOT_EVALUATED'})
    ranked = sorted(candidates,key=ranking,reverse=True)
    aggregate = {}
    for u in groups:
        windows=[e['first_hold_window_observed'] for e in episodes if e['update']==u and e['first_hold_window_observed']]
        if not windows:
            aggregate[str(u)]=None
            continue
        aggregate[str(u)]={k:np.mean([m[k] for m in windows],axis=0).tolist() for k in (
            'height_min_mean_max_m','pitch_roll_abs_mean_rad','pitch_roll_abs_peak_rad',
            'native_velocity_abs_mean_vx_vy_yaw','hip_active_mirror_abs_mean_rad',
            'wheel_target_delta_rms_rad_s','channel_torque_limit_fraction','policy_saturation_fraction')}
    sources[str(Path(__file__).resolve())]=sha(__file__)
    report = {'version':'recovery31_offline_checkpoint_review_v1','source_run':str(run),
              'online_PPO_updates':0,'offline_optimizer_steps':0,'new_physics_episodes':0,
              'selection_rule':'accepted count; entry count; physical first-window prefix sum; earlier update',
              'limits':['Unevaluated checkpoints have no closed-loop ability measurement.',
                        'Physical hold prefix excludes mean velocity and is not a success certificate.',
                        'Historical early_failure check is preserved; no historical score changed.',
                        'Evaluation protocol/interface headers contain inherited standing49 labels; actual trace critic is64 and actor31. Scoring uses student_autonomous_swing_v1, verified against each stored per-env result.'],
              'sources_sha256':sources,'inventory':inventory,'ranked_candidates':ranked,
              'observed_post_entry_window_means':aggregate,
              'best_observed_candidate':ranked[0] if ranked else None,
              'accepted_candidate':next((r for r in ranked if r['accepted_envs']==32),None),
              'episodes':episodes}
    output.mkdir(parents=True)
    (output/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'per_environment.csv').open('w',newline='') as f:
        keys=['update','case_id','env','entry','complete_window','accepted','physical_hold_prefix_s','failures','mean_abs_vx','mean_abs_yaw','pitch_peak','roll_peak','mirror_hip','mirror_active','saturation']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for ep in episodes:
            r=ep['rescored'];m=ep['first_hold_window_observed'] or ep['controlled']
            w.writerow(dict(zip(keys,[ep['update'],ep['case_id'],ep['env'],r.get('entry_index') is not None,r['hold_complete'],r['accepted'],ep['physical_hold_prefix_s'],','.join(r['failed_checks']),*m['native_velocity_abs_mean_vx_vy_yaw'][::2],*m['pitch_roll_abs_peak_rad'],*m['hip_active_mirror_abs_mean_rad'],m['policy_saturation_fraction']])))
    lines=['# Recovery checkpoint离线能力核验','', '仅消费本次已保存轨迹；无仿真、无optimizer更新。未测checkpoint不能据此推断能力。','',
           '| update | 入口 | 完整窗口 | 完整通过 | 物理保持前缀范围/s | 失败项 |','|---|---|---|---|---|---|']
    for r in sorted(candidates,key=lambda r:r['update']):
        lines.append(f"| {r['update']} | {r['entry_envs']}/32 | {r['complete_windows']}/32 | {r['accepted_envs']}/32 | {r['physical_hold_prefix_range_s']} | {r['failed_checks']} |")
    lines+=['','以下为入口后已记录部分的逐环境均值；1000/1500包含末尾失衡，窗口不满10秒，不能与500视为等时长保持认证。','',
            '| update | 高度均值/m | pitch/roll绝对均值/rad | abs(vx)/abs(yaw) | 髋/连杆镜像MAE/rad | policy触限比例 |',
            '|---|---|---|---|---|---|']
    for u,m in sorted(aggregate.items(),key=lambda item:int(item[0])):
        if m is None:
            lines.append(f'| {u} | 未进入站姿 | — | — | — | — |')
            continue
        fmt=lambda x:'/'.join(f'{v:.4f}' for v in x)
        lines.append(f"| {u} | {m['height_min_mean_max_m'][1]:.4f} | {fmt(m['pitch_roll_abs_mean_rad'])} | {fmt(m['native_velocity_abs_mean_vx_vy_yaw'][::2])} | {fmt(m['hip_active_mirror_abs_mean_rad'])} | {m['policy_saturation_fraction']:.2%} |")
    lines+=['',f"已保存checkpoint：{len(inventory)}；有完整同协议轨迹：{len(candidates)}。",'',
            '最佳实验候选：'+(ranked[0]['checkpoint'] if ranked else 'NONE'),
            'accepted：'+('NONE' if report['accepted_candidate'] is None else report['accepted_candidate']['checkpoint']),
            '最佳实验候选SHA：'+(ranked[0]['sha256'] if ranked else 'NONE'),
            '', '逐环境指标见per_environment.csv；完整阶段指标、SHA、未测清单见review.json。',
            '保持前缀只描述连续物理条件，不包含速度/yaw均值验收；不能称为合格站稳。',
            '同一训练流的原生Swing与小yaw条件，不代表广泛扰动或独立训练seed认证。',
            '历史early_failure字段照原规则保留。评估header含继承的standing49标签，而实际数组为Actor31/Critic64；本报告使用与原逐环境结果一致的自主Swing评分，不修改历史证据。',
            '', '未做独立闭环评估的update：'+', '.join(str(r['update']) for r in inventory if r['closed_loop_status']=='NOT_EVALUATED')]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--additional-evaluations',action='append',default=[])
    a=p.parse_args();r=review(a.run,a.output,a.additional_evaluations)
    print(json.dumps({'checkpoints':len(r['inventory']),'evaluated':len(r['ranked_candidates']),
                      'best':r['best_observed_candidate'],'accepted':r['accepted_candidate']},indent=2))
