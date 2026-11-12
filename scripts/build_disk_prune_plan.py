"""生成磁盘清理的保留清单与删除计划（只读；默认不删除）。

规则（用户 2026-11-05 决定）：
- 每轮训练**只保留一个 checkpoint**（优先 40 保留清单 → completion_report 的 Best →
  analysis 的 latest → 最大 update 编号）；其余 checkpoint 删除。
- 旧 run 的 npz / tfevents / metrics.jsonl / 逐 update json / run 内冗余源码副本删除。
- spec / 终态 / ledger / 摘要 / source_snapshot 一律保留。
- **活跃训练 run 整个目录不删任何文件**（按进程 --output-dir 判定）。
- World Model 与其他会话数据、未独立审计的 Recovery 历史整体排除。
"""

import argparse
import glob
import hashlib
import io
import json
import os
import re
import subprocess
from datetime import datetime, timezone, timedelta

R = '/home/kellen/wheel_leg_native_train'
CST = timezone(timedelta(hours=8))
KEEP_SRC = 'artifacts/2026_10_28-15_46_17__motion9995_baseline_keep/keep_recommendations.json'

KEEP_NAMES = {
    'spec.json', 'run_state.json', 'summary.json', 'analysis.json',
    'terminal_report.json', 'supervisor_terminal.json', 'chain_terminal_report.json',
    'monitor_exit.json', 'worker_failure.json', 'completion_report.json',
    'budget.json', 'publication_complete.json', 'manifest.json',
    'candidate_selection.json', 'screening_summary.json', 'evaluation_matrix.json',
    'source_selection.json', 'readback.json', 'physics_readback.json',
    'runtime_readback_00.json', 'reward_snapshot.json',
}
KEEP_SUFFIX = ('.md',)
KEEP_PREFIX_NAMES = ('evaluation_', 'trace_manifest')

PROTECT_DIRS = [
    'deployment_protocol',
    'artifacts/2026_10_28-13_45__motion9000_cross_engine_wide/evaluation_framefix',
    'artifacts/2026_10_28-15_46_17__motion9995_baseline_keep',
    'artifacts/2026_10_28-15_19__motion_highspeed_standing_actor2048',
    'src', 'tests', 'scripts', 'docs', 'configs', 'assets', 'manual_train', '.git', 'logs',
    'artifacts/2026_11_5-16_51__motion_smoke12_supported015_dqgain_formal9988',
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_keep():
    d = json.load(open(os.path.join(R, KEEP_SRC)))
    keep = {}
    for it in d['core_models'] + d['actual_parent_models']:
        rel = it['path'].replace(R + '/', '')
        keep[rel] = {'role': it.get('role', 'parent'), 'bytes': it.get('bytes'),
                     'sha256': it.get('sha256')}
    return d, keep


def _add_scope(scope, target):
    """加入活跃作用域，并把同名目录在 runs/isolated/artifacts 三个前缀下一起保护。

    同一任务常在不同前缀下建同名目录（如 isolated/<name>/runtime_view_r4 与
    artifacts/<name>/），只保护一个前缀会误删另一个。
    """
    rel = os.path.relpath(target, R)
    parts = rel.split('/')
    if len(parts) >= 2 and parts[0] in ('runs', 'isolated', 'artifacts'):
        name = parts[1]
        for prefix in ('runs', 'isolated', 'artifacts'):
            if os.path.isdir(os.path.join(R, prefix, name)):
                scope.add(f'{prefix}/{name}')


def active_runs():
    """判定活跃 run：同时消费 --output-dir、--spec 与进程 cwd（runtime_view）。

    只看 --output-dir 会漏掉训练实际使用的 isolated/*/runtime_view/src；删掉那里的
    源码会让正在运行的进程在下次 import 时失败。无法判定时 fail closed。
    """
    try:
        out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return None
    scope = set()
    for line in out.splitlines()[1:]:
        if 'train.py' not in line:
            continue
        pid = line.strip().split()[0]
        for m in re.finditer(r'--output-dir\s+(\S+)', line):
            _add_scope(scope, m.group(1))
        for m in re.finditer(r'--spec\s+(\S+)', line):
            spec_dir = os.path.dirname(m.group(1))
            if os.path.basename(spec_dir).startswith('2026_') or 'experiments' in spec_dir:
                scope.add('configs')
        try:
            cwd = os.readlink(f'/proc/{pid}/cwd')
        except OSError:
            continue
        _add_scope(scope, cwd)
    return scope


def update_index(path):
    m = re.search(r'model_update_(\d+)\.pt$', path)
    return int(m.group(1)) if m else -1


def choose_one(all_files):
    """每 run 选一个 checkpoint：Best → latest → 最大 update → best_validation。"""
    pts = [p for p in all_files if p.endswith('.pt')]
    if not pts:
        return None, 'no_pt'
    for cand in all_files:
        if cand.endswith('completion_report.md'):
            try:
                text = io.open(cand, encoding='utf-8', errors='ignore').read()
            except OSError:
                break
            m = re.search(r'^Best:\s*(\S+?);', text, re.M)
            if m:
                rel = os.path.relpath(m.group(1), R)
                if rel in pts:
                    return rel, 'completion_report_best'
            break
    for cand in all_files:
        if cand.endswith('analysis.json'):
            try:
                d = json.loads(io.open(cand, encoding='utf-8', errors='ignore').read())
            except Exception:
                break
            sel = d.get('candidate_selection') or {}
            for key in ('latest_checkpoint', 'best_checkpoint', 'selected'):
                v = sel.get(key) if isinstance(sel, dict) else None
                p = v.get('path') if isinstance(v, dict) else (v if isinstance(v, str) else None)
                if p:
                    rel = os.path.relpath(p, R)
                    if rel in pts:
                        return rel, 'analysis_' + key
            break
    updates = [p for p in pts if update_index(p) >= 0]
    if updates:
        return max(updates, key=update_index), 'max_update'
    bv = [p for p in pts if p.endswith('best_validation.pt')]
    if bv:
        return bv[0], 'best_validation'
    return max(pts, key=lambda p: os.path.getmtime(p)), 'latest_mtime'


def is_protected(rel):
    for d in PROTECT_DIRS:
        if rel == d or rel.startswith(d + '/'):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ts = datetime.now(CST).strftime('%Y%m%d_%H%M')
    out = args.out or f'artifacts/{ts}__disk_prune_keepone'
    outdir = os.path.join(R, out)
    if os.path.isdir(outdir):
        raise SystemExit(f'输出目录已存在: {out}')

    d, keep40 = load_keep()
    act = active_runs()
    if act is None:
        raise SystemExit('无法判定活跃进程，fail closed')

    # 1) 采集
    files = {'runs': [], 'isolated': [], 'artifacts': []}
    for root in files:
        base = os.path.join(R, root)
        if not os.path.isdir(base):
            continue
        for b, ds, fs in os.walk(base):
            ds[:] = [x for x in ds if x != '.git']
            for f in fs:
                files[root].append(os.path.join(b, f))

    # 2) 每 run 选一个
    chosen, chosen_reason, run_files = {}, {}, {}
    for root, paths in files.items():
        groups = {}
        for p in paths:
            rel = os.path.relpath(p, R)
            parts = rel.split('/')
            run = '/'.join(parts[:2]) if len(parts) >= 3 else None
            if run:
                groups.setdefault(run, []).append(p)
        for run, ps in groups.items():
            rel_run = run
            if rel_run in act:
                chosen_reason[rel_run] = 'ACTIVE'
                continue
            pick, why = choose_one(ps)
            if pick:
                chosen[pick] = run
                chosen_reason[run] = why
        run_files.update(groups)

    # 3) 分类
    entries = []
    skipped = {}
    for root, paths in files.items():
        for p in paths:
            rel = os.path.relpath(p, R)
            parts = rel.split('/')
            run = '/'.join(parts[:2]) if len(parts) >= 3 else ''
            base = os.path.basename(p)
            low = rel.lower()

            if run in act:
                skipped['active_run'] = skipped.get('active_run', 0) + 1
                continue
            if is_protected(rel):
                skipped['protected_scope'] = skipped.get('protected_scope', 0) + 1
                continue
            if rel in keep40 or rel in chosen:
                skipped['keep_checkpoint'] = skipped.get('keep_checkpoint', 0) + 1
                continue
            if ('world_model' in low) or ('/wm_' in low) or ('world/fit' in low):
                skipped['excluded_world_model'] = skipped.get('excluded_world_model', 0) + 1
                continue
            if 'recovery' in low:
                skipped['excluded_recovery_unaudited'] = skipped.get('excluded_recovery_unaudited', 0) + 1
                continue
            if '/source_snapshot' in rel or '/frozen_cpu' in rel or '/runtime_view' in rel:
                skipped['keep_source_identity'] = skipped.get('keep_source_identity', 0) + 1
                continue
            if root == 'artifacts' and not base.endswith(('.npz', '.tmp', '.partial')):
                skipped['artifact_audit_keep'] = skipped.get('artifact_audit_keep', 0) + 1
                continue

            cat = why = None
            if base in KEEP_NAMES or base.endswith(KEEP_SUFFIX) or base.startswith(KEEP_PREFIX_NAMES):
                skipped['keep_record'] = skipped.get('keep_record', 0) + 1
                continue
            elif base.endswith(('.tmp', '.partial', '.temp')):
                cat, why = 'TEMP_ATOMIC_WRITE', '异常退出任务的临时写文件'
            elif base.startswith('events.out.tfevents'):
                cat, why = 'TENSORBOARD_EVENTS', '训练曲线原始事件（终态与摘要已留）'
            elif base == 'metrics.jsonl':
                cat, why = 'METRICS_JSONL', '逐更新度量转储'
            elif base.endswith('.npz'):
                cat, why = 'RAW_TRACE_NPZ', '原始轨迹/失败尾窗（结论已入 evaluation JSON 与摘要）'
            elif base.endswith('.pt'):
                cat, why = 'UNSELECTED_CHECKPOINT', '同一轮训练只保留一个 checkpoint，此为未选用者'
            elif re.match(r'^update_\d+\.json$', base):
                cat, why = 'INTERMEDIATE_JSON', '逐 update 状态转储'
            elif base.endswith(('.py', '.yaml', '.yml')):
                # 收益仅 0.00 GiB，但删掉会失去 run 内源码副本；不值得承担风险
                skipped['keep_source_copy'] = skipped.get('keep_source_copy', 0) + 1
                continue
            if cat is None:
                skipped['unclassified_keep'] = skipped.get('unclassified_keep', 0) + 1
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if sz == 0:
                continue
            entries.append({'path': rel, 'type': 'file', 'size': sz, 'sha256': sha256(p),
                            'category': cat, 'reason': why, 'run': run})

    by_cat = {}
    for e in entries:
        c = by_cat.setdefault(e['category'], {'count': 0, 'bytes': 0, 'runs': set()})
        c['count'] += 1
        c['bytes'] += e['size']
        c['runs'].add(e['run'])
    summary = {k: {'count': v['count'], 'bytes': v['bytes'], 'runs': len(v['runs'])} for k, v in by_cat.items()}

    os.makedirs(outdir)
    io.open(os.path.join(outdir, 'protected_models.json'), 'w', encoding='utf-8').write(
        json.dumps({'schema': 'disk_prune_protected_models_v2',
                    'generated_at': datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S CST'),
                    'source': KEEP_SRC, 'core_and_parent_models': [{'path': k, **v} for k, v in sorted(keep40.items())],
                    'active_runs': sorted(act), 'protect_entire_scopes': d.get('protect_entire_scopes'),
                    'excluded_from_any_deletion': d.get('excluded'),
                    'rule': '每 run 只保留一个 checkpoint；40 个保留模型优先；活跃 run 整目录不动'}, ensure_ascii=False, indent=1))

    io.open(os.path.join(outdir, 'checkpoint_selection.json'), 'w', encoding='utf-8').write(
        json.dumps({'schema': 'checkpoint_selection_v1', 'rule': '每 run 一个',
                    'selected': {k: {'run': v, 'reason': chosen_reason.get(v)} for k, v in sorted(chosen.items())},
                    'active_runs': sorted(act)}, ensure_ascii=False, indent=1))

    io.open(os.path.join(outdir, 'deletion_entries.json'), 'w', encoding='utf-8').write(
        json.dumps({'schema': 'disk_prune_entries_v2', 'entry_count': len(entries),
                    'total_bytes': sum(e['size'] for e in entries),
                    'summary_by_category': summary, 'entries': entries}, ensure_ascii=False, indent=1))

    plan = {'schema': 'disk_prune_plan_v2', 'plan_version': '2', 'root': R,
            'generated_at': datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S CST'),
            'inputs': {'keep_recommendations': {'path': KEEP_SRC, 'sha256': sha256(os.path.join(R, KEEP_SRC))},
                       'checkpoint_selection': {'path': f'{out}/checkpoint_selection.json',
                                                'sha256': sha256(os.path.join(outdir, 'checkpoint_selection.json'))},
                       'deletion_entries': {'path': f'{out}/deletion_entries.json',
                                            'sha256': sha256(os.path.join(outdir, 'deletion_entries.json'))}},
            'rules': ['每 run 只保留一个 checkpoint', '旧 run 的 npz/tfevents/metrics/逐update json 删除',
                      'spec/终态/ledger/摘要/source_snapshot 保留', '活跃 run 整目录不动',
                      'World Model 与未审计 Recovery 排除'],
            'active_runs': sorted(act),
            'entry_count': len(entries), 'total_bytes': sum(e['size'] for e in entries),
            'approval_status': 'UNAUTHORIZED', 'execution_status': 'NOT_EXECUTED',
            'entries': entries}
    payload = json.dumps(plan, ensure_ascii=False, indent=1).encode('utf-8')
    plan_sha = hashlib.sha256(payload).hexdigest()
    pname = f'{out}/deletion_plan_{plan_sha[:12]}.json'
    open(os.path.join(R, pname), 'wb').write(payload)

    io.open(os.path.join(outdir, 'dryrun_summary.json'), 'w', encoding='utf-8').write(
        json.dumps({'schema': 'disk_prune_dryrun_v2', 'generated_at': plan['generated_at'],
                    'plan_path': pname, 'plan_sha256': plan_sha,
                    'entry_count': len(entries), 'total_bytes': plan['total_bytes'],
                    'total_gib': round(plan['total_bytes'] / 2 ** 30, 2),
                    'by_category': summary, 'skipped_counts': skipped,
                    'active_runs': sorted(act),
                    'checkpoints_total': len(files['runs']) and
                    len([p for p in files['runs'] + files['isolated'] if p.endswith('.pt')]),
                    'checkpoints_kept': len(chosen) + len(keep40)}, ensure_ascii=False, indent=1))

    print(f'输出目录: {out}')
    print(f'计划: {pname}')
    print(f'plan SHA256 = {plan_sha}')
    print(f'条目 {len(entries)} 个，合计 {plan["total_bytes"]/2**30:.2f} GiB')
    print()
    for k, v in sorted(summary.items()):
        print('  %-24s %5d 个  %6.2f GiB  runs=%d' % (k, v['count'], v['bytes'] / 2 ** 30, v['runs']))
    print()
    print('活跃 run:', sorted(act))
    print('保留 checkpoint:', len(chosen), '+ 40 清单内', len(keep40))
    print('跳过:', json.dumps(skipped, ensure_ascii=False))


if __name__ == '__main__':
    main()
