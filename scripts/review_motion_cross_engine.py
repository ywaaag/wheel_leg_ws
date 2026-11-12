"""Read saved MuJoCo traces only; emit a NEW goal-focused audit, optional CPU2048.

No MuJoCo/Isaac stepping, no online PPO, no legacy raw/selection overwrite.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from native_stand.students.data import sha
from native_stand.command_v6.completion import write
from native_stand.students.motion_priority_audit import VERSION,review,summarize


def bigger_candidate(source,output,probe_inputs,*,selection_note=None):
    from native_stand.students.actor_widening import expand_mcu_checkpoint,wide_network
    from native_stand.students.model import Student,load
    actor,saved=load(source['checkpoint'],'MotionZeroStudent');actor.eval()
    wide=expand_mcu_checkpoint(saved,source_checkpoint_sha256=source['sha256'],width=2048)
    path=output/'actor2048_initial.pt'
    if path.exists():raise FileExistsError(path)
    torch.save(wide,path)
    bigger,_=load(path,'MotionZeroStudent');bigger.eval()
    x=torch.tensor(np.concatenate(probe_inputs),dtype=torch.float32)
    with torch.inference_mode():error=float((actor.mean(x)-bigger.mean(x)).abs().max())
    if error>2e-5:raise ValueError('2048 saved/read-back action error')
    old=copy.deepcopy(actor).double();new=copy.deepcopy(bigger).double()
    probe=x[::max(1,len(x)//8)][:8].double().requires_grad_(True)
    y0,y1=old.mean(probe),new.mean(probe);jacobian=0.
    for col in range(6):
        a=torch.autograd.grad(y0[:,col].sum(),probe,retain_graph=True)[0]
        b=torch.autograd.grad(y1[:,col].sum(),probe,retain_graph=True)[0]
        jacobian=max(jacobian,float((a-b).abs().max()))
    if jacobian>1e-4:raise ValueError('2048 local Jacobian error')
    actor_names=list(saved['actor']);ids=[p for g in saved['optimizer']['param_groups'] for p in g['params']]
    critic_equal=all(torch.equal(v,wide['critic'][k]) for k,v in saved['critic'].items())
    critic_adam_equal=all(all(torch.equal(v,wide['optimizer']['state'][pid][k]) if torch.is_tensor(v)
        else v==wide['optimizer']['state'][pid][k] for k,v in saved['optimizer']['state'][pid].items())
        for pid in ids[len(actor_names):] if pid in saved['optimizer']['state'])
    if not critic_equal or not critic_adam_equal:raise ValueError('Critic or Critic Adam changed')
    # Disposable copy: check finite CPU gradients and mapped Adam state can step.
    # This does not train the delivered initialization or any source controller.
    train=copy.deepcopy(bigger).train()
    critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version=saved['critic_privileged_version'],
                   network_architecture=wide_network(2048));critic.load_state_dict(wide['critic'])
    optimizer=torch.optim.Adam([*train.parameters(),*critic.parameters()]);optimizer.load_state_dict(copy.deepcopy(wide['optimizer']))
    optimizer.zero_grad();loss=train.mean(x[:64]).square().mean()+critic(torch.zeros(64,312)).square().mean()
    loss.backward();params=[*train.parameters(),*critic.parameters()]
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):raise ValueError('2048 nonfinite CPU gradients')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in params):raise ValueError('2048 nonfinite CPU Adam update')
    report=dict(status='CPU_INITIALIZATION_ONLY',source=source,checkpoint=str(path.resolve()),sha256=sha(path),
        architecture=wide_network(2048),actor_mean_parameters=2691590,source_actor_mean_parameters=181382,
        states_audited=len(x),action_max_error=error,local_Jacobian_max_error=jacobian,
        critic_unchanged=critic_equal,critic_Adam_unchanged=critic_adam_equal,
        CPU_optimizer_steps_on_disposable_copy=1,delivered_candidate_optimizer_steps=0,
        online_PPO=0,physical_steps=0,runtime_recipe_ready=False,
        baseline_selection=selection_note or
            f"{source['name']} engineering reference only; no comprehensive winner claimed")
    write(output/'actor2048_review.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--prepare-2048-from',choices=['u9000','u9995','source4500'])
    a=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    terminal=a.evaluation/'terminal_report.json';t=json.loads(terminal.read_text())
    if t['status']!='COMPLETED' or t['completed_cases']!=180 or t['models_unchanged'] is not True:
        raise ValueError('complete immutable180-condition source required')
    manifest=json.loads((a.evaluation/'manifest.json').read_text());models=manifest['models']
    files={str(terminal.resolve()):sha(terminal)}
    for m in models:
        if sha(m['checkpoint'])!=m['sha256'] or sha(m['spec'])!=m['spec_sha256']:
            raise ValueError('source model/spec changed')
        files[m['checkpoint']]=m['sha256'];files[m['spec']]=m['spec_sha256']
    a.output_dir.mkdir(parents=True)
    records=[];probe=[]
    for i,r in enumerate(t['results']):
        path=Path(r['trace']['path']);path=path if path.is_absolute() else a.evaluation/path
        if sha(path)!=r['trace']['sha256']:raise ValueError('raw trace changed: '+str(path))
        files[str(path.resolve())]=r['trace']['sha256']
        with np.load(path) as z:arrays={k:z[k] for k in z.files}
        prefix=r['readback']['controlled_clock_origin_s']
        result=review(arrays,r['case'],r['assessment'],prefix=prefix)
        record=dict(model=r['model'],tested_model=r.get('tested_model'),case=r['case'],termination=r['termination'],
                    trace=r['trace'],review=result)
        if r['model']==a.prepare_2048_from:probe.append(arrays['actor_input'][::100])
        records.append(record)
        if (i+1)%30==0:print(json.dumps(dict(reviewed=i+1,total=180,simulation_steps=0)),flush=True)
    summary=summarize(records,[m['name'] for m in models])
    larger=None
    if a.prepare_2048_from:
        source=next(m for m in models if m['name']==a.prepare_2048_from)
        larger=bigger_candidate(source,a.output_dir,probe)
    changed=[path for path,digest in files.items() if sha(path)!=digest]
    if changed:raise RuntimeError('read-only evidence changed: '+str(changed))
    result=dict(version=VERSION,status='COMPLETED',wall_seconds=time.monotonic()-start,summary=summary,
        records=records,inputs_sha256=files,source_terminal=str(terminal.resolve()),
        priorities=dict(required=['stable standing','straight and in-place yaw magnitude2..4'],
            diagnostic_only=['low-speed magnitude below2 tracking precision'],
            standing_design_goal=dict(mean_horizontal_mps=.1,p95_horizontal_mps=.15,mean_abs_yaw_rad_s=.1,
                                      height_MAE_m=.01,height_p95_m=.02),
            semantics='new design/selection view; source gates and accepted counts remain immutable'),
        larger_Actor=larger,physical_steps=0,new_PPO_updates=0,legacy_selection_changed=False,
        recommended_branch='preserve4500 for high-speed backward and9995 for height/PD; no complete winner',
        new_runtime_recipe_ready=False)
    write(a.output_dir/'review.json',result)
    lines=['# 高速与站立优先：离线核对及Actor2048准备','',
        '只读取180个既有MuJoCo条件，0新物理、0在线PPO。原评估/选模文件未改。低速精度保留诊断，不参与高速与站立的主目标判定。',
        '', '|模型|高速2–4完整窗口|高速原通过→补齐可映射门|高速跌倒|站立原通过→补齐可映射门|站立新设计目标通过|',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        h=s['groups']['highspeed'];z=s['groups']['standing']
        lines.append(f"|{s['model']}|{h['full30']}/{h['conditions']}|{h['original_passed']}→{h['mapped_passed']}|{h['falls']}|{z['original_passed']}→{z['mapped_passed']}/{z['conditions']}|{s['standing_design_goals_passed']}/{z['conditions']}|")
    lines += ['', '没有模型同时完整满足高速和站立目标。来源4500保留高速后退优势，9995保留高度/PD站立优势；不以单个加权总分拼成已通过模型，不覆盖旧9000选择。',
        '', '## 评分对齐','',
        '补齐Native自转spin_center≤.1m/s、关节镜像保持段均值≤.1rad，并复用真实QualityTermination状态机定位首次质量终止。原轨迹不会在该时刻被裁掉，也不据记录的后续恢复宣称Native不会终止。',
        'Native collider clearance、5ms PD反馈/1ms接触完整历史以及八环境/PhysX差异尚不可由本批100Hz记录还原；所有mapped门通过也不是Native等价认证。',
        '站立设计目标mean水平速度≤.10m/s、P95≤.15、mean(abs(yaw))≤.10rad/s、高度MAE≤10mm/P95≤20mm，单独列出，不改旧门或用它阻止未来工程smoke/学习。低速放宽不等于放宽失触、姿态或闭链保护。',
        '', '## 更大Actor','',
        json.dumps(larger,ensure_ascii=False,indent=2) if larger else '本次未生成更大初始化。',
        '', '保持31D/6D、ELU/Tanh、三层隐藏层；2048/1024/512约269万mean参数，Critic312保持。初始化函数与局部Jacobian一致不等于训练轨迹一致、容量已被证明不足或新技能已学会。',
        '', '## 下一份训练设计','',
        '优先固定.34m恢复/保持高速直线和自转，同时训练名义/PD应力站立；高度边界站立复习保留，多高度运动作为下一小阶段。初版配额：20%nominal zero、20%执行应力zero、50%高速复习/学习（四方向等额）、10%较低速度过渡。前后退和正负yaw独立记录，不把未过±4当保留成功。',
        '先复用当前实际PD/armature/MCU执行，LR/std/奖励权重不同时改；为站立抑漂移保留宽核/连续成本，为高速保留tracking、水平姿态、接触、摆角/腿长/轮心及共模/差模约束。低速gain不作为晋级拦截，仍要求方向和物理安全。',
        '更大的网络配方尚无新在线授权；不自动沿用已耗尽10000、不启动PPO或修改旧账本。未来准备runtime仍需独立版本、完整恢复身份、真实31/312/物理读回、奖励快照与TensorBoard。']
    (a.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETED',summary=summary,actor2048=larger,physical_steps=0),ensure_ascii=False))


if __name__=='__main__':main()
