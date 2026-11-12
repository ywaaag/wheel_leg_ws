"""Read a completed control capture; preserve old evidence, exclusive report output."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from native_stand.students.motion_control_readback import VERSION,analyze

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def review(run,output):
 run=Path(run).resolve();output=Path(output).resolve()
 if output.exists():raise FileExistsError(output)
 terminal=json.loads((run/'terminal_report.json').read_text())
 if terminal['status']!='COMPLETED' or terminal['exit_code']!=0:raise ValueError('capture not completed; inspect failure evidence')
 request=json.loads((run/'request.json').read_text());summary=json.loads((run/'summary.json').read_text())
 if request['version']!=VERSION or summary['version']!=VERSION or len(summary['results'])!=4:raise ValueError('capture version/coverage differs')
 inputs={};records=[]
 for name in ('terminal_report.json','request.json','summary.json','budget.json','reward_snapshot.json','effective_capture.json'):
  inputs[str(run/name)]=digest(run/name)
 cp=Path(request['checkpoints'][0]['checkpoint']);inputs[str(cp)]=digest(cp)
 if inputs[str(cp)]!=request['checkpoints'][0]['sha256']:raise ValueError('checkpoint SHA differs')
 for i,command in enumerate(request['commands']):
  ep=run/f'evaluation_{i:02d}.json';trace=run/f'trace_{i:02d}.npz'
  inputs[str(ep)]=digest(ep);inputs[str(trace)]=digest(trace)
  payload=json.loads(ep.read_text())
  if payload['case']['command']!=command or payload['checkpoint_sha256']!=inputs[str(cp)]:raise ValueError('case identity differs')
  with np.load(trace) as arrays:metrics=analyze(arrays)
  if metrics!=payload['readback']:raise ValueError('raw replay versus capture analysis differs')
  records.append({'command':command,'metrics':metrics,'accepted':payload['per_env'][0]['accepted'],'failed_checks':payload['per_env'][0]['failed_checks']})
 lines=['# 750轮速与转矩PPO0读回','',f'模型：`{cp}`',f'SHA256：`{inputs[str(cp)]}`','',
  '四条件各1个实际/评分环境，确定性Actor；每例30秒受控，保持窗口10–18秒。PPO/optimizer更新0。不是8环境或多seed能力认证。','',
  '|vx/yaw命令|目标轮速 L/R rad/s|实际dq L/R rad/s|施加转矩 L/R Nm|实际yaw rad/s|水平速度 m/s|',
  '|---|---|---|---|---:|---:|']
 pair=lambda v:'/'.join(f'{x:+.4f}' for x in v)
 for row in records:
  m=row['metrics'];lines.append(f"|{row['command'][:2]}|{pair(m['target_mean_rad_s'])}|{pair(m['actual_joint_dq_mean_rad_s'])}|{pair(m['applied_torque_mean_nm'])}|{m['yaw_mean_rad_s']:+.4f}|{m['horizontal_mean_mps']:.4f}|")
 lines+=['','|vx/yaw命令|目标反号比例|实际dq反号比例|几何支撑点滑动 L/R m/s|双轮接触|轴向极性残差 max rad/s|',
 '|---|---:|---:|---|---:|---:|']
 for row in records:
  m=row['metrics'];lines.append(f"|{row['command'][:2]}|{m['target_opposite_sign_fraction']:.1%}|{m['actual_opposite_sign_fraction']:.1%}|{pair(m['support_slip_mean_mps'])}|{m['dual_contact_fraction']:.1%}|{m['joint_vs_relative_axial_max_abs_rad_s']:.3g}|")
 lines+=['','## 原评分与测量限制','']
 for row in records:lines.append(f"- {row['command'][:2]}：accepted={row['accepted']}，failed_checks={row['failed_checks']}")
 lines+=['','轮速为实体joint dq（模型left/right顺序），不是电机安装极性。几何滚动正方向为机体水平前向；joint正方向与其并非同义。relative axial = dot(轮体世界角速度−父腿世界角速度, 轮轴世界方向)。',
 '转矩是最后5ms PD保持值，速度是10ms末端值；pre-action几何数据另存具名字段。力矩反号不意味着轮速已经反号。轮体运动学的几何支撑点速度不是solver接触斑，不认证真机无滑移。100Hz只能检查可见变化，不排除100Hz以上振荡。',
 '未调整任何奖励/课程/物理，不依据本单环境结果授予完整能力或自动续训。详细变化、饱和、转矩符号切换次数与原始SHA见audit.json。']
 for p,h in inputs.items():
  if digest(p)!=h:raise ValueError('input changed while auditing')
 output.mkdir(parents=True,exist_ok=False)
 for name,obj in [('audit.json',{'version':VERSION,'inputs_sha256':inputs,'results':records,'PPO':0,'simulation':0,'historical_writes':0}),('audit_manifest.json',{'inputs_sha256':inputs,'review_source_sha256':digest(__file__)})]:
  with (output/name).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
 with (output/'audit.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print(output/'audit.md')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();review(a.run,a.output_dir)
