"""核对显式.15/40输出及旧教师；默认几何加支撑偏移，不训练或采物理。"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from native_stand.students.motion_centered_output import CenteredMotionOutput, VERSION as LINEAR_VERSION, LEGS, WHEELS
from native_stand.students.motion_kinematic_output import KinematicMotionOutput, VERSION as KINEMATIC_VERSION
from native_stand.students.motion_supported_output import SupportedKinematicMotionOutput, VERSION as SUPPORTED_VERSION
from native_stand.students.motion_distillation_pipeline import validate_teacher

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/interfaces/motion31_kinematic_support_centered_015_40_v1.design.json'
SOURCE=ROOT/'artifacts/2026_11_05__motion9995_height_motion_range_r2'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def decoder_for(c):
    if c.get('action_version')==SUPPORTED_VERSION:return SupportedKinematicMotionOutput(c)
    if c.get('action_version')==KINEMATIC_VERSION:return KinematicMotionOutput(c)
    if c.get('action_version')==LINEAR_VERSION:return CenteredMotionOutput(c)
    raise ValueError('未声明支持的动作版本')


def c_header_name(c):
    if c['action_version']==SUPPORTED_VERSION:return 'motion31_supported_output_v1.h'
    return ('motion31_kinematic_output_v1.h' if c['action_version']==KINEMATIC_VERSION
            else 'motion31_height_centered_output_v1.h')


def header(c):
    formula=c['center_formula']
    def number(v):return f'{float(v):.9f}f'
    if c['action_version'] in (KINEMATIC_VERSION,SUPPORTED_VERSION):
        center_code=f'''const float upper={number(formula['equivalent_upper_m'])};
    const float lower={number(formula['equivalent_lower_m'])};
    const float L=h+{number(formula['hip_above_root_m'])}-{number(formula['wheel_radius_m'])};
    if(L<=0 || L<fabsf(upper-lower) || L>upper+lower) return -1;
    const float cosine=(L*L+upper*upper-lower*lower)/(2.0f*L*upper);
    if(!isfinite(cosine) || cosine < -1.0f || cosine > 1.0f) return -1;
    const float beta=acosf(cosine);
    ctx->center[0]=beta-{number(np.deg2rad(formula['hip_zero_deg']))};
    ctx->center[1]=beta-{number(np.deg2rad(formula['active_zero_deg']))};'''
        if c['action_version']==SUPPORTED_VERSION:
            offset=c['support_target_offset_rad']
            center_code+=f'''\n    ctx->center[0]+={number(offset[0])};
    ctx->center[1]+={number(offset[1])};'''
    else:
        center_code=f'''const float d={number(formula['slope_rad_per_m'])}*(h-{number(formula['reference_height_m'])});
    ctx->center[0]={number(formula['hip_at_reference_rad'])}+d;
    ctx->center[1]={number(formula['active_at_reference_rad'])}+d;'''
    guard=c_header_name(c).upper().replace('.','_')
    return f'''/* 新动作设计的CPU参考，尚非部署包；不含Actor、PD、补偿或电机输出。 */
#ifndef {guard}
#define {guard}
#include <math.h>
#define MOTION31_NEW_ACTION_VERSION "{c['action_version']}"
typedef struct {{float height_m; float center[4]; int valid;}} Motion31HeightOutput;
static inline int Motion31HeightOutputInit(Motion31HeightOutput *ctx, float h) {{
    if(!ctx) return -1;
    ctx->valid=0;
    if(!isfinite(h) || h<0.30f || h>0.40f) return -1;
    {center_code}
    ctx->height_m=h;
    ctx->center[2]=-ctx->center[0];
    ctx->center[3]=ctx->center[1];
    ctx->valid=1;
    return 0;
}}
static inline int Motion31HeightOutputCheckCommand(const Motion31HeightOutput *ctx, float h) {{
    return ctx && ctx->valid && isfinite(h) && fabsf(h-ctx->height_m)<=0.0000005f ? 0 : -1;
}}
static inline int Motion31HeightOutputDecode(const Motion31HeightOutput *ctx,
                                             const float raw[6], float requested[6]) {{
    if(!ctx || !ctx->valid || !raw || !requested) return -1;
    for(int j=0;j<6;++j) if(!isfinite(raw[j])) return -1;
    const int legs[4]={{0,1,3,4}};
    for(int j=0;j<6;++j) {{
        const float a=fminf(fmaxf(raw[j],-1.0f),1.0f);
        requested[j]=(j==2 || j==5)?40.0f*a:0.15f*a;
    }}
    for(int j=0;j<4;++j) requested[legs[j]]+=ctx->center[j];
    return 0;
}}
#endif
'''


def c_check(out,c,decoder):
    hp=out/c_header_name(c);hp.write_text(header(c))
    wrapper=out/'decoder_cpu.c'
    wrapper.write_text(f'#include "{hp.name}"\n'
       'int decode(float h,const float *a,float *t){Motion31HeightOutput c;'
       'if(Motion31HeightOutputInit(&c,h))return -1;return Motion31HeightOutputDecode(&c,a,t);}\n'
       'int changed_height(float a,float b){Motion31HeightOutput c;'
       'if(Motion31HeightOutputInit(&c,a))return -1;return Motion31HeightOutputCheckCommand(&c,b);}\n')
    so=out/'decoder_cpu.so'
    command=['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',str(wrapper),'-o',str(so),'-lm']
    completed=subprocess.run(command,capture_output=True,text=True,check=True)
    library=ctypes.CDLL(str(so));ptr=ctypes.POINTER(ctypes.c_float)
    library.decode.argtypes=[ctypes.c_float,ptr,ptr];library.decode.restype=ctypes.c_int
    library.changed_height.argtypes=[ctypes.c_float,ctypes.c_float];library.changed_height.restype=ctypes.c_int
    rng=np.random.default_rng(20261105)
    heights=np.r_[np.linspace(.30,.40,257),[.30,.32,.34,.36,.38,.40,.33,.35,.37,.39]].astype('float32')
    raw=rng.uniform(-2,2,(len(heights),6)).astype('float32')
    raw[:4]=np.array([[0]*6,[1]*6,[-1]*6,[2,-2,.5,.25,-.25,-2]],dtype='float32')
    expected=decoder.bind_height(torch.tensor(heights)).decode(torch.tensor(raw))['requested_target'].numpy()
    actual=np.zeros_like(expected)
    for i in range(len(heights)):
        status=library.decode(float(heights[i]),raw[i].ctypes.data_as(ptr),actual[i].ctypes.data_as(ptr))
        if status!=0:raise ValueError('C原型拒绝合法golden')
    error=np.abs(actual-expected)
    if error[:,LEGS].max()>2e-6 or error[:,WHEELS].max()>1e-5:raise ValueError('Python/C解码不同')
    invalid=[]
    for height in [.299,.401,float('nan')]:
        target=np.full(6,123,dtype='float32');before=target.copy()
        result=library.decode(height,raw[0].ctypes.data_as(ptr),target.ctypes.data_as(ptr))
        if result!=-1 or not np.array_equal(before,target):raise ValueError('C非法高度未拒绝或污染目标')
        invalid.append(str(height))
    for value in [float('nan'),float('inf')]:
        action=raw[0].copy();action[3]=value;target=np.full(6,123,dtype='float32')
        status=library.decode(.34,action.ctypes.data_as(ptr),target.ctypes.data_as(ptr))
        if status!=-1 or not (target==123).all():raise ValueError('C非法动作未拒绝或污染目标')
    if library.changed_height(.34,.35)!=-1 or library.changed_height(.34,.34)!=0:
        raise ValueError('C回合高度绑定不一致')
    np.savez_compressed(out/'decoder_golden.npz',height_m=heights,raw_action=raw,requested_target=expected,c_requested=actual)
    return dict(compiler_command=command,compiler_stdout=completed.stdout,compiler_stderr=completed.stderr,
                golden_samples=len(heights),max_error_rad=float(error[:,LEGS].max()),
                max_error_rad_s=float(error[:,WHEELS].max()),invalid_heights_rejected=invalid,
                in_episode_height_change_rejected=True,nonfinite_actions_rejected_without_writes=True,
                scope='仅新动作解码C主机计算；不是STM32网络或电机执行测试')


def select_teacher(c,inputs):
    manifest_path=ROOT/'artifacts/2026_10_28-13_45__motion9000_cross_engine_wide/evaluation_framefix/manifest.json'
    old=read(manifest_path);inputs[str(manifest_path)]=sha(manifest_path)
    matrix_path=Path(c['distillation_teacher_recommendation']['checkpoint']).parent/'evaluation_matrix_360.json'
    matrix_path=ROOT/matrix_path;inputs[str(matrix_path)]=sha(matrix_path)
    native=[];digest=c['distillation_teacher_recommendation']['sha256']
    for row in read(matrix_path):
        path=matrix_path.parent/row['result'];e=read(path);inputs[str(path)]=sha(path)
        if e['checkpoint_sha256']!=digest or e['scored_envs']!=8:
            raise ValueError('原Native矩阵模型或环境数不符')
        native.append(dict(case_id=row['case_id'],command=e['case'].get('command'),
                           passed_envs=e['passed_envs'],scored_envs=e['scored_envs'],accepted=e['accepted']))
    candidates=[]
    extra=read(ROOT/'artifacts/2026_11_4-15_41__motion600_855_dq_ablation/models.json')
    models=list(old['models'])+[dict(name=k,**extra[k]) for k in ['teacher600','teacher855']]
    models.append(dict(name='teacher1900',checkpoint=str(ROOT/'runs/2026_11_4-01_29__motion15_transient_catch863_remainder/formal/model_update_1900.pt'),sha256='985ee947b1f69e0479fa0b393a93d441964ffe8fdc9efca3b9f2c6f8411ef6fb'))
    for row in models:
        p=Path(row['checkpoint']);exists=p.exists()
        if exists:
            actual=sha(p)
            if actual!=row['sha256']:raise ValueError('候选模型SHA不同：'+str(p))
            inputs[str(p)]=actual
        candidates.append(dict(name=row['name'],checkpoint=str(p),expected_sha256=row['sha256'],exists=exists,
                               current_sha_matches=(True if exists else None)))
    return dict(primary='u9995',candidates=candidates,
                reason='同31D、六高度旧证据和现存完整权重；以不同固定高度运动为主需求',
                Native_original=dict(accepted_cases=sum(r['accepted'] for r in native),cases=len(native),
                                     passed_envs=sum(r['passed_envs'] for r in native),scored_envs=296,rows=native),
                source_not_full_recovery=True,source_not_hardware_qualified=True,
                excluded9000='原路径缺权重，旧同MuJoCo多个高度失败；保留历史，不伪造恢复',
                contrast_only=['source4500原.34运动','teacher600实际上机反馈/旧控制行为',
                               'teacher855固定.34支撑与受扰','teacher1900固定.34减振但有刚性回归'],
                mixed_teacher_labels_enabled=False,
                distillation_goal='新输出可表示范围内保留旧物理目标；新范围外明确报告，随后独立任务学习',
                old_Adam_or_Critic_copied=False)


def run(out,config_path=CONFIG):
    start=time.monotonic();torch.set_num_threads(1)
    if out.exists():raise FileExistsError(out)
    config_path=config_path.resolve();c=read(config_path);decoder=decoder_for(c)
    kinematic=c['action_version']==KINEMATIC_VERSION
    supported=c['action_version']==SUPPORTED_VERSION
    geometric=kinematic or supported
    inputs={str(config_path):sha(config_path)}
    for p in [ROOT/'src/native_stand/students/motion_centered_output.py',
              ROOT/'src/native_stand/students/motion_kinematic_output.py',Path(__file__),
              ROOT/'src/native_stand/students/motion_supported_output.py',
              SOURCE/'results.json',SOURCE/'range_analysis.json',ROOT/c['center_source']]:
        inputs[str(p)]=sha(p)
    bind=c['distillation_teacher_recommendation'];cp=ROOT/bind['checkpoint'];spec=ROOT/bind['parent_spec']
    if sha(cp)!=bind['sha256'] or sha(spec)!=bind['parent_spec_sha256']:
        raise ValueError('所选9995来源身份不同')
    inputs[str(cp)]=sha(cp);inputs[str(spec)]=sha(spec)
    teacher,saved,parent=validate_teacher(cp,spec)
    selection=select_teacher(c,inputs)
    if kinematic:
        selection['primary']='u9995_COMPARISON_ONLY_UNDER_NEW_GEOMETRIC_RANGE'
        selection['distillation_goal']='纯几何中心的旧稳态目标不可表示；不保证小模型复制，须先验证承重并另定学习方案'
    elif supported:
        selection['primary']='u9995_SUPPORT_SHIFTED_OUTPUT_CANDIDATE'
        selection['distillation_goal']='支撑目标参考移至网络0附近；可表示标签按物理目标蒸馏，启动越界与新闭环能力分别验证'
    ranges=read(SOURCE/'range_analysis.json');heights=np.array([r['height_command_m'] for r in ranges['by_height']])
    old_centers=np.array([r['center_proposal_rad'] for r in ranges['by_height']])
    proposed=decoder.centers(torch.tensor(heights,dtype=torch.float64)).numpy();fit_error=proposed-old_centers
    if not geometric and abs(fit_error).max()>.003:raise ValueError('线性拟合误差超过本设计.003rad约束')
    fit=dict(reference_source=str(SOURCE/'range_analysis.json'),height_m=heights.tolist(),
             table_centers_rad=old_centers.tolist(),formula_centers_rad=proposed.tolist(),
             max_error_rad=float(abs(fit_error).max()),max_error_deg=float(np.rad2deg(abs(fit_error)).max()),
             corresponding_Kp40_request_change_nm=float(abs(fit_error).max()*40),
             formula_coefficients=c['center_formula'],
             support_target_offset_rad=c.get('support_target_offset_rad'),
             comparison_kind=('几何加固定支撑偏移与六高度旧目标比较' if supported else
                              '几何角与旧承重PD目标不同，不以此误差作为几何准确度' if kinematic else '线性拟合旧PD目标'),
             limitation=('支撑偏移按原MCU的.34m旧目标估计；不是实测负载，也不保证所有高度／动态稳定' if supported else
                         '纯几何中心且支撑偏移0；旧目标的范围覆盖另报，不能假称直接保真蒸馏' if kinematic
                         else '只拟合原六高度目标参考；不是真实运动学或中间高度控制认证'))
    summaries=[];max_inference=0.;max_decode=0.;max_reinferred_physical=0.;records=0
    with torch.inference_mode():
        for row in read(SOURCE/'results.json'):
            path=SOURCE/row['trace']['path']
            if sha(path)!=row['trace']['sha256']:raise ValueError('原轨迹身份不同')
            inputs[str(path)]=row['trace']['sha256']
            with np.load(path) as npz:
                x=torch.tensor(npz['actor_input']);logged=torch.tensor(npz['mean']);target=torch.tensor(npz['requested_target']);t=npz['time'].copy()
            mean=torch.cat([teacher.mean(part) for part in x.split(1024)])
            inference=float((mean-logged).abs().max());max_inference=max(max_inference,inference)
            scales=mean.new_tensor([1.5,1.5,100.,1.5,1.5,100.])
            old_physical=mean*scales
            # 同帧保存的mean与保存目标验证解码；批量重推理舍入另报告，不能混用rad/rad_s阈值。
            decode_error=float((logged*scales-target).abs().max());max_decode=max(max_decode,decode_error)
            max_reinferred_physical=max(max_reinferred_physical,float((old_physical-target).abs().max()))
            if inference>2e-5 or decode_error>2e-5:raise ValueError('旧教师推理或物理目标不符')
            h=row['case']['command'][2];bound=decoder.bind_height(torch.full((len(x),),float(h)))
            bound.validate_actor_input(x);labels=bound.labels(old_physical)
            exact=labels['exactly_representable'].numpy();raw=labels['raw_label'].numpy()
            windows={}
            for name,lo,hi in [('all',0.,30.),('initial0_0p5',0.,.5),('catch0p5_2',.5,2.),
                               ('ramp2_10',2.,10.),('hold10_18',10.,18.),('brake18_26',18.,26.),('tail26_30',26.,30.)]:
                mask=(t>lo+1e-8)&(t<=hi+1e-8)
                windows[name]=dict(samples=int(mask.sum()),exact_frames=int(exact[mask].sum()),
                                   exact_fraction=float(exact[mask].mean()),max_label_magnitude=float(abs(raw[mask]).max()),
                                   max_required_leg_offset_rad=float(abs(raw[mask][:,LEGS]).max()*.15))
            summaries.append(dict(case_id=row['case']['case_id'],command=row['case']['command'],
                                   old_quality_pass=row['assessment']['accepted'],windows=windows))
            records+=len(x)
    out.mkdir(parents=True,exist_ok=False)
    c_result=c_check(out,c,decoder)
    totals={name:dict(samples=sum(s['windows'][name]['samples'] for s in summaries),
                     exact_frames=sum(s['windows'][name]['exact_frames'] for s in summaries))
            for name in summaries[0]['windows']}
    for name,v in totals.items():
        v['exact_fraction']=v['exact_frames']/v['samples']
        v['max_required_leg_offset_rad']=max(s['windows'][name]['max_required_leg_offset_rad'] for s in summaries)
    unchanged={p:sha(Path(p))==d for p,d in inputs.items()}
    if not all(unchanged.values()):raise ValueError('只读输入变化')
    result=dict(status='CPU_PROTOTYPE_VERIFIED_NOT_TRAINED',action_version=c['action_version'],config=str(config_path),
       config_sha256=sha(config_path),teacher_selection=selection,center_comparison=fit,
       actual_teacher_inference_records=records,max_teacher_action_error=max_inference,
       max_old_decode_error=max_decode,max_reinferred_physical_difference=max_reinferred_physical,
       reinference_scope='动作在原归一化单位比较；轮物理目标的批量舍入差单独报告，不与腿rad混用',
       coverage=totals,per_case=summaries,C_decoder=c_result,
       protected_inputs_unchanged=unchanged,physical_steps=0,BC_updates=0,PPO_updates=0,
       firmware_modified=False,runtime_recipe_ready=False,deployment_ready=False,
       actual_wall_seconds=time.monotonic()-start)
    write(out/'analysis.json',result);write(out/'teacher_selection.json',selection)
    write(out/('center_comparison.json' if geometric else 'center_fit.json'),fit)
    example=decoder.bind_height(torch.tensor([.34])).decode(torch.zeros(1,6))['requested_target'][0].tolist()
    lines=['# 新Motion31输出：高度公式＋腿.15rad／轮40rad/s','',
           ('用户选择把支撑目标作为输出参考，已实现“几何角＋明确支撑偏移＋网络调整”。只验证CPU计算和旧目标范围，未蒸馏、未训练、未采物理，也不是部署包。' if supported else
            '接口设计和CPU原型已完成，未蒸馏、未训练、未采物理，也不是部署包。用户选择纯运动学中心，支撑偏移0；9995保留作不同固定高度来源对照，不能在新范围内保证复制它。' if kinematic else
            '接口设计和CPU原型已完成，未蒸馏、未训练、未采物理，也不是部署包。推荐旧9995作为不同固定高度的主教师；855／1900只保留为固定.34m对照。'),
           '', '输入继续原31D，部署Actor目标仍31→128→64→32→6、ELU/Tanh。六通道mean的数学维数不变，物理输出版本独立。',
           '', '```c','// h是目标高度，单位m；先核验0.30 <= h <= 0.40。每次运动开始前算一次。',
           *(['float L = h + 0.000024f - 0.060f;',
              'float beta = acosf((L*L + 0.21f*0.21f - 0.25f*0.25f) / (2.0f*L*0.21f));',
              'float c_hip = beta - 47.1033888f * (3.14159265358979323846f/180.0f);',
              'float c_active = beta - 47.1080837f * (3.14159265358979323846f/180.0f);'] if geometric else
             ['float d = 3.23492f * (0.34f - h);',
              'float c_hip = d - 0.02994f;', 'float c_active = d - 0.09317f;']),
           *([f'c_hip += {c["support_target_offset_rad"][0]:.12f}f;',
              f'c_active += {c["support_target_offset_rad"][1]:.12f}f;'] if supported else []),
           '// a[0..5]是已有限性核验、裁到[-1,1]的网络输出。',
           'target[0] =  c_hip    + 0.15f * a[0];',
           'target[1] =  c_active + 0.15f * a[1];',
           'target[2] =             40.0f * a[2];',
           'target[3] = -c_hip    + 0.15f * a[3];',
           'target[4] =  c_active + 0.15f * a[4];',
           'target[5] =             40.0f * a[5];','```','',
           (f'固定支撑偏移来自.34m旧对称目标减几何角，四腿为{c["support_target_offset_rad"]}rad；不是编码器零位，也不是额外弹簧／力矩补偿。各高度共用常数，与原六高度中心最大差{fit["max_error_rad"]:.6f}rad。该偏移尚非实测负载，仍需完整V4和自由站立验证。' if supported else
            f'纯几何中心与旧六高度PD目标中心最大差{fit["max_error_rad"]:.6f}rad；这不是运动学误差。支撑偏移明确为0，没有隐藏补偿。公式保持原模型零位，h是root高度命令。不能只换旧模型的decoder后直接部署。' if kinematic else
            f'原六高度参考表最大拟合误差{fit["max_error_rad"]:.6f}rad，即{fit["max_error_deg"]:.3f}°；Kp40对应约{fit["corresponding_Kp40_request_change_nm"]:.3f}Nm的P项差。无需电控逐高度改代码。公式不是逆运动学，也不改变实体零位；实际高度不是这里的h。'),
           '', f'0.34m且a全0时，六目标为{example}。四腿单位rad，轮单位rad/s；a全0表示参考目标，不表示电机力矩0。',
           '', '## 旧教师与来源边界','',
           f'主教师：`{cp}`，SHA `{bind["sha256"]}`。原Native最终37条件中36条件完整通过，共288/296环境；未过为+yaw4，且多高度主要验证站立，并非全高度所有运动。已有同MuJoCo60条件为16/60通过，本次范围34条件为7/34通过。不能称它已全面通过或已能抗实机振荡。',
           '', '原9000路径不存在且旧MuJoCo多高度不如9995；4500主要固定.34m；600／855／1900的当前完整V4、.34m抗振证据不能自动授予它们其他高度专家身份。新训练不强制复制旧教师的漂移、抖动或未达标速度。',
           '', '## 实际旧目标在新范围中的覆盖','',
           '| 时间段 | 可准确表示／全部记录 | 覆盖率 |','|---|---:|---:|']
    for name,v in totals.items():lines.append(f'| {name} | {v["exact_frames"]}/{v["samples"]} | {v["exact_fraction"]*100:.4f}% |')
    lines+=['','覆盖率针对同一旧物理目标的标签变换，不是新模型控制效果。越界标签原样记录并标记；没有裁成±1后当作准确蒸馏。新动作不保证复制全部旧启动行为。',
       '', '## 后续蒸馏和微调的具体顺序','',
       ('1. 保持显式支撑偏移，按新物理中心转换标签。即使恒速目标能表示，也须单列启动越界与受扰余量，不能当作新模型已学会。' if supported else
        '1. 纯几何中心±.15rad不能复制本批旧教师的稳态目标。先保留9995作不同高度比较来源；不能启动“保证复刻”的蒸馏或把越界标签裁小。若继续本接口，须独立验证承重能力，并声明新的任务学习方案。' if kinematic else
        '1. 使用这套新动作解码，冻结9995作为唯一初始教师；按物理目标换标签，不能继续用旧mean作为新mean。'),
       '2. 复用原31D蒸馏的实际输入、整episode划分和学生状态聚合。范围外帧单独保存，不能悄悄裁剪、删掉并称完整保真。来源诊断轨迹不直接冒充新任务独立验证或留出数据。',
       '3. 优先让小模型继承可表示的站立与不同高度低速行为；用新接口自主控制检查启动、刹停、连续交接、腿长／摆动受扰、实体振荡和双轮支撑。',
       '4. 随后按单独声明的任务PPO学习新范围下的纠偏和旧教师未掌握的运动。物理std要按新.15／40重新换算，不能继承旧归一化std而不说明。旧Critic和PPO Adam不由蒸馏自动恢复。',
       '5. 相同Plant比较Teacher／新Student，但不把“降低全部dq响应”当目标。保留真实dq和PD-D；新策略仍需有用的低频纠偏。',
       '', '当前公共fit仍按旧1.5／100的loss和标签解释运行，本原型没有覆盖它。真正蒸馏前须显式接入新动作版本及物理损失，训练、MuJoCo、ONNX元数据和电控保持同一公式；不能直接调用旧fit或给旧ONNX套新scale。',
       '', f'CPU重新执行教师{records}条输入，最大动作差{max_inference:.3g}；C原型{c_result["golden_samples"]}组解码与Python最大腿差{c_result["max_error_rad"]:.3g}rad、轮差{c_result["max_error_rad_s"]:.3g}rad/s。非法高度及本回合切高均拒绝。新C只计算目标，不证明STM32网络或实际电机行为。',
       '', 'previous继续来自机械保护后实际采用的完整物理目标，腿除1.5、轮除100；不除.15／40，不写未采用的预测。每个回合绑定一个高度，结束后才允许选另一高度；中间高度公式连续，但其控制效果尚未测试。',
       '', '新动作尚未注册为公共训练默认，未修改旧权重、旧spec、固件或任何运行。工程故障保护和原40Nm／约2.447Nm能力不变。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],output=str(out),fit=fit,
       records=records,max_teacher_action_error=max_inference,coverage=totals,C_decoder=c_result,
       wall_seconds=result['actual_wall_seconds']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=CONFIG,help='明确选择动作版本；默认几何中心加显式支撑偏移')
    args=p.parse_args();run(args.output_dir,args.config)
