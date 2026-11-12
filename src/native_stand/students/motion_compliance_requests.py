"""File/identity boundary for the explicit CPU compliance PPO owner."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from .motion_compliance_training import VERSION,CRITIC,declaration

ROOT=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def identity_files(parent,library):
    import xml.etree.ElementTree as ET
    source=json.loads(Path(parent).read_text());xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')
    files=[Path(parent),Path(library),xml,Path(source['asset_path']),Path(source['colleague_config_path'])]
    tree=ET.parse(xml);meshdir=tree.getroot().find('compiler').get('meshdir','')
    files += [xml.parent/meshdir/m.get('file') for m in tree.findall('./asset/mesh')]
    files += [ROOT/'src/native_stand/students'/n for n in (
        'motion_compliance_training.py','motion_compliance_reward.py','motion_compliance_env.py',
        'motion_compliance_eval.py','motion_compliance_owner.py','motion_compliance_requests.py',
        'firmware_sampling.py','firmware_mechanics.py','firmware_control.py','oscillation_search.py',
        'motion_tracking_reward.py','motion_quality.py','motion_yaw_quality.py','motion_differential.py',
        'motion_sensor_transfer.py','motion_antioscillation.py','motion_posture_design.py','model.py')]
    files += [ROOT/'src/native_stand/recovery/controlled_ppo.py']
    files += [ROOT/'scripts'/n for n in ('validate_native_mujoco.py','motion_mcu_playback.py','compare_motion_timing_profiles.py','search_motion31_oscillation.py','audit_motion31_electrical_cpu.py')]
    return {str(p.resolve()):sha(p) for p in files}


def build(plan,*,experiment,library,session):
    d=declaration(updates=3000)
    s=dict(recipe_version=VERSION,backend='MuJoCo_CPU',experiment=experiment,
        source_checkpoint=plan['source_checkpoint'],source_sha256=plan['source_sha256'],
        source_ONNX=plan['source_ONNX'],source_ONNX_sha256=plan['source_ONNX_sha256'],
        parent_spec=plan['parent_spec'],parent_spec_sha256=plan['parent_spec_sha256'],
        source_validation=plan['source_validation'],firmware_library=str(Path(library).resolve()),
        xml='/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml',
        interface_version='multiskill_mcu_v2',critic_privileged_version=CRITIC,
        PD=d['PD'],ppo=d['algorithm'],seed=20261029,
        sampling=dict(envs=8,steps=128,start_stage=0,profiles=d['profiles']),
        budget=dict(smoke_updates=12,formal_updates=2988,online_updates=3000,sample_limit=3072000,
            physical_steps_1ms=160002000,chain_wall_s=None),
        artifacts=dict(storage_max_bytes=1073741824,terminal_reserve_bytes=67108864),
        evaluation=d['evaluation'],authorization=dict(status='UNAUTHORIZED',hapi_session=session,
            scope='user2026-10-29 independent3000 cumulative PPO updates; unlimited training wall; CPU MuJoCo'),
        resources=dict(backend='CPU',sampler_processes=8,sampler_threads=1,learner_threads=4),
        notifications=dict(enabled=False),runtime_recipe_ready=True)
    s['inputs']=identity_files(s['parent_spec'],library)
    s['inputs'].update({s['source_checkpoint']:s['source_sha256'],s['source_ONNX']:s['source_ONNX_sha256']})
    return s


def verify(path,*,authorized=True):
    path=Path(path).resolve();s=json.loads(path.read_text())
    if (s.get('recipe_version')!=VERSION or s.get('backend')!='MuJoCo_CPU'
        or s['critic_privileged_version']!=CRITIC or s['budget']['online_updates']!=3000
        or s['budget']['smoke_updates']+s['budget']['formal_updates']!=3000
        or s['budget']['chain_wall_s'] is not None or s['sampling']['envs']!=8):
        raise ValueError('compliance recipe/budget semantics differ')
    if s['PD']!={'leg_Kp':20.,'leg_Kd':2.,'wheel_Kv':.15}:raise ValueError('compliance PD differs')
    if authorized and s['authorization']['status']!='AUTHORIZED':raise ValueError('new compliance budget unauthorized')
    if s['authorization']['hapi_session']!=os.environ.get('HAPI_SESSION_ID'):raise ValueError('compliance owner session differs')
    for p,digest in s['inputs'].items():
        if sha(p)!=digest:raise ValueError('compliance input changed: '+p)
    return s


def storage_estimate(spec,phase):
    # ~62 full states (<1MiB each), scalar matrices and bounded diagnostic tails.
    return dict(schema_version='motion_compliance_storage_v1',phase=phase,
        estimated_artifact_bytes=360*2**20,atomic_temp_peak_bytes=8*2**20,
        terminal_reserve_bytes=spec['artifacts']['terminal_reserve_bytes'])


def reserve(bookkeeper,attempt_id,phase,output):
    """Shared locked ledger, explicit backend reservation instead of GPU gates."""
    spec=verify(bookkeeper.path.with_name('spec.json'))
    from native_stand.command_v6.tools.storage_budget import check
    if check(spec,output,phase)['status']!='PASS':raise ValueError('compliance storage capacity insufficient')
    output=str(Path(output).resolve())
    def mutate(book):
        if book.get('spec_sha256')!=sha(bookkeeper.path.with_name('spec.json')):raise ValueError('compliance published spec changed')
        if book['authorization']!='AUTHORIZED' or phase not in book['limits'] or attempt_id in book['attempts']:
            raise ValueError('invalid compliance reservation')
        if any(r['status']=='RUNNING' or r['output_dir']==output for r in book['attempts'].values()):raise ValueError('compliance owner/phase already reserved')
        if book['used'][phase]>=book['limits'][phase]:raise ValueError('compliance phase exhausted')
        if phase=='formal':
            certificate=Path(spec['run'])/'smoke/wiring_certificate.json'
            if not certificate.is_file() or json.loads(certificate.read_text())['status']!='PASSED':
                raise ValueError('compliance engineering smoke required')
        book['attempts'][attempt_id]=dict(phase=phase,output_dir=output,status='RUNNING',updates=0,started_unix_s=time.time())
    bookkeeper._change(mutate)
