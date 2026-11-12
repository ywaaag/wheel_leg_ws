"""Explicit PPO0 multi-checkpoint matrix; same Recovery64 runtime and protocol."""
VERSION='student_recovery64_checkpoint_validation_v1'
FAST_VERSION='student_recovery64_parallel_native_probe_v1'


def validate(spec):
    if not spec.get('evaluation_only') or any(spec['budget'].get(k,0) for k in ('smoke_updates','formal_updates','sample_limit')):
        raise ValueError('Checkpoint validation is strictly PPO0')
    if spec['student_training']['version'] not in ('student_recovery_broad_narrow_ppo_v1','student_recovery_stop_ppo_v2'):
        raise ValueError('Checkpoint validation requires explicit Recovery64 recipe')
    cases=spec['student_evaluation']['cases']
    if spec['student_evaluation']['version']==FAST_VERSION:
        if not 1<=len(cases)<=8 or spec['evaluation']['envs']!=len(cases) or spec['evaluation']['scored_envs']!=1:
            raise ValueError('Native parallel probe requires1..8 models and exactly one physical env per model')
        if len({r['checkpoint'] for r in cases})!=len(cases):raise ValueError('duplicate probe model')
        for r in cases:
            if (r['kind']!='student' or r['role']!='RecoveryStudent' or r['family']!='swing_candidate_from_historical_r1'
                or r['yaw_rad']!=0. or r['seed']!=20261010 or r.get('command_height_m')!=.34):
                raise ValueError('Native probe requires the same native Swing/seed/zero command')
        return
    if not cases or len(cases)>60 or len(cases)%4:
        raise ValueError('Checkpoint validation requires at most15 complete four-case groups')
    seen=set()
    for i in range(0,len(cases),4):
        group=cases[i:i+4]
        identity={(r['checkpoint'],r['sha256']) for r in group}
        if len(identity)!=1 or next(iter(identity)) in seen:
            raise ValueError('Duplicate/mixed checkpoint group')
        seen|=identity
        expected=[(20261010,0.),(20261010,.01),(20261011,.01),(20261012,.01)]
        if [(r['seed'],r['yaw_rad']) for r in group]!=expected:
            raise ValueError('Checkpoint validation condition matrix differs')
        if any(r['role']!='RecoveryStudent' or r['kind']!='student' or
               r['family']!='swing_candidate_from_historical_r1' or r.get('command_height_m')!=.34
               for r in group):
            raise ValueError('Checkpoint validation student/command/family differs')
