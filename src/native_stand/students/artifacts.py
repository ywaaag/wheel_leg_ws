"""Atomic student evidence writes; bounded by frozen declarations, not date paths."""
from native_stand.command_v6.tools.artifact_io import atomic,npz,checkpoint


def storage(spec):
    maximum=spec['student_training']['storage_max_bytes']
    declared=spec.get('artifacts',{}).get('storage_max_bytes')
    if declared is not None and declared!=maximum:raise ValueError('conflicting student/artifacts storage ceilings')
    case_maximum=spec['student_evaluation']['max_output_bytes']
    # Keep the original 128MiB tail reserve, now explicitly declared.
    reserve=spec.get('artifacts',{}).get('terminal_reserve_bytes',128*2**20)
    if any(type(v) is not int or v<=0 for v in (maximum,case_maximum,reserve)) or reserve>=maximum:
        raise ValueError('Invalid student declared storage limits')
    reboot=spec.get('motion_training',{}).get('wiring_reuse',{})
    if reboot.get('version')=='motion_symmetry85_boot_loss_recovery_v1':
        from pathlib import Path
        parent=Path(reboot['parent_run'])
        maximum-=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file() and not p.is_symlink())
        if maximum<=reserve:raise ValueError('boot parent plus successor output ceiling exhausted')
    if spec.get('student_gpu_resume'):
        from pathlib import Path
        parent=Path(spec['student_gpu_resume']['parent_run'])
        maximum-=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file() and not p.is_symlink())
        if maximum<=reserve:raise ValueError('parent plus successor output ceiling exhausted')
    upgrade=spec.get('motion_timing',{}).get('critic_upgrade')
    if upgrade:
        from pathlib import Path
        parent=Path(upgrade['parent_run'])
        maximum-=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file() and not p.is_symlink())
        if maximum<=reserve:raise ValueError('Critic upgrade parent plus child storage exhausted')
    recovery=spec.get('motion_standing_delay',{}).get('preppo_recovery')
    if recovery:
        from pathlib import Path
        parent=Path(recovery['parent_run'])
        maximum-=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file() and not p.is_symlink())
        if maximum<=reserve:raise ValueError('standing delay failed parent plus successor storage exhausted')
    return {'chain_max_bytes':maximum,'evaluation_max_bytes':case_maximum,'terminal_reserve_bytes':reserve}


def secondary(error,name,operation):
    """A failed evidence save is diagnostic information, never the primary fault."""
    try:return operation()
    except BaseException as failure:
        error.add_note(name+' also failed: '+repr(failure))
        return None
