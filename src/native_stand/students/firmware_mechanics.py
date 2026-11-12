"""Explicit force separation and bounded two-inertia hypotheses for diagnostics.

No asset edits or training defaults. A40Nm motor remains a40Nm motor; external
spring and passive transmission forces are independent generalized forces.
"""
import copy

import numpy as np

from .firmware_sampling import profile,source_parameters,validate_profile,SourceExecution
from .oscillation_search import LEGS,WHEELS
from .motion_mcu_execution import WHEEL_CAP

VERSION='motion31_separated_motor_spring_v1'
ROUTES=('legacy_combined','separated')


class MechanicalEnvelopeError(ValueError):pass


class MechanicalExecution(SourceExecution):
    def step_sampled(self,*args):
        total=super().step_sampled(*args)
        self.transmitted=total-self.spring
        return total


class ForceRouter:
    """Explicit control/external inputs; readback after mj_step, before forward.

    Flex uses a virtual rotor and applies its transmitted torque at the load
    joint; rigid uses the existing actuator. Neither enlarges actuator limits.
    """
    def __init__(self,model,refs,route,flexible,*,envelope_nm=200.):
        if route not in ROUTES:raise ValueError('explicit force route required')
        if not np.isfinite(envelope_nm) or not 67<=envelope_nm<=200:raise ValueError('bounded hypothetical mechanical envelope required')
        self.model=model;self.refs=refs;self.route=route;self.flexible=bool(flexible);self.envelope=envelope_nm
        self.act=np.asarray(refs['actuators']);self.dof=np.asarray(refs['dofs']);self.gear=np.asarray(refs['gears'])
        if not np.all(model.actuator_forcelimited[self.act[LEGS]]):raise ValueError('motor force limits must stay enabled')
        effective=model.actuator_forcerange[self.act[LEGS]]*abs(self.gear[LEGS,None])
        if not np.allclose(effective,[-40,40]):raise ValueError('original leg actuator40Nm limit required')

    def apply(self,data,motor,transmitted,spring):
        arrays=[np.asarray(x,float) for x in (motor,transmitted,spring)]
        if any(x.shape!=(6,) or not np.isfinite(x).all() for x in arrays):raise ValueError('finite force6 inputs required')
        motor,transmitted,spring=arrays
        if np.any(abs(motor[LEGS])>40+1e-5) or np.any(abs(motor[WHEELS])>WHEEL_CAP+1e-5):
            raise RuntimeError('upstream motor command exceeds unchanged electrical capability')
        total=transmitted+spring
        if max(abs(total).max(),abs(transmitted).max())>self.envelope:
            raise MechanicalEnvelopeError('hypothetical passive/transmission force outside declared200Nm envelope')
        data.qfrc_applied[:]=0
        if self.route=='legacy_combined':
            ctrl=total/self.gear;bounds=self.model.actuator_ctrlrange[self.act]
            if np.any(ctrl<bounds[:,0]-1e-5) or np.any(ctrl>bounds[:,1]+1e-5):
                raise MechanicalEnvelopeError('legacy combined request outside original67Nm control envelope')
            data.ctrl[self.act]=ctrl;self.external_transmission=np.zeros(6);self.external_spring=np.zeros(6)
        else:
            actuator=motor.copy();self.external_transmission=np.zeros(6)
            if self.flexible:
                actuator[LEGS]=0;self.external_transmission[LEGS]=transmitted[LEGS]
            self.external_spring=spring.copy()
            data.ctrl[self.act]=actuator/self.gear
            data.qfrc_applied[self.dof]=self.external_transmission+self.external_spring
        self.motor=motor.copy();self.transmitted=transmitted.copy();self.spring=spring.copy()

    def readback(self,data):
        actual_act=np.asarray(data.qfrc_actuator[self.dof]).copy()
        external=np.asarray(data.qfrc_applied[self.dof]).copy()
        actual_motor=actual_act+self.external_transmission
        if self.route=='separated':
            expected=self.external_transmission+self.external_spring
            if not np.allclose(external,expected,atol=1e-9,rtol=0):raise RuntimeError('external force changed before integration')
            if not self.flexible and np.max(abs(actual_act[LEGS]-self.motor[LEGS]))>1e-6:
                raise RuntimeError('engine actuator differs from bounded rigid motor demand')
            if self.flexible and np.max(abs(actual_act[LEGS]))>1e-9:raise RuntimeError('flex motor double counted at load')
        return dict(motor_state=self.motor.copy(),transmission=self.transmitted.copy(),spring_requested=self.spring.copy(),
            actuator=actual_act,external=external,transmission_external=self.external_transmission.copy(),
            spring_external=self.external_spring.copy(),motor_at_joint=actual_motor,total=actual_act+external,
            passive=np.asarray(data.qfrc_passive[self.dof]).copy(),constraint=np.asarray(data.qfrc_constraint[self.dof]).copy(),
            bias=np.asarray(data.qfrc_bias[self.dof]).copy(),acceleration=np.asarray(data.qacc[self.dof]).copy())


def mechanical_cases():
    rows=[]
    def add(name,family,*,k=0.,b=.2,j=.006,route='separated',preload=0.,changes=None,physical=None):
        s=profile();s.update(coupling_stiffness_nm_rad=k,coupling_damping_nm_s_rad=b,rotor_inertia=j)
        s.update(changes or {});p=source_parameters();p.update(physical or {})
        rows.append(dict(name=name,family=family,parameters=p,sampling=validate_profile(s),force_route=route,
            preload_fraction=preload,mechanical_envelope_nm=200.))
    for route in ROUTES:
        add('rigid_'+route,'force_route_pair',route=route)
        add('lowgain_rigid_'+route,'force_route_pair',route=route,physical=dict(kp=20.,kv=.15))
        for k in (100.,300.):add(f'flex{k:g}_'+route,'force_route_pair',route=route,k=k)
    for j in (.003,.006,.009):
        for k in (20.,40.,80.,120.,240.,480.):
            for b in (.1,.4,1.,2.):
                add(f'J{j:g}_K{k:g}_B{b:g}','compliance_grid',j=j,k=k,b=b)
    for k in (80.,120.,240.):
        for b in (.1,.4):add(f'preload_K{k:g}_B{b:g}','initial_preload',k=k,b=b,preload=1.)
    for k in (40.,80.,120.,240.):
        for age,tau,delay in ((2,0.,0),(5,0.,0),(0,2.,0),(0,0.,2),(0,0.,5),(0,2.,2)):
            add(f'K{k:g}_age{age}_tau{tau:g}_motor{delay}','compliance_timing',k=k,b=.4,
                changes=dict(internal_dq_age_ms=age,internal_dq_tau_ms=tau),
                physical=dict(motor_delay_ms=[delay]*6))
    return rows


def followup_cases(results,limit=12,*,initial_motor=None):
    """Up to four fit-window candidates, each with three predefined contrasts.

    No holdout access, no change to the base matrix, zero repeated hypotheses.
    """
    leaders=sorted([r for r in results if r['force_route']=='separated' and
        r['sampling']['coupling_stiffness_nm_rad'] and r.get('features_fit',{}).get('coverage',0)>=.98],
        key=lambda r:r['fit']['score'])[:4]
    rows=[]
    seen={(repr(r['parameters']),repr(r['sampling']),r['force_route'],r['preload_fraction']) for r in results}
    for leader in leaders:
        for kind in ('lowgain','preload','motor_tau2'):
            row={k:copy.deepcopy(leader[k]) for k in ('parameters','sampling','force_route','preload_fraction','mechanical_envelope_nm')}
            row.update(name=kind+'__'+leader['name'],family='fit_followup',parent=leader['name'])
            if kind=='lowgain':row['parameters'].update(kp=20.,kv=.15)
            elif kind=='preload':row['preload_fraction']=1.
            else:row['parameters']['motor_tau_ms']=[2,2,0,2,2,0]
            if initial_motor is not None and row['preload_fraction']:
                twist=np.asarray(initial_motor)[LEGS]/row['sampling']['coupling_stiffness_nm_rad']
                if not np.isfinite(twist).all() or abs(twist).max()>.15:continue
            key=(repr(row['parameters']),repr(row['sampling']),row['force_route'],row['preload_fraction'])
            if key in seen:continue
            seen.add(key);rows.append(row)
            if len(rows)>=limit:return rows
    return rows


def reproduction(real,sim):
    """Strict declared waveform check, never a hardware identification result."""
    if not real.get('valid') or not sim.get('valid'):
        return dict(matched=False,reason='insufficient window',hardware_identified=False)
    ratio=np.asarray(sim['dq_std_rad_s'])/np.maximum(np.asarray(real['dq_std_rad_s']),1e-12)
    frequency=abs(np.asarray(sim['velocity_peak_hz'])-np.asarray(real['velocity_peak_hz']))
    checks=dict(full_window=sim['coverage']>=.98,
        all_four_frequencies=bool((frequency<=4).all()),
        all_four_amplitudes=bool(((ratio>=.5)&(ratio<=2)).all()),
        high_frequency_energy=np.mean(sim['band20_27_fraction'])>=.25,
        comparable_tilt=bool((abs(np.asarray(sim['tilt_mean_rad'])-np.asarray(real['tilt_mean_rad']))<=.1).all()))
    return dict(matched=all(bool(v) for v in checks.values()),checks={k:bool(v) for k,v in checks.items()},
        dq_std_ratio=ratio.tolist(),frequency_error_hz=frequency.tolist(),hardware_identified=False)
