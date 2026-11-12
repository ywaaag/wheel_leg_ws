"""Offline search semantics and host-C execution for oscillation diagnostics.

No simulator is imported here. A matching waveform is a hypothesis, never a
hardware parameter identification or a policy capability certificate.
"""
import copy
import ctypes as C
import itertools

import numpy as np
from scipy import signal

from .firmware_control import FirmwareControl, FP
from .motion_mcu_execution import WHEEL_CAP
from .real2sim_delay import TargetTransport, torque_response

LEGS = [0, 1, 3, 4]
WHEELS = [2, 5]
VERSION = 'motion31_closed_loop_oscillation_search_v1'


def dynamic_nominal():
    return dict(dq_extra_age_ms=0,dq_filter_tau_ms=0.,dq_gain=1.,
                dq_noise_std_rad_s=0.,dq_noise_seed=261029,pd_period_ms=5,
                motor_slew_nm_per_s=0.)


def validate_dynamics(value):
    if value is None:return dynamic_nominal()
    p=copy.deepcopy(value)
    if set(p)!=set(dynamic_nominal()):raise ValueError('explicit dynamic fields required')
    ranges=dict(dq_extra_age_ms=(0,20),dq_filter_tau_ms=(0,20),dq_gain=(.5,2),
                dq_noise_std_rad_s=(0,1),motor_slew_nm_per_s=(0,10000))
    for key,(lo,hi) in ranges.items():
        if not np.isfinite(p[key]) or not lo<=p[key]<=hi:raise ValueError('dynamic bound: '+key)
    if p['dq_extra_age_ms']!=int(p['dq_extra_age_ms']):raise ValueError('integer velocity age required')
    if p['pd_period_ms'] not in (5,10):raise ValueError('diagnostic PD period5/10ms only')
    if not isinstance(p['dq_noise_seed'],int) or p['dq_noise_seed']<0:raise ValueError('seed integer required')
    return p


class VelocityFeedback:
    """Explicit shared200Hz velocity packet, before Actor/PD history selection.

    Positions and IMU remain current. Additional dq age/filter/gain/noise are
    declared measurement hypotheses. Noise has a private stream and disabled
    settings consume no random draws. Default is exactly the original feedback.
    """
    def __init__(self,dynamics,initial):
        from .real2sim_delay import ObservationHistory
        self.p=validate_dynamics(dynamics)
        self.history=ObservationHistory(initial,initial_tick=-32)
        self.filtered=np.asarray(initial[12:18],float).copy()
        self.packet=self.filtered.copy();self.rng=np.random.default_rng(self.p['dq_noise_seed'])
        self.last_tick=-1
        self.enabled=any(self.p[k]!=dynamic_nominal()[k] for k in
                         ('dq_extra_age_ms','dq_filter_tau_ms','dq_gain','dq_noise_std_rad_s'))

    def step(self,tick,sensors):
        if tick!=self.last_tick+1:raise ValueError('velocity feedback needs contiguous1ms ticks')
        self.last_tick=tick
        s=np.asarray(sensors,float).copy()
        if not self.enabled:return s
        self.history.append(tick,s)
        value=self.history.at(tick-int(self.p['dq_extra_age_ms']))[12:18]
        # First-order sensor filter runs each1ms; its output is published200Hz.
        tau=self.p['dq_filter_tau_ms']
        if tau==0:self.filtered=value.copy()
        else:self.filtered+=-np.expm1(-1/tau)*(value-self.filtered)
        if tick%5==0:
            self.packet=self.filtered*self.p['dq_gain']
            if self.p['dq_noise_std_rad_s']:
                noise=self.rng.normal(0,self.p['dq_noise_std_rad_s'],4)
                self.packet[LEGS]+=noise
        s[12:18]=self.packet
        return s


def dynamic_cases():
    """Frozen targeted grid: no selection using the later temporal holdout."""
    rows=[];seen=set()
    def add(name,family,params=None,**dyn):
        p=nominal();p.update(params or {})
        d=dynamic_nominal();d.update(dyn)
        key=repr(sorted(p.items()))+repr(sorted(d.items()))
        if key in seen:return
        seen.add(key)
        rows.append(dict(name=name,family=family,parameters=validate(p),dynamics=validate_dynamics(d)))
    add('dyn_nominal','baseline')
    add('dyn_legP20','gain',dict(kp=20.))
    add('dyn_wheelKv015','gain',dict(kv=.15))
    add('dyn_legP20_wheelKv015','gain',dict(kp=20.,kv=.15))
    for age in (16,17,18,19,20):
        for kp in (40.,20.):add(f'pdage{age}_P{kp:g}','pd_age',dict(pd_age_ms=age,kp=kp))
    for age in (5,10,15,20):
        for kd in (.5,1.,3.,4.):add(f'pdage{age}_D{kd:g}','D_age',dict(pd_age_ms=age,kd=kd))
    for arm in (.003,.006,.024):
        for kd in (1.,2.,4.):add(f'J{arm:g}_D{kd:g}','inertia_D',dict(armature=arm,kd=kd))
    for delay in (0,5,15,20,25,30):
        for tau in (0.,2.,10.):
            p=nominal();p['motor_delay_ms']=[delay,delay,7,delay,delay,11]
            p['motor_tau_ms']=[tau,tau,0,tau,tau,0]
            add(f'delay{delay}_tau{tau:g}','motor_timing',p)
    for gain in (.75,1.25,1.5,2.):add(f'dq_gain{gain:g}','velocity_gain',dq_gain=gain)
    for age in (2,5,10,15,20):add(f'dq_age{age}','velocity_age',dq_extra_age_ms=age)
    for tau in (2.,5.,10.,20.):add(f'dq_filter{tau:g}','velocity_filter',dq_filter_tau_ms=tau)
    for std in (.02,.1,.5,1.):add(f'dq_noise{std:g}','velocity_noise',dq_noise_std_rad_s=std)
    for rate in (200.,500.,1000.,3000.):add(f'motor_slew{rate:g}','motor_slew',motor_slew_nm_per_s=rate)
    add('pd100Hz','scheduling',pd_period_ms=10)
    for age in (5,15,20):add(f'pd100Hz_age{age}','scheduling_age',dict(pd_age_ms=age),pd_period_ms=10)
    for damping in (.2,1.,3.):add(f'pd_age20_viscous{damping:g}','damping_age',dict(pd_age_ms=20,viscous=damping))
    for scale in (.9,1.1):add(f'pd_age18_spring{scale:g}','load_age',dict(pd_age_ms=18,spring_scale=scale))
    return rows


def nominal():
    return dict(kp=40., kd=2., kv=.2, actor_age_ms=10, pd_age_ms=10,
                motor_delay_ms=[10, 10, 7, 10, 10, 11],
                motor_tau_ms=[5., 5., 0., 5., 5., 0.],
                armature=.012, viscous=0., frictionloss=0., root_mass_scale=1.,
                root_inertia_scale=1., friction=.5, spring_scale=1.,
                motor_gain=1., root_vx=0., zero_bias=[0.] * 4)


def validate(p):
    if set(p) != set(nominal()):
        raise ValueError('explicit search fields required; no unknown or missing parameter')
    for key in ('actor_age_ms', 'pd_age_ms'):
        if p[key] != int(p[key]) or not 0 <= p[key] <= 20:
            raise ValueError('feedback ages must be integer0..20ms, within C freshness guards')
    for key in ('motor_delay_ms', 'motor_tau_ms'):
        a = np.asarray(p[key], float)
        if a.shape != (6,) or not np.isfinite(a).all() or (a < 0).any() or (a > 30).any():
            raise ValueError('six finite timing values0..30ms required')
        if key == 'motor_delay_ms' and not np.equal(a, np.round(a)).all():
            raise ValueError('transport is integer milliseconds')
    limits = dict(kp=(10, 60), kd=(0, 4), kv=(.05, .3), armature=(.003, .35),
                  viscous=(0, 5), frictionloss=(0, 2), root_mass_scale=(.7, 1.3),
                  root_inertia_scale=(.5, 2), friction=(.2, 1), spring_scale=(.8, 1.2),
                  motor_gain=(.5, 1), root_vx=(-.3, .3))
    for key, (lo, hi) in limits.items():
        if not np.isfinite(p[key]) or not lo <= p[key] <= hi:
            raise ValueError('bounded diagnostic parameter: ' + key)
    b = np.asarray(p['zero_bias'])
    if b.shape != (4,) or not np.isfinite(b).all() or (abs(b) > .1).any():
        raise ValueError('four calibration offsets within .1rad required')
    return copy.deepcopy(p)


def singles():
    rows = []
    def add(name, family, **kw):
        p = nominal(); p.update(kw)
        rows.append(dict(name=name, family=family, parameters=validate(p)))
    add('nominal', 'baseline')
    add('legP20', 'gain', kp=20.)
    add('wheelKv015', 'gain', kv=.15)
    add('legP20_wheelKv015', 'gain', kp=20., kv=.15)
    for age in (0, 5, 15, 20):
        add(f'actor_age{age}', 'actor_age', actor_age_ms=age)
        add(f'pd_age{age}', 'pd_age', pd_age_ms=age)
    for ms in (0, 5, 15, 20):
        a = nominal()['motor_delay_ms']
        for j in LEGS: a[j] = ms
        add(f'leg_motor_delay{ms}', 'motor_delay', motor_delay_ms=a)
    for ms in (0., 2., 10., 20.):
        a = [0.] * 6
        for j in LEGS: a[j] = ms
        add(f'leg_motor_tau{ms:g}', 'motor_tau', motor_tau_ms=a)
    for a in (.006, .02, .04, .11, .35): add(f'armature{a:g}', 'armature', armature=a)
    for a in (.2, 1., 3.): add(f'viscous{a:g}', 'viscous', viscous=a)
    for a in (.2, 1.): add(f'frictionloss{a:g}', 'frictionloss', frictionloss=a)
    for a in (.8, 1.2): add(f'root_mass{a:g}', 'root_mass', root_mass_scale=a)
    for a in (.5, 2.): add(f'root_inertia{a:g}', 'root_inertia', root_inertia_scale=a)
    for a in (.3, .7): add(f'friction{a:g}', 'friction', friction=a)
    for a in (.8, .9, 1.1, 1.2): add(f'spring{a:g}', 'spring', spring_scale=a)
    for a in (.6, .8): add(f'leg_motor_gain{a:g}', 'motor_gain', motor_gain=a)
    for a in (-.15, .15): add(f'initial_vx{a:g}', 'initial_velocity', root_vx=a)
    for a in (-.06, -.03, .03, .06):
        add(f'hip_bias{a:g}', 'hip_zero', zero_bias=[a, 0, -a, 0])
        add(f'active_bias{a:g}', 'active_zero', zero_bias=[0, a, 0, a])
    return rows


def combinations(leaders, limit=8):
    """Two different mechanisms only; no duplicate family or gain changes."""
    selected, seen = [], set()
    base = nominal()
    for a, b in itertools.combinations(leaders, 2):
        if a['family'] == b['family'] or {a['family'], b['family']} & {'gain', 'baseline'}:
            continue
        p = nominal();d=dynamic_nominal()
        for row in (a, b):
            for k, v in row['parameters'].items():
                if v != base[k]: p[k] = copy.deepcopy(v)
            for k,v in row.get('dynamics',{}).items():
                if v!=dynamic_nominal()[k]:d[k]=v
        key = repr(sorted(p.items()))+repr(sorted(d.items()))
        if key in seen: continue
        seen.add(key)
        selected.append(dict(name='pair_' + a['name'] + '__' + b['name'],
                             family='pair', parameters=validate(p), parents=[a['name'], b['name']]))
        if 'dynamics' in a or 'dynamics' in b:selected[-1]['dynamics']=validate_dynamics(d)
        if len(selected) >= limit: break
    return selected


def features_grid_v1(time_s, q4, dq6, mean, torque6, gravity, *, window=(.5, 6.)):
    """Use identical observable channels and fixed windows on log and simulation."""
    time_s = np.asarray(time_s)
    select = (time_s >= window[0]-1e-8) & (time_s < window[1]-1e-8)
    t = time_s[select]
    if len(t) < 16:
        return dict(samples=len(t), coverage=0., valid=False)
    if np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > .015:
        raise ValueError('frequency window requires contiguous chronological ~100Hz observations')
    grid = np.arange(t[0], t[-1]+1e-7, .01)
    def interp(a):
        a = np.asarray(a)[select]
        if not np.isfinite(a).all(): raise ValueError('nonfinite measured observable')
        return np.stack([np.interp(grid, t, a[:, j]) for j in range(a.shape[1])], -1)
    q, v, a, tau, g = map(interp, (q4, dq6, mean, torque6, gravity))
    peaks, fractions = [], []
    for j in LEGS:
        f, power = signal.welch(v[:, j], fs=100, nperseg=min(256, len(grid)))
        band = (f >= 5) & (f <= 45)
        peaks.append(float(f[np.flatnonzero(band)[np.argmax(power[band])]]))
        fractions.append(float(power[(f >= 20) & (f <= 27)].sum() / max(power[band].sum(), 1e-12)))
    tilt = np.column_stack((np.arctan2(g[:, 2], -g[:, 1]), np.arctan2(g[:, 0], -g[:, 1])))
    limits = np.array([40, 40, WHEEL_CAP, 40, 40, WHEEL_CAP])
    return dict(valid=True, samples=len(grid), coverage=min(1.,len(grid)/round((window[1]-window[0])*100)),
                velocity_peak_hz=peaks, band20_27_fraction=fractions,
                dq_std_rad_s=v[:, LEGS].std(0).tolist(), q_std_rad=q.std(0).tolist(),
                action_std=a.std(0).tolist(), torque_rms_nm=np.sqrt((tau*tau).mean(0)).tolist(),
                command_saturation=(abs(tau) >= limits-1e-4).mean(0).tolist(),
                tilt_mean_rad=tilt.mean(0).tolist(), tilt_std_rad=tilt.std(0).tolist())


def features(time_s, q4, dq6, mean, torque6, gravity, *, window=(.5, 6.)):
    """Spectra need a uniform grid; command saturation/moments use RAW samples.

    Interpolating alternating saturated torque commands first would erase their
    plateaus. Keep the original v1 function for the first search's provenance.
    """
    result=features_grid_v1(time_s,q4,dq6,mean,torque6,gravity,window=window)
    if not result['valid']:return result
    t=np.asarray(time_s)
    keep=(t>=window[0]-1e-8)&(t<window[1]-1e-8)
    q,v,a,tau,g=[np.asarray(z)[keep] for z in (q4,dq6,mean,torque6,gravity)]
    tilt=np.column_stack((np.arctan2(g[:,2],-g[:,1]),np.arctan2(g[:,0],-g[:,1])))
    limits=np.array([40,40,WHEEL_CAP,40,40,WHEEL_CAP])
    result.update(moment_samples=int(keep.sum()),score_version='raw_moments_fft_grid_v2',
        dq_std_rad_s=v[:,LEGS].std(0).tolist(),q_std_rad=q.std(0).tolist(),
        action_std=a.std(0).tolist(),torque_rms_nm=np.sqrt((tau*tau).mean(0)).tolist(),
        command_saturation=(abs(tau)>=limits-1e-4).mean(0).tolist(),
        tilt_mean_rad=tilt.mean(0).tolist(),tilt_std_rad=tilt.std(0).tolist())
    return result


def similarity(real, sim):
    if not real.get('valid'): raise ValueError('real target features missing')
    if not sim.get('valid'): return dict(score=100., coverage=0., oscillation_candidate=False)
    terms = {}
    for key, floor in [('velocity_peak_hz', 5.), ('dq_std_rad_s', 1.), ('q_std_rad', .02),
                       ('action_std', .05), ('torque_rms_nm', 2.), ('tilt_mean_rad', .03), ('tilt_std_rad', .03)]:
        r, s = np.asarray(real[key]), np.asarray(sim[key])
        terms[key] = float(np.mean(abs(s-r)/np.maximum(abs(r), floor)))
    for key in ('band20_27_fraction', 'command_saturation'):
        terms[key] = float(np.mean(abs(np.array(real[key])-np.array(sim[key]))))
    score = float(np.mean(list(terms.values())) + 3*(1-sim['coverage']))
    # Early falls and tiny numerical oscillations cannot count as reproductions.
    oscillation = (sim['coverage'] >= .98 and
                   np.mean(abs(np.array(sim['velocity_peak_hz'])-np.array(real['velocity_peak_hz']))) <= 4 and
                   np.median(sim['dq_std_rad_s']) >= .5*np.median(real['dq_std_rad_s']) and
                   np.mean(sim['band20_27_fraction']) >= .25)
    return dict(score=score, coverage=sim['coverage'], terms=terms, oscillation_candidate=bool(oscillation),
                physical_parameters_identified=False)


class CExecution:
    """C owner/adoption/PD/mapping/compensation plus declared plant hypotheses.

    Gain variants use the production C request plus only the explicit gain delta,
    then call C mapping/compensation again. Nominal output must match full C core.
    This never changes the electrical repository or its constants.
    """
    def __init__(self, library, p, initial_motor, *, dynamics=None, core_type=FirmwareControl):
        self.p = validate(p); self.core = core_type(library)
        self.dynamics=validate_dynamics(dynamics)
        self.lib = self.core.lib
        self.lib.RLMotion31MapPolicyLegTorqueToMotorOrder.argtypes = [FP, FP]
        self.lib.RLControlCompensateSpring.argtypes = [FP, FP, FP, FP]
        self.driver = np.asarray(initial_motor, np.float32).copy()
        self.fifo = TargetTransport(p['motor_delay_ms'], self.driver)
        self.motor = self.driver.copy(); self.adopted = np.zeros(6, np.float32)
        self.last = None; self.last_tick = -1; self.nominal_error = 0.

    def close(self): self.core.close()

    def compensation(self, q4):
        length, spread = self.core.geometry(q4)
        z, out = np.zeros(4,np.float32), np.empty(4,np.float32)
        if self.lib.RLControlCompensateSpring(z.ctypes.data_as(FP),length.ctypes.data_as(FP),
                    spread.ctypes.data_as(FP),out.ctypes.data_as(FP)):
            raise ValueError('C compensation rejected geometry')
        result = np.zeros(6,np.float32)
        result[[4,3,1,0]] = out*[-1,-1,1,-1]
        return result

    def update_pd(self, tick, native, command, *, packet_result=None):
        """One actual C adoption/PD operation; optional individually aged packet.

        Both schedulers reuse this physical conversion. The legacy caller still
        executes exactly on its original modulo clock and single-age ABI.
        """
        p = self.p
        native = np.asarray(native, np.float32)
        if packet_result is None:
            length, spread = self.core.geometry(native[6:10])
            self.last = self.core.pd(tick,tick-p['pd_age_ms'],native,command,length,spread)
        else:
            self.last = packet_result
            length, spread = self.core.geometry(native[6:10])
        self.adopted = self.last['adopted_target'].copy()
        raw = self.last['pd_request_nm'].copy()
        raw[LEGS] += (p['kp']-40)*(self.adopted[LEGS]-native[6:10]) - (p['kd']-2)*native[10:16][LEGS]
        raw[WHEELS] += (p['kv']-.2)*(self.adopted[WHEELS]-native[10:16][WHEELS])
        limits = np.array([40,40,5.5,40,40,5.5],np.float32)
        limited = np.ascontiguousarray(np.clip(raw,-limits,limits),np.float32)
        mapped, motor = np.empty(4,np.float32), np.empty(4,np.float32)
        self.lib.RLMotion31MapPolicyLegTorqueToMotorOrder(limited.ctypes.data_as(FP),mapped.ctypes.data_as(FP))
        if self.lib.RLControlCompensateSpring(mapped.ctypes.data_as(FP),length.ctypes.data_as(FP),
                spread.ctypes.data_as(FP),motor.ctypes.data_as(FP)):
            raise ValueError('C motor compensation failed')
        driver = np.zeros(6,np.float32)
        driver[[4,3,1,0]] = motor*[-1,-1,1,-1]
        driver[WHEELS] = np.clip(limited[WHEELS],-WHEEL_CAP,WHEEL_CAP)
        if (p['kp'],p['kd'],p['kv']) == (40.,2.,.2):
            self.nominal_error=max(self.nominal_error,float(abs(driver-self.last['motor_model_nm']).max()))
            if self.nominal_error > 2e-5: raise ValueError('nominal full-C execution mismatch')
        self.driver = driver
        self.raw = raw

    def step(self, tick, feedback, current_q4, command):
        if tick != self.last_tick+1: raise ValueError('contiguous1ms execution required')
        self.last_tick = tick
        p = self.p
        if tick % self.dynamics['pd_period_ms'] == 0:
            self.update_pd(tick,feedback,command)
            driver = self.driver
            realized_command = driver.copy(); realized_command[LEGS] *= p['motor_gain']
            self.fifo.submit(tick,realized_command)
        next_motor=torque_response(self.motor,self.fifo.delivered(tick),p['motor_tau_ms'])
        slew=self.dynamics['motor_slew_nm_per_s']
        if slew:
            delta=next_motor-self.motor
            delta[LEGS]=np.clip(delta[LEGS],-slew*.001,slew*.001)
            next_motor=self.motor+delta
        self.motor=next_motor
        self.spring = -p['spring_scale']*self.compensation(current_q4)
        return self.motor+self.spring
