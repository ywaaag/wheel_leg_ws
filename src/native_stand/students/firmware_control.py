"""Explicit ABI for the electrical repository's host-compiled C control core.

Compilation and source identity belong to the CLI, not this runtime adapter.
All times are local monotonic milliseconds; negative synthetic sensor ticks
wrap as uint32 just as the firmware clock does.
"""
import ctypes as C
import numpy as np
from native_stand.students.real2sim_delay import TargetTransport,torque_response

FP=C.POINTER(C.c_float)
UP=C.POINTER(C.c_uint32)

def _array(values,n):
    a=np.ascontiguousarray(values,dtype=np.float32)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(f'finite physical array[{n}] required')
    return a

class FirmwareControl:
    def __init__(self,library):
        self.lib=C.CDLL(str(library));lib=self.lib
        lib.motion_fw_abi_version.restype=C.c_uint
        if lib.motion_fw_abi_version()!=1:raise ValueError('unsupported firmware host ABI')
        lib.motion_fw_create.restype=C.c_void_p
        lib.motion_fw_destroy.argtypes=[C.c_void_p]
        lib.motion_fw_observe.argtypes=[C.c_void_p,C.c_uint32,C.c_uint32,FP,FP,FP]
        lib.motion_fw_publish.argtypes=[C.c_void_p,C.c_uint32,FP]
        lib.motion_fw_pd.argtypes=[C.c_void_p,C.c_uint32,C.c_uint32,FP,FP,FP,FP,FP,UP]
        lib.motion_fw_geometry.argtypes=[FP,FP,FP]
        self.context=lib.motion_fw_create()
        if not self.context:raise MemoryError('firmware context allocation failed')

    def close(self):
        if self.context:self.lib.motion_fw_destroy(self.context);self.context=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def observe(self,tick,sensor_tick,native16,command):
        x,c=_array(native16,16),_array(command,3);out=np.empty(31,np.float32)
        rc=self.lib.motion_fw_observe(self.context,int(tick),int(sensor_tick),x.ctypes.data_as(FP),c.ctypes.data_as(FP),out.ctypes.data_as(FP))
        if rc:raise ValueError('firmware observation rejected: '+str(rc))
        return out

    def publish(self,tick,mean):
        a=_array(mean,6)
        if np.max(np.abs(a))>1.00001:raise ValueError('deterministic tanh mean outside [-1,1]')
        rc=self.lib.motion_fw_publish(self.context,int(tick),a.ctypes.data_as(FP))
        if rc:raise ValueError('firmware target publication rejected: '+str(rc))

    def geometry(self,q4):
        q=_array(q4,4);length=np.empty(2,np.float32);spread=np.empty(2,np.float32)
        rc=self.lib.motion_fw_geometry(q.ctypes.data_as(FP),length.ctypes.data_as(FP),spread.ctypes.data_as(FP))
        if rc:raise ValueError('firmware five-bar geometry rejected')
        return length,spread

    def pd(self,tick,sensor_tick,native16,command,length,spread):
        a=[_array(v,n) for v,n in [(native16,16),(command,3),(length,2),(spread,2)]]
        out=np.empty(30,np.float32);meta=np.empty(8,np.uint32)
        rc=self.lib.motion_fw_pd(self.context,int(tick),int(sensor_tick),*[v.ctypes.data_as(FP) for v in a],
                                 out.ctypes.data_as(FP),meta.ctypes.data_as(UP))
        if rc:raise RuntimeError(f'firmware control refused rc={rc}, owner/fault={meta[:2].tolist()}')
        return dict(motor_model_nm=out[:6].copy(),spring_compensation_model_nm=out[6:12].copy(),
                    pd_request_nm=out[12:18].copy(),pd_limited_nm=out[18:24].copy(),
                    adopted_target=out[24:30].copy(),metadata=meta.copy())


class FirmwareExecution:
    """Thin adapter for existing MuJoCo diagnostics, with declared spring hypothesis.

    motor_only: actual C motor command in a spring-free plant (known mismatch).
    ideal_cancellation: add the opposite C compensation as an assumed external
    spring torque. This deliberately ideal bound is NOT an identified spring.
    Optional motor timing acts on post-PD motor torque, not on the locally
    adopted target. Zero settings preserve the original C-only path exactly.
    """
    def __init__(self,library,initial_target,spring_mode,*,motor_delay_ms=None,motor_tau_ms=None):
        if spring_mode not in ('motor_only','ideal_cancellation'):raise ValueError('explicit spring hypothesis required')
        delay=np.asarray([0]*6 if motor_delay_ms is None else motor_delay_ms,dtype=float)
        tau=np.asarray([0]*6 if motor_tau_ms is None else motor_tau_ms,dtype=float)
        # These act AFTER local C PD adoption, on torque commands to the driver.
        # They are not another delay on position targets or another PD feedback.
        fifo=TargetTransport(delay,np.zeros(6))
        if tau.shape!=(6,) or not np.isfinite(tau).all() or (tau<0).any() or (tau>100).any():raise ValueError('six finite motor response times0..100ms required')
        self.control=FirmwareControl(library);self.spring_mode=spring_mode
        self.adopted=np.array(initial_target,copy=True);self.pre=np.zeros(6,np.float32)
        self.torque=np.zeros(6,np.float32);self.last=None;self.pd_feedback_native16=None
        self.motor_delay_ms=delay.astype(int);self.motor_tau_ms=tau.copy()
        self.motor_physics_enabled=bool(delay.any() or tau.any())
        self.motor_fifo=fifo;self.motor_actual=np.zeros(6);self.last_physics_tick=None

    @staticmethod
    def native(sensors):
        s=np.asarray(sensors)
        return np.r_[s[:6],s[6:12][[0,1,3,4]],s[12:18]].astype('float32')

    def observe(self,tick,sensor_tick,sensors,command):
        return self.control.observe(tick,sensor_tick,self.native(sensors),command)

    def publish(self,tick,mean):self.control.publish(tick,mean)

    def step(self,tick,sensor_tick,sensors,command):
        if self.motor_physics_enabled and (tick!=0 if self.last_physics_tick is None else tick!=self.last_physics_tick+1):
            raise ValueError('motor command transport/response requires contiguous1ms physics steps from0')
        if tick%5==0:
            native=self.native(sensors);length,spread=self.control.geometry(native[6:10])
            self.last=self.control.pd(tick,sensor_tick,native,command,length,spread)
            self.pd_feedback_native16=native.copy()
            self.adopted=self.last['adopted_target'].copy();self.pre=self.last['pd_request_nm'].copy()
            self.torque=self.last['motor_model_nm'].copy()
            if self.motor_physics_enabled:self.motor_fifo.submit(tick,self.last['motor_model_nm'])
            elif self.spring_mode=='ideal_cancellation':self.torque-=self.last['spring_compensation_model_nm']
        if self.motor_physics_enabled:
            demand=self.motor_fifo.delivered(tick)
            self.motor_actual=torque_response(self.motor_actual,demand,self.motor_tau_ms)
            self.torque=self.motor_actual.copy()
            # Explicit bound: assumed external spring is held at the latest C
            # compensation, independently of motor transport/response. Not fit
            # to cancel each delayed motor command, and not an actual spring.
            if self.spring_mode=='ideal_cancellation':self.torque-=self.last['spring_compensation_model_nm']
            self.last_physics_tick=tick
        return self.torque.copy()

    def close(self):self.control.close()
