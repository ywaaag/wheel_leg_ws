"""Thread-safe manual command UI; no action feedback controller or task scoring."""
import threading
import math

class MotionKeyboard:
 def __init__(self,max_speed=2.,max_yaw=.5,step=.25):
  if not all(math.isfinite(v) for v in (max_speed,max_yaw,step)) or not max_speed>0 or not max_yaw>=0 or not 0<step<=1:raise ValueError('manual limits must be finite: vx>0, yaw>=0, step in (0,1]')
  self.max_speed=float(max_speed);self.max_yaw=float(max_yaw);self.step=float(step)
  self.vx=0.;self.yaw=0.;self.reset=False;self.lock=threading.Lock()
 def key(self,key):
  with self.lock:
   if key in (ord('W'),ord('w'),265):self.vx=min(self.max_speed,self.vx+self.step)
   elif key in (ord('S'),ord('s'),264):self.vx=max(-self.max_speed,self.vx-self.step)
   elif key in (ord('A'),ord('a'),263):self.yaw=min(self.max_yaw,self.yaw+self.step)
   elif key in (ord('D'),ord('d'),262):self.yaw=max(-self.max_yaw,self.yaw-self.step)
   elif key==32:self.vx=self.yaw=0.
   elif key in (ord('R'),ord('r')):self.reset=True;self.vx=self.yaw=0.
   else:return
   print(f'command vx={self.vx:+.2f} m/s yaw={self.yaw:+.2f} rad/s'+(' RESET' if self.reset else ''),flush=True)
 def read(self):
  with self.lock:
   reset=self.reset;self.reset=False
   return self.vx,self.yaw,reset


class MotionPanelControls:
 """Browser and optional GLFW keys write the same Plan, never two targets."""
 def __init__(self,plan,keyboard_enabled=False):
  self.plan=plan;self.keyboard_enabled=keyboard_enabled;self.key_events=0
 def key(self,key):
  if not self.keyboard_enabled:return
  row=self.plan.snapshot();vx,yaw=row['target']['vx'],row['target']['yaw']
  if key in (ord('W'),ord('w'),265):payload={'vx':vx+.25}
  elif key in (ord('S'),ord('s'),264):payload={'vx':vx-.25}
  elif key in (ord('A'),ord('a'),263):payload={'yaw':yaw+.25}
  elif key in (ord('D'),ord('d'),262):payload={'yaw':yaw-.25}
  elif key==32:payload={'center':True}
  elif key in (ord('R'),ord('r')):payload={'reset':True,'center':True}
  else:return
  self.key_events+=1;self.plan.apply(payload)
 def read(self):
  row=self.plan.take()
  if row['reset']:
   self.plan.apply({'center':True});return 0.,0.,True
  return row['vx'],row['yaw'],False
 def read_full(self):
  """vx/yaw/reset plus the editable height and 31D skill/jump one-hot."""
  vx,yaw,reset=self.read()
  row=self.plan.snapshot()
  return {'vx':vx,'yaw':yaw,'reset':reset,'height':row['height'],'skill':row['skill'],'jump':row['jump'],
          'model':row['model']}
 def publish(self,telemetry):
  self.plan.publish({**telemetry,'key_events':self.key_events,'panel_requests':self.plan.requests})
