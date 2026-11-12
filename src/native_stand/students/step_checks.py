"""Explicit tensor flags: one host decision, original checks retained on failure."""
import torch

class StepChecks:
 def __init__(self):self.flags=[];self.names=[]
 def require(self,condition,name):
  self.flags.append(condition.all());self.names.append(name)
 def verify(self,error=ValueError):
  if not self.flags:return
  flags=torch.stack(self.flags)
  if not bool(flags.all()):
   failed=[name for name,ok in zip(self.names,flags.detach().cpu().tolist()) if not ok]
   raise error('step safety: '+', '.join(failed))

def require(condition,name,checks=None):
 if checks is not None:checks.require(condition,name)
 elif not bool(condition.all()):raise ValueError(name)
