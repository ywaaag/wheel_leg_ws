"""Recover same-frame diagnostic PD gains without assuming motor array order."""
FROZEN_TEST_SCOPE='PORTABLE'
import importlib.util
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np


def module():
 with patch.object(sys,'path',[str(Path('scripts').resolve()),*sys.path]):
  spec=importlib.util.spec_from_file_location('audit_handover',Path('scripts/audit_motion_handover_torques.py'))
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m


def test_gain_search_recovers_permuted_signed_60_2_instead_of_40_2():
 rng=np.random.default_rng(23);q=rng.normal(0,.1,(80,4));target=q+rng.normal(0,.15,(80,4));dq=rng.normal(0,.5,(80,4));order=[3,2,1,0];sign=[-1,-1,1,-1]
 logged=np.clip((60*(target-q)-2*dq)[:,order]*sign,-40,40)
 result=module().gain_candidates(target,q,dq,logged,np.ones(80,bool),gains=[40.,60.],damping=[1.,2.],limit=40.)
 for i,r in enumerate(result):
  b=r['best'];assert (b['source_column'],b['gain'],b['damping'],b['sign'])==(order[i],60.,2.,sign[i]);assert b['max_abs_error_Nm']<1e-10


def test_wheel_gain_search_disambiguates_clipped1_from_point2():
 rng=np.random.default_rng(24);dq=rng.normal(0,2,(100,2));target=dq+rng.normal(0,4,(100,2));logged=np.clip(-(target-dq)[:,[1,0]],-5.5,5.5)
 result=module().gain_candidates(target,dq,np.zeros_like(dq),logged,np.ones(100,bool),gains=[.2,1.],damping=[0.],limit=5.5)
 assert all(r['best']['gain']==1. and r['best']['sign']==-1 and r['best']['RMSE_Nm']<1e-10 for r in result)
