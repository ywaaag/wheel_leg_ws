import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def test_nearby_circle_fit_recovers_hole_not_mirror_hint():
 pytest.importorskip('pxr')
 from locate_right_spring_holes import fit_circle_near
 t=np.linspace(0,2*np.pi,40,endpoint=False);center=np.array([-.04,-.02])
 yz=center+.003*np.c_[np.cos(t),np.sin(t)]
 pts=np.r_[np.c_[np.full(40,-.0025),yz],np.c_[np.full(40,.0075),yz]]
 r=fit_circle_near(pts,np.array([0.,-.0397,-.0202]))
 np.testing.assert_allclose(r['center_yz_m'],center,atol=1e-8)
 assert r['unique_points']==40 and r['coverage_deg']>340
 assert r['radius_m']==pytest.approx(.003,abs=1e-8)
 with pytest.raises(ValueError):fit_circle_near(pts,np.array([0.,1.,1.]))
