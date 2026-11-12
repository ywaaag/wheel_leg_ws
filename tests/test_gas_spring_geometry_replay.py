import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def test_batch_fk_matches_scalar_fk_on_nonzero_states():
 pytest.importorskip('pxr');pytest.importorskip('mujoco')
 from replay_gas_spring_geometry import batch_fk
 from audit_recovery_same_state import usd_fk,transform
 tree=[('joint','robot','child',transform([.1,.2,.3],np.eye(3)),transform([0,.1,0],np.eye(3)),0)]
 q=np.array([-.5,.2,.7]);poses=batch_fk(tree,{'joint':q},3)
 for i,v in enumerate(q):np.testing.assert_allclose(poses['child'][i],usd_fk(tree,{'joint':v},np.eye(4))['child'],atol=1e-12)
