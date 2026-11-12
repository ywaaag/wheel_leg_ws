import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def test_motion_adapter_preserves_physical_input_and_native_history():
    from compare_recovery_standing_mujoco import LegacyMotion
    from native_stand.students.mcu_interface import encode
    from native_stand.command_v6.domain.contract import frame,history
    seen=[]
    def actor(obs,stochastic_output):
        seen.append(obs.clone());assert stochastic_output is False
        return torch.zeros(1,6)
    adapter=LegacyMotion(actor,.4)
    ang=torch.tensor([[.2,.3,.4]]);grav=torch.tensor([[0.,-1.,0.]])
    q=torch.tensor([[.1,.2,0.,-.1,-.2,0.]]);dq=torch.arange(6.)[None]
    legs=torch.tensor([[.2,.3,-.2,-.3]]);wheels=torch.tensor([[4.,5.]])
    x=encode(ang,grav,q,dq,legs,wheels,torch.zeros(1,6),'Recovery',torch.tensor([[0.,0.,.34]]))
    adapter.mean(x)
    prev=torch.tensor([[.4,.6,.04,-.4,-.6,.05]])
    f=frame(ang,grav,torch.tensor([[0.,0.,.4]]),q,dq,prev,torch.zeros(1,6))
    torch.testing.assert_close(seen[0]['policy'],f)
    torch.testing.assert_close(seen[0]['history'],f.repeat(1,5))
    adapter.mean(x)
    torch.testing.assert_close(seen[1]['history'],history(seen[0]['history'],f,torch.tensor([False])))

def test_motion_decoder_keeps_original_scale_and_mechanical_clip():
    from compare_recovery_standing_mujoco import motion_targets
    raw=torch.tensor([[.8,.6,.04,-.8,-.6,.05]])
    limits=torch.tensor([[[-.35,.35]]*6])
    out=motion_targets(raw,torch.zeros(1,6),limits)
    torch.testing.assert_close(out['leg_requested'],torch.tensor([[.4,.3,-.4,-.3]]))
    torch.testing.assert_close(out['leg_executed'],torch.tensor([[.35,.3,-.35,-.3]]))
    torch.testing.assert_close(out['wheel_executed'],torch.tensor([[4.,5.]]))
