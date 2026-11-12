import copy
import pytest
import torch
from native_stand.students import motion_parallel_design as d

def test_slots_and_independent_levels():
 s=d.initial_state();rows=d.allocation(s)
 assert sum(r['slots'] for r in rows)==4096
 assert {r['group']:sum(x['slots'] for x in rows if x['group']==r['group']) for r in rows}==d.configuration()['slots']
 assert {tuple(r['target']) for r in rows if r['group']=='backward'}=={(-.5,0.),(-1.,0.)}
 assert not any(abs(r['target'][0])==4 or abs(r['target'][1])==4 for r in rows)
 s['forward']['level']=2;rows=d.allocation(s)
 assert any(r['target']==[3.5,0.] and r['group']=='forward' for r in rows)
 assert any(r['target']==[2.5,0.] and r['group']=='retention' for r in rows)
 assert sum(r['slots'] for r in rows)==4096

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('target',[-4.,-1.,1.,4.])
def test_directional_gradient_no_flat_tail(axis,target):
 c=torch.zeros(1,2,dtype=torch.float64);c[0,axis]=target
 x=torch.tensor([0.],dtype=torch.float64,requires_grad=True)
 v=torch.stack((x if axis==0 else x*0,x*0),-1);yaw=x if axis==1 else x*0
 r=d.signals(c,v,yaw,new_task=torch.ones(1,dtype=torch.bool),mix=x*0)
 grad=torch.autograd.grad(r['density'].sum(),x)[0]
 assert torch.isfinite(grad).all() and grad.item()*target>0 and abs(grad.item())>.1

@pytest.mark.parametrize('axis',[0,1])
def test_target_beats_still_wrong_and_fast_overshoot(axis):
 c=torch.zeros(4,2,dtype=torch.float64);c[:,axis]=4
 x=torch.tensor([4.,0.,-4.,8.],dtype=torch.float64)
 v=torch.stack((x if axis==0 else x*0,x*0),-1);y=x if axis==1 else x*0
 r=d.signals(c,v,y,new_task=torch.ones(4,dtype=torch.bool),mix=x*0)['density']
 assert r[0]>r[1]>r[2] and r[0]>r[3]

def test_spin_forward_shortcut_and_zero_penalized():
 c=torch.tensor([[0.,1.],[0.,1.],[0.,0.],[0.,0.]],dtype=torch.float64)
 v=torch.tensor([[0.,0.],[.6,0.],[0.,0.],[.6,0.]],dtype=torch.float64)
 r=d.signals(c,v,torch.tensor([1.,1.,0.,0.],dtype=torch.float64),new_task=torch.ones(4,dtype=torch.bool),mix=torch.zeros(4,dtype=torch.float64))
 assert r['density'][0]>r['density'][1] and r['density'][2]>r['density'][3]

def evidence(s,lane,update):
 targets={(0.,0.),*d.BASE,*d.endpoints(lane,s[lane]['level'])}
 return {'update':update,'source_sha256':'a'*64,'checkpoint_sha256':'b'*64,'protocol_sha256':'c'*64,
 'records':[{'target':list(t),'checkpoint_sha256':'b'*64,'protocol_sha256':'c'*64,'envs':[{'env':i,'accepted':True,'controlled_s':30.} for i in range(8)]} for t in targets]}

def test_evidence_two_updates_and_independent_advance():
 s=d.initial_state();a=evidence(s,'forward',100)
 s,r=d.advance(s,'forward',a,source_sha='a'*64,protocol_sha='c'*64);assert not r['passed']
 with pytest.raises(ValueError):d.advance(s,'forward',a,source_sha='a'*64,protocol_sha='c'*64)
 a['update']=200
 with pytest.raises(ValueError):d.advance(s,'forward',a,source_sha='a'*64,protocol_sha='c'*64)
 a['checkpoint_sha256']='d'*64
 for rec in a['records']:rec['checkpoint_sha256']='d'*64
 s,r=d.advance(s,'forward',a,source_sha='a'*64,protocol_sha='c'*64)
 assert r['passed'] and s['forward']['level']==1 and s['backward']['level']==s['yaw']['level']==0

def test_missing_failed_and_conflict_evidence():
 s=d.initial_state();a=evidence(s,'yaw',100);a['records'][0]['envs'].pop()
 _,r=d.advance(s,'yaw',a,source_sha='a'*64,protocol_sha='c'*64);assert not r['passed'] and r['missing_evidence']
 a=evidence(s,'yaw',100);a['records'][0]['envs'][0]['accepted']=False
 _,r=d.advance(s,'yaw',a,source_sha='a'*64,protocol_sha='c'*64);assert r['known_failure']
 a=evidence(s,'yaw',100);a['records'][0]['checkpoint_sha256']='d'*64
 with pytest.raises(ValueError):d.advance(s,'yaw',a,source_sha='a'*64,protocol_sha='c'*64)

def test_design_version_is_not_a_runtime_activation_and_scope():
 from native_stand.students.motion_course import implementation
 with pytest.raises(ValueError):implementation(d.VERSION)
 assert d.configuration()['authorization'].startswith('CPU_ONLY')
 with pytest.raises(ValueError):d.signals(torch.ones(1,2),torch.zeros(1,2),torch.zeros(1),new_task=torch.ones(1,dtype=torch.bool),mix=torch.zeros(1))

def test_old_tracking_components_match_actual_v9_base():
 from native_stand.students.motion_tracking_reward import reward,configuration,COMBO_VERSION
 c=torch.tensor([[0.,0.,.34],[1.,0.,.34],[0.,.5,.34]],dtype=torch.float64);v=torch.tensor([[.1,.02,0.],[.9,.02,0.],[.02,.01,0.]],dtype=torch.float64);yaw=torch.tensor([.02,.01,.45],dtype=torch.float64);mix=torch.tensor([0.,.5,0.],dtype=torch.float64)
 z=torch.zeros(3,dtype=torch.float64);yes=torch.ones(3,dtype=torch.bool)
 actual=reward(command=c,target=c[:,:2],velocity=v,yaw=yaw,height=z+.34,angles=z[:,None].expand(3,2),contacts=z[:,None].expand(3,2)+20,leg_q=z[:,None].expand(3,4),wheel_offset=z,leg_lengths=z[:,None].expand(3,2)+.3,applied_torque=z[:,None].expand(3,6),previous_targets=z[:,None].expand(3,6),executed_targets=z[:,None].expand(3,6),controlled=yes,valid=yes,physical_failure=~yes,fault=~yes,config=configuration(COMBO_VERSION),linear_narrow_mix=mix)
 proposed=d.signals(c[:,:2],v[:,:2],yaw,new_task=~yes,mix=mix)
 for name,value in proposed['fields'].items():assert torch.allclose(value,actual['density'][:,actual['fields'].index(name)],atol=1e-12)
