"""Explicit Motion61 contact context; Actor31 and legacy Motion49 unchanged."""
import copy
import torch
VERSION='motion_spin_privileged61_v1'
NETWORK_VERSION='motion31_spin_large_actor_critic_v2'

def network():
 return {'version':NETWORK_VERSION,'actor_hidden_dims':[512,256,128],
  'critic_hidden_dims':[512,512,256],'actor_input':31,'critic_input':61,
  'hidden_activation':'ELU(alpha=1)','mean_output':'Tanh'}

def contract():
 from .motion_privileged import contract as base
 c=base();c.update(critic_dim=61,critic_privileged_version=VERSION)
 c['critic_privileged']=c['critic_privileged']+[
  'left_right_support_tangent_velocity_native_xy/.5:4',
  'left_right_rolling_forward_speed/4:2',
  'last_executed_target_delta_legs/.1_wheels/10:6']
 c['contact_context']='geometric cylinder support; not actual solver contact patch; policy100Hz'
 return c

def critic_input(base49,measurement,previous_targets,executed_targets,root_quaternion_w):
 n=len(base49)
 if base49.shape!=(n,49) or previous_targets.shape!=(n,6) or executed_targets.shape!=(n,6) or root_quaternion_w.shape!=(n,4):raise ValueError('Motion61 explicit input shape differs')
 # Quaternion WXYZ; horizontal forward Native -bodyZ, left Native -bodyX.
 q=root_quaternion_w;w,x,y,z=q.unbind(-1)
 left=-torch.stack((1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),-1)
 forward=-torch.stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)),-1)
 tangent=measurement['tangent_velocity_w']
 native=torch.stack(((tangent*forward[:,None,:2]).sum(-1),(tangent*left[:,None,:2]).sum(-1)),-1)
 extra=torch.cat((native.reshape(n,4)/.5,measurement['rolling_speed_forward_mps']/4.,
  (executed_targets-previous_targets)/base49.new_tensor([.1,.1,10.,.1,.1,10.])), -1)
 return torch.cat((base49,extra),-1)

def expand_source(source):
 """Zero new Critic columns and their Adam moments; preserve all existing state."""
 from .motion_large_combo import network as base_network
 if source.get('network_architecture')==network() and source.get('critic_privileged_version')==VERSION:return source
 if source.get('network_architecture')!=base_network() or source.get('critic_privileged_version')!='motion_privileged49_v1':raise ValueError('Motion61 migration requires explicit large Motion49 source')
 out=copy.deepcopy(source);weight=out['critic']['mlp.0.weight']
 if weight.shape!=(512,49):raise ValueError('Motion49 first critic layer differs')
 def pad(t):return torch.cat((t,torch.zeros(t.shape[0],12,dtype=t.dtype,device=t.device)),1)
 out['critic']['mlp.0.weight']=pad(weight)
 # Actor log_std + eight Actor linear tensors precede Critic parameters in ControlledPPO.
 ids=[p for g in out['optimizer']['param_groups'] for p in g['params']]
 actor_names=list(out['actor']);critic_names=list(out['critic'])
 if len(ids)!=len(actor_names)+len(critic_names):raise ValueError('Motion61 Adam parameter count differs')
 pid=ids[len(actor_names)+critic_names.index('mlp.0.weight')]
 state=out['optimizer']['state'].get(pid,{})
 for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
  if key in state:
   if state[key].shape!=weight.shape:raise ValueError('Motion61 source Adam shape differs')
   state[key]=pad(state[key])
 out.update(network_architecture=network(),critic_privileged_version=VERSION)
 out['critic_migration']={'version':VERSION,'base_dim':49,'new_dim':61,'initial_new_columns':'zeros','Adam_new_columns':'zeros','Actor':'unchanged','physical_environment':'reset'}
 return out
