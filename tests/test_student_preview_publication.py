"""Preview consumes authorized publication receipt, before any Isaac launch."""
FROZEN_TEST_SCOPE='PORTABLE'
import json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from native_stand.command_v6.tools.experiment_preparation import publish,digest
from native_stand.students import play

def fixture(tmp_path,monkeypatch):
 checkpoint=tmp_path/'student.pt';checkpoint.write_bytes(b'identity fixture')
 spec={'student_evaluation':{'cases':[{'kind':'student','family':'Swing','checkpoint':str(checkpoint),'sha256':digest(checkpoint),'role':'RecoveryStudent'}]},'recovery':{'active_family':'Swing'},'interface_version':'multiskill_mcu_v2','asset_module_root':str(tmp_path),'device':{'physical_index':0}}
 folder=tmp_path/'published';publish(folder,{'spec.json':spec,'budget.json':{'authorization':'UNAUTHORIZED'}},{},completion_marker=True)
 # Mirror actual registration and subsequent mutable reservation ledger.
 spec['authorization']={'status':'AUTHORIZED'};(folder/'spec.json').write_text(json.dumps(spec))
 (folder/'budget.json').write_text(json.dumps({'authorization':'AUTHORIZED','approval_id':'approved','attempts':{'completed':{'status':'COMPLETED'}}}))
 (folder/'authorization_receipt.json').write_text(json.dumps({'spec_sha256':digest(folder/'spec.json'),'approval_id':'approved'}))
 args=SimpleNamespace(spec=folder/'spec.json',checkpoint=checkpoint,preview=True,handoff_condition=None,keyboard=False,run=False,physical_gpu=0)
 monkeypatch.setattr(play,'validate',lambda s:[])
 return args,spec,folder

def test_plan_authorized_receipt_and_no_launcher(tmp_path,monkeypatch,capsys):
 args,spec,folder=fixture(tmp_path,monkeypatch);play.execute(args,spec)
 assert json.loads(capsys.readouterr().out)['publication']['status']=='VERIFIED'
 # Truly changed spec must still fail, rather than bypassing strict SHA.
 spec['unexpected']='tampered';args.spec.write_text(json.dumps(spec))
 with pytest.raises(ValueError,match='input SHA differs'):play.execute(args,spec)

def test_gui_uses_same_receipt_before_launcher(tmp_path,monkeypatch):
 args,spec,folder=fixture(tmp_path,monkeypatch);args.run=True
 class ReachedLauncher(Exception):pass
 def launcher(*a,**kw):raise ReachedLauncher('verified before Isaac')
 monkeypatch.setitem(sys.modules,'isaaclab.app',SimpleNamespace(AppLauncher=launcher))
 monkeypatch.setattr(play.sys,'path',list(sys.path));monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 with pytest.raises(ReachedLauncher):play.execute(args,spec)
 # Wrong receipt also blocks GUI before launcher.
 receipt=json.loads((folder/'authorization_receipt.json').read_text());receipt['spec_sha256']='f'*64
 (folder/'authorization_receipt.json').write_text(json.dumps(receipt))
 with pytest.raises(ValueError,match='input SHA differs'):play.execute(args,spec)

def test_preview_gpu_override_preserves_spec_and_maps_renderer(tmp_path,monkeypatch):
 args,spec,folder=fixture(tmp_path,monkeypatch);args.run=True;args.physical_gpu=1
 original=args.spec.read_bytes()
 class Reached(Exception):pass
 def launcher(*a,**kw):
  import os
  assert os.environ['CUDA_VISIBLE_DEVICES']=='1' and a[0]['device']=='cuda:0'
  assert a[0]['multi_gpu'] is False and '/renderer/activeGpu=1' in a[0]['kit_args']
  raise Reached()
 monkeypatch.setattr(play.sys,'path',list(sys.path))
 monkeypatch.setitem(sys.modules,'isaaclab.app',SimpleNamespace(AppLauncher=launcher))
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES'):play.execute(args,spec)
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
 with pytest.raises(Reached):play.execute(args,spec)
 assert args.spec.read_bytes()==original and spec['device']['physical_index']==0

def test_student_gui_consumes_public_floor_options_and_follow_camera(tmp_path,monkeypatch):
 import torch
 from native_stand.students import evaluator,model
 from native_stand.command_v6.tools import ground_visual
 args,spec,folder=fixture(tmp_path,monkeypatch);args.run=True
 args.max_policy_steps=200;args.max_wall_seconds=30;args.ground_style='solid';args.ground_extent_m=64.;args.ground_grid_spacing_m=.5
 calls=[]
 app=SimpleNamespace(is_running=lambda:True,close=lambda:calls.append('app_close'))
 monkeypatch.setitem(sys.modules,'isaaclab.app',SimpleNamespace(AppLauncher=lambda *a,**k:SimpleNamespace(app=app)))
 stage=object();monkeypatch.setitem(sys.modules,'omni',SimpleNamespace(usd=SimpleNamespace(get_context=lambda:SimpleNamespace(get_stage=lambda:stage))))
 monkeypatch.setitem(sys.modules,'omni.usd',sys.modules['omni'].usd)
 env=SimpleNamespace(device='cpu',student_yaw_amplitude=0.,select_initial_family=lambda *a:None,reset=lambda:({'student_actor':torch.zeros(1,31)},{}),step=lambda a:({'student_actor':torch.zeros(1,31)},),sim=object(),robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=torch.tensor([[2.,3.,.2]])))),close=lambda:calls.append('env_close'))
 from native_stand.recovery.domain import Mode
 env.num_envs=1;env.fsm=SimpleNamespace(mode=torch.tensor([int(Mode.RECOVERY)]));env.initialization_checked=torch.tensor([True])
 spec['student_evaluation']['cases'][0].update(yaw_rad=0.,family='Swing',seed=20261015)
 monkeypatch.setattr(evaluator,'environment',lambda *a,**k:env)
 class Actor(torch.nn.Module):
  def forward(self,obs,stochastic_output=False):return torch.zeros(1,6)
 monkeypatch.setattr(model,'load',lambda *a:(Actor(),{}))
 def setup(s,sim,**kw):assert s is stage and sim is env.sim;calls.append(('setup',kw));return {}
 monkeypatch.setattr(ground_visual,'setup',setup)
 monkeypatch.setattr(ground_visual,'camera_view',lambda sim,xy=(0.,0.):calls.append(('camera',tuple(xy))))
 monkeypatch.setattr(play.sys,'path',list(sys.path));monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 play.execute(args,spec)
 assert ('setup',{'style':'solid','extent_m':64.,'grid_spacing_m':.5}) in calls
 assert ('camera',(0.,0.)) in calls and ('camera',(2.,3.)) in calls
 assert calls[-2:]==['env_close','app_close']
