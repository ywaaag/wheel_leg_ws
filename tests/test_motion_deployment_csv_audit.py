"""Real CSV reader/ONNX ReferenceEvaluator/package checks, without simulation."""
FROZEN_TEST_SCOPE='PORTABLE'
import csv,importlib.util,json
from pathlib import Path
import numpy as np
import pytest
import onnx
from onnx import helper,TensorProto,numpy_helper


def module():
 import sys
 from unittest.mock import patch
 with patch.object(sys,'path',[str(Path('scripts').resolve()),*sys.path]):
  spec=importlib.util.spec_from_file_location('deployment_csv_audit',Path('scripts/audit_motion_deployment_csv.py'))
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m


def fixture(tmp_path):
 m=module();package=tmp_path/'package';package.mkdir();out=tmp_path/'out';out.mkdir()
 weights=np.zeros((31,6),np.float32);weights[:6]=np.eye(6,dtype=np.float32)
 graph=helper.make_graph([helper.make_node('MatMul',['observation','weights'],['mean_action'])],'test',[helper.make_tensor_value_info('observation',TensorProto.FLOAT,[1,31])],[helper.make_tensor_value_info('mean_action',TensorProto.FLOAT,[1,6])],[numpy_helper.from_array(weights,'weights')])
 model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);onnx.save(model,package/'actor.onnx')
 (package/'manifest.json').write_text(json.dumps({'actor_onnx_sha256':m.sha(package/'actor.onnx')}))
 (package/'files_sha256.json').write_text(json.dumps({name:m.sha(package/name) for name in ('actor.onnx','manifest.json')}))
 source=tmp_path/'log.csv';x=np.zeros((4,25));x[:,0]=[.01,.02,.03,.04];x[:,4]=-1;x[:,6]=[.1,.11,.12,.13];x[:,7:10]=.1;x[:,24]=.68
 y=x[:,:6].copy();y[-1]=y[-2]
 for i in range(1,4):x[i,16:22]=y[i-1,[0,1,3,4,2,5]]
 indices=[*range(22),23,22,24] # named lookup must survive swapped command columns
 with source.open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['Index','Time','(g_rl_shadow_debug).rl_active']+[f'(g_rl_shadow_debug).observation[{i}]' for i in indices]+[f'(g_rl_shadow_debug).mean_action[{i}]' for i in range(6)])
  for i in range(4):writer.writerow([i,i*.01,int(i<3),*x[i,indices],*y[i]])
 return m,source,package,out


def test_actual_onnx_replay_separates_disabled_stale_outputs_and_named_columns(tmp_path):
 m,source,package,out=fixture(tmp_path)
 report,data,infer,manifest=m.replay(source,package,out)
 assert report['windows'][0]['fraction_all_channels_within_1e_5']==1
 assert report['windows'][1]['fraction_all_channels_within_1e_5']==0
 assert report['previous_vs_last_prediction_active']['match_fraction']==1
 assert report['missing_task_flags_assumed']==[1,0,0,1,0,0]
 assert (out/'replayed.csv').is_file() and (out/'replay.npz').is_file()


def test_package_mutation_is_rejected_before_inference(tmp_path):
 m,source,package,out=fixture(tmp_path)
 with (package/'actor.onnx').open('ab') as f:f.write(b'changed')
 with pytest.raises(ValueError,match='SHA differs'):m.replay(source,package,out)


def test_fixed_windows_keep_missing_denominators_and_use_executed_current_limit():
 m=module();t=np.arange(1,3001)*.01
 a={'time':t,'state':np.zeros((len(t),6)),'contact':np.ones((len(t),2))*10,
    'torque':np.zeros((len(t),6)),'adopted':np.zeros((len(t),6))}
 a['torque'][:,2]=2.4469478638
 limits=[40,40,2.4469478638,40,40,2.4469478638]
 windows=m.diagnostic_windows(a,limits)
 assert [w['samples'] for w in windows]==[200,800,1000,1000]
 assert all(w['status']=='COMPLETE' and not w['certificate_eligible'] for w in windows)
 assert windows[2]['motor_command_saturation_per_channel'][2]==1
 assert windows[2]['nominal_5p5_wheel_torque_fraction'][0]==0
 # Missing one frame is never repaired by selecting a later interval.
 keep=np.arange(len(t))!=1100
 partial=m.diagnostic_windows({k:v[keep] for k,v in a.items()},limits)
 assert partial[2]['status']=='PARTIAL' and partial[2]['expected_samples']==1000 and partial[2]['samples']==999
 short=m.diagnostic_windows({k:v[:1030] for k,v in a.items()},limits)
 assert short[2]['status']=='PARTIAL' and short[3]['status']=='NOT_RUN'
 a['time'][300]=a['time'][299]
 with pytest.raises(ValueError,match='increasing'):m.diagnostic_windows(a,limits)


def test_firmware_timing_screen_has_explicit_fixed_conditions():
 import sys
 from unittest.mock import patch
 with patch.object(sys,'path',[str(Path('scripts').resolve()),*sys.path]):
  spec=importlib.util.spec_from_file_location('logged_transition',Path('scripts/diagnose_motion_logged_transition.py'))
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 cases=m.firmware_scenarios(5,30,Path('/explicit/library.so'),True)
 assert len(cases)==14 and len({c[0] for c in cases})==14
 assert all(c[2]=='zero' and c[4]==30 for c in cases)
 assert sum(c[4] for c in cases)==420
 assert {(c[5]['actor_age_ms'],c[5]['feedback_ms']) for c in cases if 'ideal_cancellation' in c[0]}=={(0,0),(10,0),(0,10),(10,10),(20,20)}
