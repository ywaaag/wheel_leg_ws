FROZEN_TEST_SCOPE='PORTABLE'
import json
from native_stand.students.motion_quick_evaluation import guarded_worker


def test_zero_return_without_summary_is_nonzero(tmp_path):
 r={'output':str(tmp_path),'checkpoint_sha256':'a'*64,'update':100}
 assert guarded_worker(r,lambda r:None)==1
 assert json.loads((tmp_path/'worker_exit_failure.json').read_text())['exit_code']==1


def test_systemexit_zero_cannot_hide_error(tmp_path):
 def exit0(r):raise SystemExit(0)
 assert guarded_worker({'output':str(tmp_path)},exit0)==1
 assert 'SystemExit' in json.loads((tmp_path/'worker_exit_failure.json').read_text())['exception']


def test_complete_matching_summary_success(tmp_path):
 r={'output':str(tmp_path),'checkpoint_sha256':'a'*64,'update':100}
 def success(r):(tmp_path/'summary.json').write_text(json.dumps({'status':'COMPLETED','checkpoint_sha256':'a'*64,'update':100}))
 assert guarded_worker(r,success)==0


def test_single_screen_does_not_enable_keyboard_command():
 import ast
 from pathlib import Path
 root=Path(__file__).resolve().parents[1]
 code=(root/'src/native_stand/students/evaluator.py').read_text()
 tree=ast.parse(code)
 function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='environment')
 condition=next(n for n in ast.walk(function) if isinstance(n,ast.If) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='enable_motion_preview' for stmt in n.body for x in ast.walk(stmt)))
 assert 'not single_condition_screen' in ast.unparse(condition.test)


def test_single_screen_readback_checks_actual_hold_command():
 from pathlib import Path
 code=(Path(__file__).resolve().parents[1]/'src/native_stand/students/motion_single_screen.py').read_text()
 assert "actual_command=arrays['effective_command_pre']" in code
 assert "raise ValueError('hold command readback mismatch')" in code
 assert 'single_condition_screen=True' in code


def test_automatic_screen_selects_only_positive_yaw_and_records_count():
 from pathlib import Path
 code=(Path(__file__).resolve().parents[1]/'src/native_stand/students/motion_quick_evaluation.py').read_text()
 assert "if r['command'][:2]==[0.,.5]" in code
 assert "s['student_evaluation']['cases']=targets" in code
 assert "'condition_count':len(s['student_evaluation']['cases'])" in code
