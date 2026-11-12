"""Explicit local shutdown receipt; never launch physics or mutate ledgers."""
import copy,json
from pathlib import Path
import pytest
from native_stand.students.motion_reboot_recovery import verify
ROOT=Path(__file__).resolve().parents[1]
def spec():
 p=ROOT/'configs/experiments/2026_10_18-15_45__motion750_symmetry85_reboot_resume/spec.json'
 if not p.exists():pytest.skip('local shutdown recovery unavailable')
 return json.loads(p.read_text())
def test_recovery_inclusive_budget_and_wiring():
 r=verify(spec());assert r['remaining_formal_updates']+r['historical_smoke_updates']+r['historical_formal_updates']+r['conservative_uncommitted_update_reserve']==3000
 assert r['old_ledger_rewritten'] is False
@pytest.mark.parametrize('kind',['lr','quota','source'])
def test_recovery_rejects_undeclared_changes(kind):
 s=copy.deepcopy(spec())
 if kind=='lr':s['ppo']['learning_rate']=6e-5
 if kind=='quota':s['budget']['formal_updates']+=1
 if kind=='source':s['source_sha256']='x'*64
 with pytest.raises(ValueError):verify(s)
