FROZEN_TEST_SCOPE='PORTABLE'
import json
import pytest
from native_stand.command_v6.ledger import Ledger

def test_source_finalize_legacy_missing_accumulator_counts_prior_time(tmp_path):
 p=tmp_path/'budget.json';out=tmp_path/'source'
 book={'used':{'source_eval':1},'attempts':{
  'old':{'phase':'source_eval','status':'COMPLETED','wall_s':12.,'output_dir':'old'},
  'new':{'phase':'source_eval','status':'RUNNING','output_dir':str(out.resolve())}}}
 p.write_text(json.dumps(book));Ledger(p).finalize('new',out,'COMPLETED',exit_code=0,wall_s=3.)
 d=json.loads(p.read_text());assert d['source_eval_wall_used_s']==15. and d['used']['source_eval']==2
 assert d['attempts']['new']['status']=='COMPLETED'
 with pytest.raises(ValueError):Ledger(p).finalize('new',out,'COMPLETED',wall_s=3.)
 assert json.loads(p.read_text())==d

def test_source_finalize_preserves_explicit_accumulator(tmp_path):
 p=tmp_path/'budget.json';out=tmp_path/'source'
 p.write_text(json.dumps({'source_eval_wall_used_s':20.,'used':{'source_eval':0},'attempts':{
  'x':{'phase':'source_eval','status':'RUNNING','output_dir':str(out.resolve())}}}))
 Ledger(p).finalize('x',out,'COMPLETED',wall_s=4.)
 assert json.loads(p.read_text())['source_eval_wall_used_s']==24.
