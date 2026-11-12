"""Host-dependent frozen-source consumers, isolated to preserve module identity."""
import json
from pathlib import Path
import subprocess
import sys

FROZEN_TEST_SCOPE='HOST'


def test_actual_frozen_Critic_timing_and_modal_event_consumers(tmp_path):
    result=subprocess.run([sys.executable,str(Path(__file__).parent/'fixtures/motion_host_consumer_probe.py'),str(tmp_path/'receipt.json')],
        check=True,capture_output=True,text=True)
    receipt=json.loads(result.stdout)
    assert receipt['Critic_width']==863
    assert all(receipt['checks'].values())
    assert receipt['modal_events_by_cohort'][:8]==[1]*8
    assert receipt['physics_steps']==receipt['PPO_updates']==0


def test_source3302_first_training_candidate_publication(tmp_path):
    result=subprocess.run([sys.executable,str(Path(__file__).parent/'fixtures/motion_training_probe.py'),str(tmp_path)],
        check=True,capture_output=True,text=True)
    receipt=json.loads(result.stdout)
    assert receipt['first_publication']
    assert receipt['source_sha256']=='739e38287a8a12508422291de0cc82c863ce3c4d3047f8ab2e5cc9867f947107'
    assert receipt['target_cohorts']==86
    assert receipt['extra_simulation_steps']==receipt['PPO_updates']==0


def test_actual_training_episode_denominators_and_new_world_restore():
    result=subprocess.run([sys.executable,str(Path(__file__).parent/'fixtures/motion_episode_probe.py')],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['early_failure_denominator']==1
    assert receipt['restored_unfinished_episodes']==receipt['extra_simulation_steps']==0
