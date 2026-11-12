"""Recovery data adapter tests exercise pre-action/physical target alignment."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
import numpy as np
import pytest
from native_stand.students.recovery_distillation import extract_episode,phase_ids


def episode():
 n=1085;a={};a['actor_input']=np.zeros((n,1,31),np.float32);x=a['actor_input'][:,0];x[:,24]=.68;x[:,26]=x[:,28]=1
 a['actor_valid']=np.ones((n,1,1),bool);a['actor_valid'][:40]=False
 a['controller']=a['actor_valid'].astype(int)*3;a['first_episode_valid']=np.ones((n,1),bool)
 a['state']=np.zeros((n,1,39));a['state'][:,:,29]=1;a['policy_time_s']=np.arange(1,n+1)*.01
 a['active_indices']=[0,1,2,3,4,5];a['joint_q_pre']=np.zeros((n,1,6));a['joint_dq_pre']=np.zeros((n,1,6));a['previous_targets']=np.zeros((n,1,6));a['control']=np.zeros((n,1,54))
 a['control'][:,:,28:32]=[.15,.3,-.15,.3];a['control'][:,:,20]=.04;a['control'][:,:,23]=-.05
 r={'accepted':True,'hold_complete':True,'entry_index':84,'hold_index_half_open':[85,1085]}
 return a,r


def test_actual_pre_action_and_executed_label_order_and_initialization_exclusion():
 a,r=episode();z=extract_episode(a,r,0,np.zeros(6))
 assert z['x'].shape==(1045,31) and z['sample'][0]==40
 np.testing.assert_allclose(z['y'][0],[.1,.2,.04,-.1,.2,-.05],atol=1e-7)
 np.testing.assert_array_equal(np.bincount(z['stage']),[10,15,20,1000])
 assert np.all(z['x'][:,26]==1) and not z['x'][:,25].any()


@pytest.mark.parametrize('fault',['q','previous','reset','clock','target','task','gap'])
def test_bad_alignment_reset_missing_interval_and_unrepresentable_labels_rejected(fault):
 a,r=episode()
 if fault=='q':a['joint_q_pre'][50,0,0]=.2
 if fault=='previous':a['previous_targets'][50,0,0]=.2
 if fault=='reset':a['state'][50,0,31]=1
 if fault=='clock':a['policy_time_s'][50]+=.01
 if fault=='target':a['control'][50,0,28]=1.6
 if fault=='task':a['actor_input'][50,0,25]=1
 if fault=='gap':a['actor_valid'][50]=False
 with pytest.raises(ValueError):extract_episode(a,r,0,np.zeros(6))
