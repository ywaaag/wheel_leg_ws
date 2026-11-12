"""Check effective backend limits separately from actuator settings and caches."""
FROZEN_TEST_SCOPE='PORTABLE'
from types import SimpleNamespace as S
import pytest
import torch
from native_stand.control_readback import check_control


def environment(*,mcu=True):
    n,joints=2,10
    ids=[1,0,8,3,2,9]
    zero=torch.zeros(n,joints)
    limits=torch.full((n,joints),1.e9)
    legs=[ids[i] for i in (0,1,3,4)]
    limits[:,legs]=67. if mcu else 40.
    def actuator(columns,limit):
        return S(joint_indices=torch.tensor(columns),stiffness=torch.zeros(n,len(columns)),
                 damping=torch.zeros(n,len(columns)),effort_limit=torch.full((n,len(columns)),limit))
    groups={'leg_motors':actuator(legs,67. if mcu else 40.),
            'wheel_motors':actuator([8,9],5.5),
            'passive_linkage':actuator([4,5],50.),'guide_wheels':actuator([6,7],0.)}
    data=S(joint_stiffness=S(torch=zero.clone()),joint_damping=S(torch=zero.clone()),
           joint_effort_limits=S(torch=limits.clone()))
    view=S(get_dof_stiffnesses=lambda:zero,get_dof_dampings=lambda:zero,
           get_dof_max_forces=lambda:limits)
    env=S(device='cpu',ids=ids,robot=S(data=data,root_view=view,actuators=groups,
                                     joint_names=[str(i) for i in range(joints)]))
    if mcu:env.mcu_execution={}
    return env,limits


@pytest.mark.parametrize('mcu',[False,True])
def test_explicit_actuator_allows_large_backend_wheel_limit(mcu):
    env,limits=environment(mcu=mcu)
    result=check_control(env)
    assert result['active_effort_limits_match']
    assert result['backend_effort_limits_compatible']
    assert result['backend_effort_limits_nm'][0]==[67. if mcu else 40.]*2+[1.e9]+[67. if mcu else 40.]*2+[1.e9]


@pytest.mark.parametrize('channel,value',[(0,40.),(2,2.4),(3,float('nan'))])
def test_backend_mismatch_cannot_hide_behind_correct_cache(channel,value):
    env,limits=environment()
    limits[1,env.ids[channel]]=value
    # Actuator and joint cache still advertise the original correct values.
    with pytest.raises(RuntimeError,match='backend/cache effort limits'):
        check_control(env)


def test_stale_joint_cache_is_reported_separately_from_actual_backend():
    env,_=environment()
    env.robot.data.joint_effort_limits.torch[1,env.ids[1]]=40.
    with pytest.raises(RuntimeError,match='backend/cache effort limits'):
        check_control(env)


def test_readback_uses_physx_view_instead_of_relabeled_cache():
    env,limits=environment()
    limits[1,env.ids[0]]=80.
    result=check_control(env)
    assert result['backend_effort_limits_nm'][1][0]==80.
    assert result['joint_cache_effort_limits_nm'][1][0]==67.
