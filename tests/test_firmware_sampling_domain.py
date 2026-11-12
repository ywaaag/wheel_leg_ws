"""Pure declared-clock, coverage and flexible-transmission checks, no physics."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy

import numpy as np
import pytest

from native_stand.students.firmware_sampling import (
    CompliantTransmission, DriverPackets, PacketEnvelopeError, cases, profile, source_parameters, validate_profile)


def test_source_software_fixed_and_only_unknowns_or_task_phases_vary():
    rows=cases();assert len(rows)==47 and len({r['name'] for r in rows})==47
    for row in rows:
        p=row['parameters'];assert p['actor_age_ms']==p['pd_age_ms']==0
        assert p['friction']==.5 and p['armature']==.012 and p['zero_bias']==[0.]*4
        assert p['root_mass_scale']==p['root_inertia_scale']==p['spring_scale']==1
        assert row['sampling']['inference_ms']<10
    assert source_parameters()['motor_delay_ms']==[0]*6
    rows[0]['sampling']['leg_packet_phase_ms'][0]=100
    assert cases()[0]['sampling']['leg_packet_phase_ms'][0]==0


@pytest.mark.parametrize('key,value',[('actor_phase_ms',10),('robot_phase_ms',5),('motor_send_phase_ms',2),
    ('leg_packet_phase_ms',[0,0,0,2]),('leg_sample_age_ms',16),('inference_ms',4),
    ('internal_dq_tau_ms',float('nan')),('rotor_inertia',.012),('motor_before_robot',1)])
def test_invalid_or_ambiguous_profile_is_rejected(key,value):
    p=profile();p[key]=value
    with pytest.raises(ValueError):validate_profile(p)


def test_encoder_range_is_not_silently_clipped_and_resolution_is_declared():
    assert DriverPackets.word(-45,45,12)==0 and DriverPackets.word(45,45,12)==4095
    assert DriverPackets.word(0,45,12)==2048
    assert 90/4095==pytest.approx(.021978021978)
    with pytest.raises(ValueError):DriverPackets.word(45.01,45,12)


def test_disabled_flex_is_exact_and_uses_no_rng():
    rng=np.random.get_state();p=profile();q=np.arange(4)*.1
    f=CompliantTransmission(p,q,np.zeros(4));motor=np.arange(6,dtype=float)
    assert np.array_equal(f.step(motor,q,np.zeros(4)),motor)
    assert np.array_equal(rng[1],np.random.get_state()[1])


def test_passive_compliance_conserves_angular_momentum_and_dissipates_energy():
    p=profile();p.update(coupling_stiffness_nm_rad=300.,coupling_damping_nm_s_rad=.2)
    f=CompliantTransmission(p,np.zeros(4),np.zeros(4));f.theta[:]=.01
    last_energy=.5*f.k*np.sum(f.theta**2)
    for _ in range(50):
        before=f.omega.copy();out=f.step(np.zeros(6),np.zeros(4),np.zeros(4))
        assert np.allclose(out[[0,1,3,4]],-f.j*(f.omega-before)/.001)
        energy=.5*f.k*np.sum(f.theta**2)+.5*f.j*np.sum(f.omega**2)
        assert energy<=last_energy+1e-14;last_energy=energy
    assert last_energy<.02 # less than a third of the initial .06J, without drive


def test_total_armature_allocation_is_positive_without_double_counting():
    for row in cases():
        p=row['sampling'];total=row['parameters']['armature']
        if p['coupling_stiffness_nm_rad']:
            assert 0<total-p['rotor_inertia']<total
            assert (total-p['rotor_inertia'])+p['rotor_inertia']==total


def test_finite_wire_envelope_failure_is_distinct_from_nonfinite_runtime_fault():
    with pytest.raises(PacketEnvelopeError):DriverPackets.word(-49.778,45,12)
    with pytest.raises(ValueError) as failure:DriverPackets.word(float('nan'),45,12)
    assert not isinstance(failure.value,PacketEnvelopeError)


@pytest.mark.parametrize('stiffness',[100.,300.,900.])
def test_coupled_two_inertia_without_drive_does_not_create_oscillation_energy(stiffness):
    p=profile();p['coupling_stiffness_nm_rad']=stiffness
    f=CompliantTransmission(p,np.zeros(4),np.zeros(4));f.theta[:]=.01
    q=np.zeros(4);dq=np.zeros(4);energies=[]
    initial_energy=.5*stiffness*4*.01**2
    for _ in range(500):
        tau=f.step(np.zeros(6),q,dq)[[0,1,3,4]]
        dq+=tau*.001/.006;q+=dq*.001
        energies.append(.5*stiffness*np.sum((f.theta-q)**2)+.5*.006*np.sum(f.omega**2+dq**2))
    assert max(energies)<=initial_energy and energies[-1]<initial_energy*1e-6
