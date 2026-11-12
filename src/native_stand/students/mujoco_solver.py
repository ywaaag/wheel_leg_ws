"""Explicit MuJoCo convergence/compliance profiles; never edit the source MJCF.

More iterations change numerical convergence. ``*_tight_*`` ALSO change the
model's constraint compliance, so they are diagnostic physical assumptions,
not a higher floating-point precision or a hardware calibration.
"""
import numpy as np


PROFILES = (
    'legacy_xml_v1',
    'convergence_v1',
    'closure_tight_v1',
    'wheel_contact_tight_v1',
    'closure_contact_tight_v1',
    'closure_contact_converged_v1',
)
DEFAULT_PROFILE = 'closure_contact_tight_v1'


def resolve(spec, requested=None):
    """New Motion31 runs use the paired tight profile; old formats stay explicit.

    A historical request can always select legacy_xml_v1. Never infer interface
    semantics from a Tensor shape or silently reinterpret an old request.
    """
    if requested is not None and requested not in PROFILES:
        raise ValueError('unknown MuJoCo solver profile: ' + str(requested))
    motion31 = (spec.get('interface_version') == 'multiskill_mcu_v2' and
                spec.get('student_training', {}).get('version') == 'student_motion31_training_v1')
    if not motion31:
        if requested not in (None, 'legacy_xml_v1'):
            raise ValueError('solver override requires the explicit Motion31 path')
        return 'legacy_xml_v1'
    return DEFAULT_PROFILE if requested is None else requested


def surface_ids(model, refs):
    wheels = np.flatnonzero(np.isin(model.geom_bodyid, refs['wheels']))
    return np.unique(np.r_[int(refs['ground']), wheels]).astype(int)


def readback(model, refs):
    ids = surface_ids(model, refs)
    opt = model.opt
    return dict(solver=int(opt.solver), iterations=int(opt.iterations),
        tolerance=float(opt.tolerance), ls_iterations=int(opt.ls_iterations),
        ls_tolerance=float(opt.ls_tolerance), timestep_s=float(opt.timestep),
        integrator=int(opt.integrator), cone=int(opt.cone),
        disableflags=int(opt.disableflags), enableflags=int(opt.enableflags),
        noslip_iterations=int(opt.noslip_iterations),
        equality_solref=model.eq_solref.tolist(), equality_solimp=model.eq_solimp.tolist(),
        surface_geom_ids=ids.tolist(), surface_solref=model.geom_solref[ids].tolist(),
        surface_solimp=model.geom_solimp[ids].tolist(),
        surface_friction=model.geom_friction[ids].tolist(),
        surface_condim=model.geom_condim[ids].tolist(),
        surface_margin=model.geom_margin[ids].tolist(),
        scope='wheel geoms and shared ground; ground compliance also affects other ground contacts; actual wheel/ground contacts must be read back')


def apply(model, refs, version=DEFAULT_PROFILE):
    """Apply to a freshly loaded model. Legacy is a strict no-op.

    Keep MuJoCo's reference safety clamp (time constant >= 2*dt). No friction,
    cone, collision geometry, limits, inertia, timestep or actuator writes.
    Applying a legacy profile does not undo a previous tight profile: callers
    must use a fresh model for each independent comparison.
    """
    if version not in PROFILES:
        raise ValueError('unknown MuJoCo solver profile: ' + str(version))
    before = readback(model, refs)
    if version != 'legacy_xml_v1':
        if not np.isclose(model.opt.timestep, .001, rtol=0, atol=1e-12):
            raise ValueError('solver profiles require the explicit 1ms physical contract')
        # Numeric enum 2 is Newton; preserve the solver rather than change it.
        if int(model.opt.solver) != 2 or int(model.opt.disableflags) & 4096:
            raise ValueError('profiles require Newton and the enabled reference safety clamp')
        model.opt.iterations = 300
        model.opt.tolerance = 1e-12
        model.opt.ls_iterations = 100
        model.opt.ls_tolerance = 1e-8
        closure = version in ('closure_tight_v1', 'closure_contact_tight_v1', 'closure_contact_converged_v1')
        contact = version in ('wheel_contact_tight_v1', 'closure_contact_tight_v1', 'closure_contact_converged_v1')
        if closure:
            model.eq_solref[:] = [.002, 1.]
            model.eq_solimp[:] = [.9999, .9999, .001, .5, 2.]
        if contact:
            ids = surface_ids(model, refs)
            model.geom_solref[ids] = [.002, 1.]
            model.geom_solimp[ids] = [.999, .9999, .0001, .5, 2.]
        if version == 'closure_contact_converged_v1':
            # Same compliance as combined; only convergence is tightened again.
            model.opt.iterations = 600
            model.opt.tolerance = 1e-14
            model.opt.ls_iterations = 200
            model.opt.ls_tolerance = 1e-10
    return dict(version=version, before=before, effective=readback(model, refs),
        changes_compliance=version not in ('legacy_xml_v1', 'convergence_v1'),
        certification=False)


class ConstraintRecorder:
    """Read solved constraints at 100Hz without mj_forward/extra physical steps.

    After mj_step, contact/kinematic arrays describe the last solver state,
    normally one physical tick before the integrated qpos. Explicit timestamps
    prevent presenting them as simultaneous post-integration measurements.
    """
    def __init__(self):
        self.contacts = {}
        self.last_integrated_time_s = 0.

    def __call__(self, model, data, refs):
        penetration = np.zeros(2)
        count = np.zeros(2, dtype=int)
        active = np.zeros(2, dtype=int)
        for c in data.contact:
            if refs['ground'] not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == refs['ground'] else c.geom1
            body = int(model.geom_bodyid[other])
            if body not in refs['wheels']:
                continue
            side = refs['wheels'].index(body)
            count[side] += 1
            active[side] += int(c.efc_address >= 0)
            penetration[side] = max(penetration[side], -float(c.dist))
            key = (side, int(c.dim), *c.solref.tolist(), *c.solimp.tolist(), *c.friction.tolist())
            self.contacts[key] = self.contacts.get(key, 0) + 1
        vectors = np.array([data.site_xpos[a] - data.site_xpos[b] for a, b in refs['loops']])
        equality = np.abs(data.efc_pos[np.asarray(data.efc_type) == 0])
        self.last_integrated_time_s = float(data.time)
        return dict(constraint_time_s=float(data.time - model.opt.timestep),
            closure_vectors_m=vectors, wheel_ground_penetration_m=penetration,
            wheel_ground_contact_count=count, wheel_ground_active_contact_count=active,
            equality_position_peak_m=float(equality.max()) if len(equality) else 0.,
            solver_iterations_total=int(np.sum(data.solver_niter)),
            solver_iterations_max=int(np.max(data.solver_niter)), constraint_rows=int(data.nefc))

    def contact_readback(self):
        return [dict(model_wheel_index=int(k[0]), condim=int(k[1]),
                     solref=list(k[2:4]), solimp=list(k[4:9]),
                     friction=list(k[9:]), sampled_contact_count=n)
                for k, n in sorted(self.contacts.items())]
