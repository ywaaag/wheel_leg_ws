"""Differentiable tree kinematics, SO(3) integration and explicit closure checks.

No contact solver or hidden stabilizer. All passive/guide coordinates are retained.
Predicted closure drift is penalized AND rejected, never silently projected away.
"""
import torch
from torch import nn


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)), -1)


def quat_exp(vector):
    theta = vector.norm(dim=-1, keepdim=True)
    # torch.sinc is finite and differentiable at zero.
    return torch.cat(((theta*.5).cos(), .5*torch.sinc(theta/(2*torch.pi))*vector), -1)


def quat_matrix(q):
    q = q/q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def rotate(r, x):
    return (r @ x.unsqueeze(-1)).squeeze(-1)


def integrate(q, v, dt=.001):
    rotation = quat_mul(q[:, 3:7], quat_exp(v[:, 3:6]*dt))
    rotation = rotation/rotation.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.cat((q[:, :3]+v[:, :3]*dt, rotation, q[:, 7:]+v[:, 6:]*dt), -1)


def rotation_error(a, b):
    # Double cover invariant chord distance, angle separately reported in radians.
    a = a/a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = b/b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.minimum((a-b).square().sum(-1), (a+b).square().sum(-1))


class Kinematics(nn.Module):
    def __init__(self, description):
        super().__init__()
        self.description = description
        for key in ('pos', 'quat', 'axis', 'anchor', 'ipos', 'iquat', 'mass', 'inertia',
                    'q0', 'site_pos', 'geom_pos', 'geom_quat', 'geom_size', 'armature', 'limits', 'solver_actuator_limits',
                    'all_joint_limits','all_joint_limited'):
            self.register_buffer(key, torch.tensor(description[key], dtype=torch.float32))
        self.parent = description['parent']
        self.joint_q = description['joint_q']
        self.joint_v = description['joint_v']
        self.root = description['root']
        self.active_q = description['active_q']
        self.active_v = description['active_v']
        self.wheels = description['wheels']
        self.hips = description['hips']
        self.loops = description['loops']
        self.site_body = description['site_body']
        self.contact_geoms = description['contact_geoms']
        self.geom_body = description['geom_body']

    def forward(self, q, v):
        n = len(q)
        zero = q.new_zeros(n, 3)
        eye = torch.eye(3, device=q.device, dtype=q.dtype).expand(n, -1, -1)
        positions, rotations, linear, angular = [zero], [eye], [zero], [zero]
        for body in range(1, len(self.parent)):
            if body == self.root:
                p, r, lv = q[:, :3], quat_matrix(q[:, 3:7]), v[:, :3]
                av = rotate(r, v[:, 3:6])
            else:
                parent = self.parent[body]
                pr, pp, pl, pa = rotations[parent], positions[parent], linear[parent], angular[parent]
                base = pr @ quat_matrix(self.quat[body]).expand(n, -1, -1)
                p = pp + rotate(pr, self.pos[body].expand(n, -1))
                r = base
                lv = pl+torch.cross(pa, p-pp, dim=-1)
                av = pa
                jq, jv = self.joint_q[body], self.joint_v[body]
                if jq >= 7:
                    delta = q[:, jq]-self.q0[jq]
                    local = quat_matrix(quat_exp(delta[:, None]*self.axis[body]))
                    anchor = self.anchor[body].expand(n, -1)
                    world_anchor = p+rotate(base, anchor)
                    p = p+rotate(base, anchor-rotate(local, anchor))
                    axis = rotate(base, self.axis[body].expand(n, -1))
                    r = base @ local
                    av = pa+axis*v[:, jv, None]
                    lv = pl+torch.cross(pa, p-pp, dim=-1)+torch.cross(axis*v[:, jv, None], p-world_anchor, dim=-1)
            positions.append(p); rotations.append(r); linear.append(lv); angular.append(av)
        p, r, lv, av = map(lambda x: torch.stack(x, 1), (positions, rotations, linear, angular))
        com = p+rotate(r, self.ipos.expand(n, -1, -1))
        cv = lv+torch.cross(av, com-p, dim=-1)
        site = p[:, self.site_body]+rotate(r[:, self.site_body], self.site_pos.expand(n, -1, -1))
        residual = torch.stack([site[:, a]-site[:, b] for a, b in self.loops], 1)
        root_r = r[:, self.root]
        body_w = rotate(root_r.transpose(-1, -2), av[:, self.root])
        gravity = rotate(root_r.transpose(-1, -2), q.new_tensor([0., 0., -1.]).expand(n, -1))
        sensors = torch.cat((body_w, gravity, q[:, self.active_q], v[:, self.active_v]), -1)
        body_linear = rotate(root_r.transpose(-1, -2), cv[:, self.root])
        velocity = torch.stack((-body_linear[:, 2], -body_linear[:, 0], body_linear[:, 1]), -1)
        cad = q.new_tensor([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
        upright = root_r @ cad.T
        pitch = torch.asin((-upright[:, 2, 0]).clamp(-1+1e-7, 1-1e-7))
        roll = torch.atan2(upright[:, 2, 1], upright[:, 2, 2])
        hip = torch.stack([p[:, b]+rotate(r[:, b], self.anchor[b].expand(n, -1)) for b in self.hips], 1)
        legs = rotate(root_r[:, None].transpose(-1, -2), p[:, self.wheels]-hip)
        lengths = legs.norm(dim=-1)
        swing = torch.atan2(-legs[..., 2], -legs[..., 1])
        swing_diff = torch.atan2((swing[:, 0]-swing[:, 1]).sin(), (swing[:, 0]-swing[:, 1]).cos())
        wheel_offset = -rotate(root_r.transpose(-1, -2), p[:, self.wheels[0]]-p[:, self.wheels[1]])[:, 2]
        forward = -root_r[:, :, 2]
        from native_stand.students.motion_yaw_quality import measurement
        spin = measurement(p[:, self.wheels], com[:, self.wheels], cv[:, self.wheels], av[:, self.wheels],
                           r[:, self.wheels, :, 0], forward)
        penetrations = [];wheel_gaps={}
        for g in self.contact_geoms:
            b = self.geom_body[g]
            center = p[:, b]+rotate(r[:, b], self.geom_pos[g].expand(n, -1))
            gr = r[:, b] @ quat_matrix(self.geom_quat[g]).expand(n, -1, -1)
            axial_z = gr[:, 2, 2]
            extent = self.geom_size[g, 0]*(1-axial_z.square()).clamp_min(1e-12).sqrt()+self.geom_size[g, 1]*axial_z.abs()
            penetrations.append(torch.relu(extent-center[:, 2]))
            if b in self.wheels:wheel_gaps[self.wheels.index(b)]=center[:,2]-extent
        ir = r @ quat_matrix(self.iquat).expand(n, -1, -1, -1)
        iw = rotate(ir.transpose(-1, -2), av)
        kinetic = .5*(self.mass*cv.square().sum(-1)+(self.inertia*iw.square()).sum(-1)).sum(-1)
        kinetic = kinetic+.5*(self.armature*v.square()).sum(-1)
        energy = kinetic+(self.mass*com[..., 2]*9.81).sum(-1)
        return dict(p=p, r=r, linear=lv, angular=av, com=com, com_velocity=cv, sensors=sensors,
                    closure=residual, velocity=velocity, yaw=body_w[:, 1], height=p[:, self.root, 2],
                    angles=torch.stack((pitch, roll), -1), lengths=lengths, swing_diff=swing_diff,
                    wheel_offset=wheel_offset, spin=spin, energy=energy,
                    wheel_ground_gap=torch.stack([wheel_gaps[i] for i in range(2)],-1),
                    penetration=torch.stack(penetrations, -1) if penetrations else q.new_zeros(n, 1))

    def valid(self, q, v, measured=None):
        m = self(q, v) if measured is None else measured
        return (torch.isfinite(q).all(-1) & torch.isfinite(v).all(-1)
                & (q[:, 3:7].norm(dim=-1)-1).abs().lt(1e-4)
                & (((q[:,7:]>=self.all_joint_limits[:,0]-.01)&(q[:,7:]<=self.all_joint_limits[:,1]+.01))
                   | ~self.all_joint_limited.bool()).all(-1)
                & m['closure'].norm(dim=-1).amax(-1).le(.005)
                & m['penetration'].amax(-1).le(.01))

    def contact_valid(self,state):
        # Force labels belong to the last solver pose, not post-integration q.
        m=self(state.solver_q,state.solver_v)
        return ~((m['wheel_ground_gap']>.005)&(state.contact>2)).any(-1)

    def effective_torque(self, requested):
        return requested.clamp(-self.solver_actuator_limits,self.solver_actuator_limits)


class ClosureOnlyKinematics(nn.Module):
    """Exact closure FK for mechanics supervision, never physical acceptance.

    Keep precisely the loop-site ancestor tree and the original arithmetic order.
    No velocity/energy/contact features, projection, or altered loop geometry.
    Full Kinematics still supplies reward and physical validity checks elsewhere.
    """
    def __init__(self, geometry):
        super().__init__()
        self.geometry=geometry
        self.sites=sorted({s for pair in geometry.loops for s in pair})
        needed={0,geometry.root}
        for site in self.sites:
            body=geometry.site_body[site]
            while body not in needed:
                needed.add(body);body=geometry.parent[body]
        self.bodies=sorted(needed-{0})

    def forward(self,q,v):
        g=self.geometry;n=len(q)
        positions={0:q.new_zeros(n,3)}
        rotations={0:torch.eye(3,device=q.device,dtype=q.dtype).expand(n,-1,-1)}
        for body in self.bodies:
            if body==g.root:p,r=q[:,:3],quat_matrix(q[:,3:7])
            else:
                parent=g.parent[body];pr,pp=rotations[parent],positions[parent]
                base=pr@quat_matrix(g.quat[body]).expand(n,-1,-1)
                p=pp+rotate(pr,g.pos[body].expand(n,-1));r=base
                jq=g.joint_q[body]
                if jq>=7:
                    delta=q[:,jq]-g.q0[jq]
                    local=quat_matrix(quat_exp(delta[:,None]*g.axis[body]))
                    anchor=g.anchor[body].expand(n,-1)
                    p=p+rotate(base,anchor-rotate(local,anchor));r=base@local
            positions[body]=p;rotations[body]=r
        site={s:positions[g.site_body[s]]+rotate(rotations[g.site_body[s]],g.site_pos[s].expand(n,-1)) for s in self.sites}
        return dict(closure=torch.stack([site[a]-site[b] for a,b in g.loops],1))
