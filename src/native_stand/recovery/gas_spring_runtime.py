"""Explicit per-substep wrench adapter. Not enabled by any existing recipe."""
import torch
from .gas_spring import SpringLaw,EndStops,force_pair,with_end_stops

def rotate_xyzw(q,v):
    xyz=q[...,:3];return v+2*torch.cross(xyz,torch.cross(xyz,v,dim=-1)+q[...,3:]*v,dim=-1)

class GasSpringWrenches:
    def __init__(self,robot,geometry,law,stops,*,stop_mode='compliant'):
        if geometry['version']!='gas_spring_bilateral_planar_geometry_v2':raise ValueError('explicit bilateral geometry v2 required')
        if not isinstance(law,SpringLaw):raise ValueError('explicit force law required')
        if stop_mode not in ('compliant','knee_solver'):raise ValueError('unknown stop mode')
        if stop_mode=='compliant' and not isinstance(stops,EndStops):raise ValueError('explicit compliant stops required')
        if stop_mode=='knee_solver' and stops is not None:raise ValueError('do not double-count solver and compliant stops')
        self.stop_mode=stop_mode;self.calls=0
        if abs(law.compressed_m-.155)>1e-9 or abs(law.extended_m-.215)>1e-9:raise ValueError('measured stroke differs')
        self.robot=robot;self.law=law;self.stops=stops;self.rows=[]
        if [s['side'] for s in geometry['springs']]!=['left','right']:raise ValueError('two declared springs required')
        for spring in geometry['springs']:
            anchors=spring['anchors']
            if len(anchors)!=2:raise ValueError('two anchors per spring required')
            self.rows.append([(robot.body_names.index(a['body'].rsplit('/',1)[-1]),a['local_m']) for a in anchors])
        self.last=None

    def apply(self):
        """Call after base _apply_action and before scene.write_data_to_sim, every1ms.

        Uses add, never reset/set, so it cannot erase another external wrench.
        No motor target/effort limit API is touched.
        """
        d=self.robot.data;allforces=[];allpositions=[];ids=[];diagnostics=[]
        for row in self.rows:
            bid=[x[0] for x in row];q=d.body_link_quat_w.torch[:,bid];origin=d.body_link_pos_w.torch[:,bid]
            local=origin.new_tensor([x[1] for x in row]).expand_as(origin)
            pos=origin+rotate_xyzw(q,local);com=d.body_com_pos_w.torch[:,bid]
            vel=d.body_com_lin_vel_w.torch[:,bid];omega=d.body_com_ang_vel_w.torch[:,bid]
            r=force_pair(position_a=pos[:,0],position_b=pos[:,1],com_a=com[:,0],com_b=com[:,1],velocity_com_a=vel[:,0],velocity_com_b=vel[:,1],omega_a=omega[:,0],omega_b=omega[:,1],law=self.law)
            if self.stop_mode=='compliant':r=with_end_stops(r,pos[:,0],pos[:,1],com[:,0],com[:,1],self.law,self.stops)
            allforces.append(torch.stack((r['force_a_world'],r['force_b_world']),1));allpositions.append(pos);ids.extend(bid);diagnostics.append(r)
        # Both springs computed/validated before either is submitted.
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(forces=torch.cat(allforces,1),positions=torch.cat(allpositions,1),body_ids=torch.tensor(ids,device=origin.device),is_global=True)
        self.last=diagnostics
        self.calls+=1
        return diagnostics

def diagnostic_subclass(base):
    """Only an explicit diagnostic declaration selects this subclass."""
    class SpringDiagnosticEnv(base):
        def _apply_action(self):
            super()._apply_action()
            adapter=getattr(self,'gas_spring_adapter',None)
            if adapter is not None:
                result=adapter.apply()
                # Bounded diagnostic only: <=14000 substeps, no unbounded training log.
                import numpy as np
                self.gas_spring_samples.append(np.array([[r['length_m'][0].item(),r['force_n'][0].item(),r['net_force_world'][0].norm().item(),r['net_torque_about_origin_world'][0].norm().item()] for r in result]))
    return SpringDiagnosticEnv
