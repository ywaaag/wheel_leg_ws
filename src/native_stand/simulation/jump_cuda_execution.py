"""Device C/FSM/PD and V4; PhysX remains external and owns integration."""

import torch

from native_stand.simulation.jump_cuda_control import CudaControl
from native_stand.simulation.jump_device_c import abi_field
from native_stand.simulation.jump_shared_c import Frame
from native_stand.simulation.jump_unified import UnifiedV4Execution


class CudaExecution(UnifiedV4Execution):
    def __init__(self, *args, cuda_output, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu = CudaControl(self.c, cuda_output, self.v4.device)

    def reset(self, ids, q, dq):
        self.gpu.sync_to_host()
        super().reset(ids, q, dq)
        self.gpu.reset_host_rows(ids.detach().cpu().tolist())

    def finalize_reset(self, ids):
        self.gpu.reset_host_rows(ids)

    def prepare_device(self, feedback, request, *, reset_ids=None):
        g = self.gpu
        t = self.tick
        for name, value in feedback.items():
            if name != "motion_entry":
                g.set_input(name, value, source=True)
        fields = dict(
            now=t,
            sensor_tick=t,
            imu_tick=t,
            feedback_mask=63,
            imu_valid=1,
            enabled_mode=1,
            key_down=request,
            upstream_owner=2,
            legacy_jump=0,
            motion_entry=feedback["motion_entry"],
            motion_success=1,
            support_tick=t,
            support_mask=3,
            support_model_valid=1,
            adopted_valid=1,
            adopted_sequence=g.field("adopted_sequence"),
            upstream_target=g.field("adopted"),
            feedback_tick=torch.full((self.c.n, 6), t, device=self.v4.device),
            command=self.v4.last_q.new_tensor([0, 0, 0.34]).expand(self.c.n, -1),
        )
        for name, value in fields.items():
            g.set_input(name, value, source=True)
        if reset_ids is None:
            g.inputs.copy_(g.source_inputs)
            g.launch(1, t)
        else:
            # FSM updates are skipped on untouched rows during a same-tick reset.
            # Save/restore device rows around the identical scalar-C invocation.
            ids = torch.as_tensor(reset_ids, device=self.v4.device)
            saved = g.states.clone()
            oldinputs = g.inputs.clone()
            g.inputs[ids] = g.source_inputs[ids]
            if t % 5 == 0:
                g.launch(1, t)
            keep = torch.ones(self.c.n, dtype=torch.bool, device=self.v4.device)
            keep[ids] = False
            g.states[keep] = saved[keep]
            g.inputs[keep] = oldinputs[keep]

    def observe_device(self):
        g, v = self.gpu, self.v4
        g.inputs.copy_(g.source_inputs)

        def f(k):
            return g.field(k, inputs=True)

        frame = torch.cat(
            (f("gyro") * 0.25, f("gravity"), f("q") / 1.5, f("dq") * 0.05), -1
        ).float()
        sensed = v.report(frame, v.last_q, v.last_dq)[:, :16]
        ages = v.effective_packet_age
        g.set_input("gyro", sensed[:, :3] / 0.25)
        g.set_input("gravity", sensed[:, 3:6])
        g.set_input("q", sensed[:, 6:10] * 1.5)
        g.set_input("dq", sensed[:, 10:16] / 0.05)
        for key in ("sensor_tick", "imu_tick"):
            g.set_input(key, f(key).long() - ages)
        g.set_input("feedback_tick", f("feedback_tick").long() - ages[:, None])
        g.launch(2, self.tick)
        # Same CPU bridge semantics: sensed C input stays until next FSM/PD.
        return g.actor.clone(), g.mask.bool().clone()

    def publish_raw(self, raw, mask=None):
        raw = raw.contiguous().float()
        mask = (
            torch.ones(self.c.n, dtype=torch.int32, device=self.v4.device)
            if mask is None
            else torch.as_tensor(mask, device=self.v4.device, dtype=torch.int32)
        )
        self.gpu.launch(3, self.tick, raw, mask)
        return self.gpu.rc

    def _pd(self, heldq, heldv, ignored_target):
        g, v = self.gpu, self.v4
        g.inputs.copy_(g.source_inputs)
        g.set_input("q", v.unpair(heldq)[:, [0, 1, 3, 4]] + v.bias)
        dq = g.field("dq", inputs=True).clone()
        dq[:, [0, 1, 3, 4]] = v.unpair(heldv)[:, [0, 1, 3, 4]].double()
        g.set_input("dq", dq)
        g.launch(4, self.tick)
        final = abi_field(g.frames, Frame, "final_model_nm").float()
        v.adopted.copy_(abi_field(g.frames, Frame, "adopted"))
        v.wheel_raw.copy_(abi_field(g.frames, Frame, "pd_nm")[:, [2, 5]])
        v.wheel_PD.copy_(final[:, [2, 5]])
        self.C_PD_calls += 1
        return v.pair(final)

    def step(self, q, dq):
        # Reuse original V4 integration; only its external PD is device C.
        if self.input_rows is None:
            self.input_rows = True
        return super().step(q, dq)

    def state_dict(self):
        self.gpu.sync_to_host()
        self.input_rows = list(self.c.inputs)
        return super().state_dict()

    def load_state_dict(self, value):
        super().load_state_dict(value)
        self.gpu.reset_host_rows(list(range(self.c.n)))
