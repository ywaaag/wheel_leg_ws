"""The exact NEW Recovery878 column layout, supplied by causal physical data."""

from __future__ import annotations

import torch

from native_stand.tasks.recovery.cubic_task import BASE_FIELDS, EXTRA_FIELDS

from .execution import SCALE


def pack(values, extra):
    if set(values) != {k for k, _ in BASE_FIELDS} or set(extra) != {k for k, _ in EXTRA_FIELDS}:
        raise ValueError("Missing/extra Recovery878 source")
    out = []
    n = None
    for data, fields in [(values, BASE_FIELDS), (extra, EXTRA_FIELDS)]:
        for key, width in fields:
            v = data[key]
            if v.ndim != 2 or v.shape[1] != width:
                raise ValueError(
                    "Actual named Critic source "
                    + key
                    + " shape="
                    + str(tuple(v.shape))
                    + " value="
                    + str(v.detach().cpu().tolist())
                )
            if n is None:
                n = len(v)
            if len(v) != n:
                raise ValueError("Mixed environment rows")
            out.append(v)
    packed = torch.cat(out, -1)
    # One safety readback at the same observation boundary, not one per field.
    if not bool(torch.isfinite(packed).all()):
        bad = next(
            key
            for data in (values, extra)
            for key, value in data.items()
            if not bool(torch.isfinite(value).all())
        )
        raise ValueError("Actual named Critic source " + bad + " contains nonfinite values")
    return packed


def sources(
    ex,
    life,
    actor,
    *,
    true_frame,
    q,
    dq,
    state,
    root_position,
    root_velocity,
    physical_profile,
    friction,
    armature,
):
    n = ex.n
    scale = q.new_tensor(SCALE)
    r = ex.leg

    def native(v):
        return v.reshape(n, -1)

    phase = torch.where(
        life.entry < 0,
        torch.where(life.elapsed == 0, 0, 1),
        torch.where(
            life.elapsed - life.entry < 0.5, 2, torch.where(life.elapsed - life.entry < 2, 3, 4)
        ),
    )
    age = torch.cat(
        (
            r.delay.reshape(n, 2),
            r.qage.reshape(n, 2),
            r.vage.reshape(n, 2),
            ex.packet_age[:, None],
            q.new_full((n, 1), ex.tick % 5),
        ),
        -1,
    )
    values = dict(
        actor=actor,
        true_frame=true_frame,
        q=q / scale,
        dq=dq * 0.05,
        requested=ex.requested / scale,
        adopted=ex.adopted / scale,
        target_history20=ex.recent(ex.targets, 20) / scale.repeat(20),
        joint_history10=torch.cat(
            (
                ex.recent(ex.qhistory, 10, policy=True) / scale.repeat(10),
                ex.recent(ex.dqhistory, 10, policy=True) * 0.05,
            ),
            -1,
        ),
        motor_history24=ex.recent(ex.motors, 24) / ex.caps.repeat(1, 24),
        NET_history24=ex.recent(ex.NETs, 24) / 40,
        physical_packets3=ex.recent(ex.packet, 3, policy=True),
        fitted_parameters=native(r.x),
        fitted_reference=native(r.ref) / 1.5,
        observer=torch.cat(
            (
                native(r.heldq)
                + native(
                    ex.pair(
                        torch.stack(
                            (
                                ex.bias[:, 0],
                                ex.bias[:, 1],
                                torch.zeros_like(ex.bias[:, 0]),
                                ex.bias[:, 2],
                                ex.bias[:, 3],
                                torch.zeros_like(ex.bias[:, 0]),
                            ),
                            -1,
                        )
                    )
                ),
                native(r.heldv),
                native(ex.theta),
                native(ex.omega),
            ),
            -1,
        ),
        native_execution=torch.cat(
            (
                native(r.actual),
                native(r.des_unclipped),
                native(ex.theta),
                native(ex.omega),
                native(r.filtered),
            ),
            -1,
        ),
        physical_profile=physical_profile,
        kinematics=torch.cat((state, root_position, root_velocity), -1),
        wheel_actual=ex.motor[:, [2, 5]] / ex.caps[:, [2, 5]],
        sensor_ages=age / age.new_tensor([40, 40, 10, 10, 10, 10, 30, 5]),
        PD_gains=q.new_tensor([40] * 4 + [2] * 4 + [0.2] * 2).expand(n, -1),
        motor_caps=ex.caps,
        wheel_timing=torch.cat((ex.wheel_delay, q.new_zeros(n, 2)), -1),
        previous_policy_target=ex.recent(ex.target_policy_history, 1, policy=True) / scale,
        previous_policy_motor=ex.recent(ex.motor_policy_history, 1, policy=True) / ex.caps,
        quality_timers=torch.stack(
            (
                life.dwell,
                life.hold,
                (abs(state[:, 4]) > 0.1).to(q.dtype),
                (abs(state[:, 5]) > 0.1).to(q.dtype),
                (~(state[:, 9:11] > 2).all(-1)).to(q.dtype),
                (state[:, 11] > 0.005).to(q.dtype),
                (state[:, 3] < 0.3).to(q.dtype),
            ),
            -1,
        ),
        phase_onehot=torch.nn.functional.one_hot(phase.long(), 5).to(q.dtype),
        task_onehot=q.new_tensor([0, 1, 0]).expand(n, -1),
        encoder_bias=ex.bias,
        motor_strength=torch.ones_like(q),
        passive_friction=friction,
        armature=armature,
        target_history_ages20=q.new_tensor([0.001 * i for i in range(1, 21)]).expand(n, -1),
    )
    return values


def observe(self):
    """Single causal observation route for ordinary and pre-reset terminal states."""
    frame, q, v, state, pos, vel, clearance, restricted, angular = self.measured()
    actor = self.execution.report(frame, q, v)
    if self.height_conditioned:
        actor[:, 24] = self.commanded_height() / 0.5
    values = sources(
        self.execution,
        self.life,
        actor,
        true_frame=frame,
        q=q,
        dq=v,
        state=state,
        root_position=pos - self.scene.env_origins,
        root_velocity=vel,
        physical_profile=self.physical_profile(),
        friction=self.physics.coulomb[:, self.ids],
        armature=self.physics.armature[:, self.ids],
    )
    critic = pack(values, self.life.extra(state, clearance, restricted, self.initialized))
    from native_stand.tasks.recovery import reward_context

    if getattr(self.cfg, "recovery_critic_version", None) in (
        reward_context.VERSION,
        "recovery_handover_causal908_v4",
        "recovery_standalone_height_causal909_v5",
    ):
        critic = reward_context.append(
            critic,
            reward_context.sources(
                self.execution,
                self.life,
                prefix_s=self.prefix_s,
                next_student=self.recovery_control_owner(),
                pulse_onset=self.pulse_onset,
                pulse_pattern=self.pulse_pattern,
                pulse_enabled=self.pulse_enabled,
            ),
        )
    from native_stand.tasks.recovery import handover_context

    if self.handover_mode:
        critic = handover_context.append(critic, self.life)
    if self.height_conditioned:
        from native_stand.tasks.recovery.height_context import append as height_append

        critic = height_append(
            critic, self.height_initial, self.commanded_height(), self.height_dynamic
        )
    valid = (
        self.recovery_control_owner()
        & (state[:, 11] <= 0.005)
        & (state[:, 12] >= 0.5)
        & (state[:, 3] >= 0.10)
    )
    self._obs = dict(
        student_actor=actor, student_critic=critic, recovery_actor_valid=valid[:, None]
    )
    return self._obs


def physical_profile(self):
    if getattr(self, "_physical_profile_cache", None) is not None:
        return self._physical_profile_cache
    rows = []
    p = self.physics
    for _i, d in enumerate(self.draws):
        rows.append(
            [
                d["root_mass_scale"],
                *[d["root_inertia_scale"]] * 3,
                *d["root_COM_m"],
                d["friction"],
                *d["encoder_bias_rad"],
                *[1.0] * 6,
                d["Actor_age_ms"],
                d["target_delay_ms"],
                *d["wheel_delay_ms"],
            ]
        )
    self._physical_profile_cache = torch.cat(
        (
            torch.tensor(rows, device=self.device, dtype=self.effort.dtype),
            p.mass[:, 0:1],
            torch.diagonal(p.inertia[:, 0].reshape(self.num_envs, 3, 3), dim1=-2, dim2=-1),
            p.com[:, 0, :3],
            self.effort.new_tensor([0.0, 0.0, 1.0]).expand(self.num_envs, -1),
        ),
        -1,
    )
    return self._physical_profile_cache
