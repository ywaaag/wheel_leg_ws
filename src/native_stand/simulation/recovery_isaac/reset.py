"""Reset-boundary batching; preserves per-environment course/DR call ordering."""

import torch


class InitialStates:
    """Immutable legal pose tensors, not substitutes for measured observations."""

    def __init__(self, catalog, joint_names, device):
        poses = catalog["poses"]
        self.indices = {name: i for i, name in enumerate(poses)}
        self.q = torch.tensor(
            [[p["joint_position"][n] for n in joint_names] for p in poses.values()], device=device
        )
        self.dq = torch.tensor(
            [[p["joint_velocity"][n] for n in joint_names] for p in poses.values()], device=device
        )
        self.root = torch.tensor([p["root_pose_xyzw"] for p in poses.values()], device=device)
        self.velocity = torch.tensor(
            [p["root_velocity_linear_angular"] for p in poses.values()], device=device
        )

    def select(self, specs, origins, *, height_conditioned):
        names = [
            (f"standing_h{s['height_m']:.2f}" if height_conditioned else "standing")
            if s["standing"]
            else "Swing"
            for s in specs
        ]
        ids = torch.tensor([self.indices[name] for name in names], device=self.q.device)
        q, dq, root, velocity = (x[ids] for x in (self.q, self.dq, self.root, self.velocity))
        root[:, :3] += origins
        return q, dq, root, velocity


def ended_snapshot(env, ids):
    """One compact D2H for IDs and all ended-episode consumers, no per-row reads.

    float64 exactly carries source float32 seconds and integer counters. Phase
    shortening is still computed in its original float32 before transport.
    """
    rows = len(ids)
    life = env.life
    done = life.last_result.get("done", torch.zeros_like(env.episode_steps, dtype=torch.bool))
    quality = (env.HP10_count[ids] >= 500) & (
        (env.HP10_sum[ids] / 500).clamp_min(0).sqrt().amax(-1) <= 0.1
    )
    zero = env.student_seconds[ids].new_zeros(rows)
    columns = [
        ids,
        env.episode_steps[ids],
        done[ids],
        life.success[ids],
        quality,
        env.student_seconds[ids],
        env.teacher_seconds[ids],
        life.handover_active[ids] if env.handover_mode else zero,
        life.handover_time[ids] if env.handover_mode else zero,
    ]
    packed = torch.stack([v.to(torch.float64) for v in columns], -1)
    phase = env.phase_readback
    if phase is not None:
        initial = phase.initial_geometry[ids, :2]
        minimum = phase.minimum_length[ids]
        packed = torch.cat(
            (
                packed,
                torch.cat(
                    (
                        initial,
                        minimum,
                        initial - minimum,
                        phase.maximum_swing_change[ids],
                        phase.maximum_height[ids, None],
                        phase.first_entry[ids, None],
                        phase.maximum_hold[ids, None],
                    ),
                    -1,
                ).to(torch.float64),
            ),
            -1,
        )
    timer = getattr(env, "reset_measurement", None)
    if timer is not None and packed.device.type != "cpu":
        timer.count("recovery.explicit_transfers")
        timer.count("recovery.explicit_transfer_bytes", packed.numel() * packed.element_size())
    return packed.cpu().tolist()


def phase_record(row):
    """Same fields/rounding as PhaseReadback.record, from the batch readback."""
    return dict(
        initial_leg_length_m=row[9:11],
        minimum_leg_length_m=row[11:13],
        maximum_shortening_m=row[13:15],
        maximum_swing_angle_change_rad=row[15:17],
        maximum_root_height_m=row[17],
        first_entry_s=row[18] if row[18] >= 0 else None,
        maximum_continuous_hold_s=row[19],
        scope="actual geometry extrema, not a contraction/rotation success certificate",
    )
