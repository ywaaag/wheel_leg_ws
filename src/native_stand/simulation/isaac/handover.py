"""实例拥有的Teacher/Student路由；交接不重置机械状态、FIFO或previous。"""

import torch


class StudentOwnedClock:
    """组合既有统计时钟；Teacher不计入Student回合年龄。"""

    def __init__(self, clock, env):
        self.clock = clock
        self.env = env

    def observe(self, *args, active, **kwargs):
        return self.clock.observe(*args, active=active & self.env.student_actor_valid, **kwargs)

    def __getattr__(self, name):
        return getattr(self.clock, name)


class Handover:
    def __init__(self, student_decoder, teacher_decoder, teacher, *, fixed_prefix):
        self.student_decoder = student_decoder
        self.teacher_decoder = teacher_decoder
        self.teacher = teacher
        self.fixed_prefix = fixed_prefix

    def initialize(self, env, info):
        env.r4_prefix_teacher = self.teacher.to(env.device).eval().requires_grad_(False)
        env.r4_prefix_remaining = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env.r4_prefix_override = None
        env.r4_owner = torch.zeros_like(env.r4_prefix_remaining)
        env.r4_handoff_count = torch.zeros((), device=env.device, dtype=torch.long)
        env.r4_prefix_count = env.r4_handoff_count.clone()
        if hasattr(env, "rise_hold"):
            env.rise_hold = StudentOwnedClock(env.rise_hold, env)

    def ordinary(self, env):
        from native_stand.recovery.controlled_ppo import pre_action_valid
        from native_stand.recovery.domain import Mode

        return pre_action_valid(env.fsm.mode == int(Mode.INITIALIZING), env.initialization_checked)

    def configure(self, env, cases):
        env.r4_prefix_override = torch.tensor(
            [
                cases[min(i // 8, len(cases) - 1)].get("_prefix_intervals", 0)
                for i in range(env.num_envs)
            ],
            device=env.device,
        )
        env.identified_dr.purpose_override = [
            cases[min(i // 8, len(cases) - 1)].get("_recovery_purpose", "motion")
            for i in range(env.num_envs)
        ]

    def reset(self, env, ids):
        values = torch.tensor(env.stage_spec["prefix_intervals"], device=env.device)[
            env.command_index[ids]
        ]
        if env.r4_prefix_override is not None:
            values = env.r4_prefix_override[ids]
        elif not self.fixed_prefix:
            from runtime_support import noise

            u = noise(env.identified_dr.episodes[ids], ids, 1, 1217)[:, 0]
            values = torch.where(values > 0, (100 + (u + 1) * 100).round().long(), values)
        env.r4_prefix_remaining[ids] = values
        env.r4_owner[ids] = 0

    def actions(self, env, raw):
        from native_stand.students.step_checks import StepChecks

        checks = StepChecks()
        active = self.ordinary(env)
        teacher = active & (env.r4_prefix_remaining > 0)
        student = active & ~teacher
        height = env.height_adapter.target
        decoded = self.student_decoder.decode(raw, height, checks=checks)
        ids = teacher.nonzero(as_tuple=True)[0]
        if ids.numel():
            with torch.no_grad():
                mean = env.r4_prefix_teacher.mean(env.student_obs["student_actor"][ids])
                prefix = self.teacher_decoder.decode(mean, height[ids], checks=checks)
            decoded["requested_target"][ids] = prefix["requested_target"]
        # All tensor validity checks finish before any control/cache/clock mutation
        # or the caller's PD/physics path. Do not defer them across env.step().
        checks.verify(ValueError)
        env.supported_raw_action = raw.detach().clone()
        env.supported_requested_physical_target = decoded["requested_target"].detach().clone()
        env.supported_raw_clip = decoded["raw_clipped"].detach().clone()
        env.r4_handoff_count.add_((student & (env.r4_owner == 2)).sum())
        env.r4_owner.copy_(torch.where(student, 3, torch.where(teacher, 2, 0)))
        captured = getattr(env, "collection_control_owner", None)
        if captured is not None:
            # Keep the actual action owner across a subsequent partial reset.
            captured.copy_(env.r4_owner)
        env.r4_prefix_count.add_(teacher.sum())
        env.r4_prefix_remaining.sub_(teacher.long())
        # Unit conversion only. Mechanical protection and actual previous remain downstream.
        return decoded["requested_target"] / raw.new_tensor([1.5, 1.5, 100, 1.5, 1.5, 100]), student
