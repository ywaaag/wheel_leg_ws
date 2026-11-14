"""Engine-specific Critic, reusing complete C/V4 and task reward memory."""

from types import SimpleNamespace

import numpy as np
import torch

from native_stand.simulation import jump_direct_critic
from native_stand.simulation.jump_isaac_bridge import execution_row
from native_stand.tasks.jump.controller_critic import PackedCritic, extend_fields
from native_stand.tasks.jump.isaac_contract import CRITIC, engine_fields
from native_stand.tasks.jump.pretraining import TaskMemory


class BaseCritic(PackedCritic):
    def contract(self):
        return {
            **super().contract(),
            "version": CRITIC + "_base",
            "scope": "actual PhysX fields plus complete C/V4 simulation feedback state",
        }


class IsaacCritic(PackedCritic):
    def contract(self):
        return {
            **super().contract(),
            "version": CRITIC,
            "scope": "actual PhysX/C/V4/reward; no MuJoCo solver padding",
            "physical_snapshot_bitwise": False,
        }


class RowCritic:
    def __init__(self, joints):
        self.fields = engine_fields(joints)
        self.schema = self.tail_schema = None
        self.base = SimpleNamespace(schema=None)

    def pack(self, execution, index, actor, physical, reward, request):
        if set(physical) != set(self.fields):
            raise ValueError("Actual Isaac fields missing/extra")
        values = {}
        for name, (shape, scale, unit) in self.fields.items():
            value = np.asarray(physical[name])
            if value.shape != shape:
                raise ValueError("Actual Isaac shape " + name)
            values[name] = (value, scale, unit)
        self.base.plant = SimpleNamespace(execution=execution_row(execution, index))
        try:
            self.base.memory = TaskMemory(request["kind"], request["seconds"])
            head, self.base.schema = jump_direct_critic.pack(
                self.base,
                actor,
                wheel_cap=4.8,
                phase_count=10,
                schema_class=BaseCritic,
                engine_fields=values,
            )
        finally:
            # A terminal row is a view into a full-batch CPU snapshot.
            # Keeping it on one of thousands of row packers pins that entire
            # allocation until this particular environment next terminates.
            del self.base.plant
        tail, fields = extend_fields({}, [], reward.critic_fields())
        if self.tail_schema is None:
            # Versioned source defines units/scales, not changing physical
            # values. Freeze that layout at the first actual observation.
            # encode below still validates every dynamic name/shape/value.
            self.tail_schema = PackedCritic(fields)
            self.schema = IsaacCritic(list(self.base.schema.fields) + fields)
        return np.concatenate((head, self.tail_schema.encode(tail)))


class BatchCritic:
    """Keep native histories on device; upload only the actual CPU C/task state.

    RowCritic establishes the versioned schema and remains the terminal/debug
    reference. No history is shortened, and normalization retains float64
    arithmetic before the original float32 output conversion.
    """

    def __init__(self, env, actor, host):
        from native_stand.tasks.jump.controller_critic import controller_fields

        self.controller_fields = controller_fields
        env.row_observation(0, host, actor[0], execution=env.critic_snapshot())
        self.schema = env.critics[0].schema
        self.fields = self.schema.fields
        self.layout = env.critics[0].base._controller_layout
        self.n, self.device = env.num_envs, env.device
        self.scales = torch.as_tensor(
            np.concatenate([np.broadcast_to(f.scale, f.shape).reshape(-1) for f in self.fields]),
            device=self.device,
        )
        memory = TaskMemory(env.requests[0]["kind"], env.requests[0]["seconds"]).state_dict()
        self.constant = {
            "task." + k: np.asarray(memory[k])
            for k in (
                "elapsed_s",
                "horizon_s",
                "stable_s",
                "had_support",
                "actual_takeoff",
                "previous_contact",
                "landing_paid",
            )
        }
        for name, value in memory["reward"].items():
            if name == "initial_kind":
                continue
            if value is None:
                self.constant["reward." + name + ".known"] = np.asarray(0.0)
                value = 0.0
            elif name in {"takeoff_COM_height_m", "peak_COM_height_m"}:
                self.constant["reward." + name + ".known"] = np.asarray(1.0)
            self.constant["reward." + name] = np.asarray(value)

    def pack(self, env, actor):
        n, tick, v = self.n, env.execution.tick, env.execution.v4
        values = {}
        from native_stand.simulation.jump_device_c import controller_device_fields

        values.update(controller_device_fields(env.execution, self.device))
        if hasattr(env, "device_reward"):
            values.update(env.device_reward.critic_values())
        else:
            rows = [r.critic_fields() for r in env.rewards]
            for key in rows[0]:
                values[key] = torch.as_tensor(
                    np.stack([row[key][0] for row in rows]), device=self.device
                )
        values["Actor.delayed37"] = torch.as_tensor(actor, device=self.device)
        m = env.scene.measured()
        values.update(
            {
                "Isaac." + key: value
                for key, value in dict(
                    root_relative_state=m["root"],
                    all_joint_q_dq_ddq=m["joint"],
                    COM_height_velocity=torch.cat((m["COM"][:, 2:3], m["COMv"]), -1),
                    leg_length_swing_spring=torch.cat((m["length"], m["swing"], m["spring"]), -1),
                    closure_residuals=m["closure"],
                    wheel_normal_slip_clearance=torch.cat(
                        (m["normal"], m["slip"], m["clearance"]), -1
                    ),
                    motor_NET_passive=torch.stack((env.motor, env.NET, env.passive), 1),
                    root_mass_COM=torch.as_tensor(env._root_mass_COM, device=self.device),
                    material_min_max=torch.as_tensor(env._material_bounds, device=self.device),
                    elapsed_request=torch.tensor(
                        [
                            [
                                (tick - env.starts[i]) * 0.001,
                                1.5 if r["kind"] == "ground_active_jump" else 0.0,
                                float(r["kind"] == "ground_active_jump"),
                            ]
                            for i, r in enumerate(env.requests)
                        ],
                        dtype=torch.float64,
                        device=self.device,
                    ),
                ).items()
            }
        )
        values["execution.subcycle"] = self.scales.new_tensor([tick % 5, tick % 10]).expand(n, -1)
        histories = {"targets", "origins", "wheel_queue", "motors", "NETs"}
        policy = {
            "packet",
            "qhistory",
            "dqhistory",
            "target_policy_history",
            "motor_policy_history",
        }
        stamps = {"request_tick", "adopted_origin", "birth", "observed_tick"}
        for name, value in vars(v).items():
            if not torch.is_tensor(value) or name in {"ids", "columns", "sign"}:
                continue
            key = "V4." + name
            if name in stamps:
                values[key + ".valid"] = value >= 0
                value = torch.where(value >= 0, tick - value, 0).double()
            elif name == "origins":
                value = tick - value
            if name in histories | policy:
                shift = -tick % len(value) if name in histories else -(tick // 10 + 1) % len(value)
                value = torch.roll(value, shift, 0).movedim(1, 0)
            elif name in {"theta", "omega"}:
                value = value.reshape(n, 2, *value.shape[1:])
            values[key] = value
        for name, value in vars(v.leg).items():
            if not torch.is_tensor(value) or name == "ids":
                continue
            if name in {"qhist", "vhist", "queue"}:
                value = torch.roll(value, -tick % len(value), 0)
                value = value.reshape(len(value), n, 2, *value.shape[2:]).movedim(1, 0)
            else:
                value = value.reshape(n, 2, *value.shape[1:])
            values["response." + name] = value
        for key, value in self.constant.items():
            values[key] = torch.as_tensor(value, device=self.device).expand(n, *value.shape)
        values["task.horizon_s"] = self.scales.new_tensor([r["seconds"] for r in env.requests])
        values["task.landing_only"] = self.scales.new_tensor(
            [r["kind"] == "landing_only" for r in env.requests]
        )
        if set(values) != {f.name for f in self.fields}:
            raise ValueError("Batched Isaac Critic source fields differ")
        output = []
        for f in self.fields:
            output.append(values[f.name].reshape(n, -1).double())
        result = (torch.cat(output, -1) / self.scales).float()
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Batched Isaac Critic nonfinite")
        env.measurement.count(
            "jump_critic_device_packs"
            if hasattr(env.execution, "gpu")
            else "jump_critic_CPU_batch_uploads"
        )
        return result
