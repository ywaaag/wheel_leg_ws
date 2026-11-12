"""Engine-specific Critic, reusing complete C/V4 and task reward memory."""

from types import SimpleNamespace

import numpy as np

from native_stand.simulation import jump_direct_critic
from native_stand.simulation.jump_isaac_bridge import execution_row
from native_stand.tasks.jump.controller_critic import NormalizedCritic, extend_fields
from native_stand.tasks.jump.isaac_contract import CRITIC, engine_fields
from native_stand.tasks.jump.pretraining import TaskMemory


class BaseCritic(jump_direct_critic.FullCritic):
    def contract(self):
        return {
            **super().contract(),
            "version": CRITIC + "_base",
            "scope": "actual PhysX fields plus complete C/V4 simulation feedback state",
        }


class IsaacCritic(NormalizedCritic):
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
        self.base.memory = TaskMemory(request["kind"], request["seconds"])
        head, self.base.schema = jump_direct_critic.pack(
            self.base,
            actor,
            wheel_cap=4.8,
            phase_count=10,
            schema_class=BaseCritic,
            engine_fields=values,
        )
        tail, fields = extend_fields({}, [], reward.critic_fields())
        self.tail_schema = NormalizedCritic.reuse(fields, self.tail_schema)
        self.schema = IsaacCritic.reuse(list(self.base.schema.fields) + fields, self.schema)
        return np.concatenate((head, self.tail_schema.encode(tail)))
