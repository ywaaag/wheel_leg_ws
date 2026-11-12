"""同一次受控更新中的两个独立Adam；保留Actor KL回退所需的统一视图。"""

from collections import ChainMap

import torch


class DualAdam:
    VERSION = "supported_independent_actor_critic_Adam_v1"

    def __init__(self, actor, critic, *, actor_lr, critic_lr):
        self.actor = torch.optim.Adam([dict(params=list(actor), lr=actor_lr, owner="Actor")])
        self.critic = torch.optim.Adam([dict(params=list(critic), lr=critic_lr, owner="Critic")])

    @property
    def param_groups(self):
        return self.actor.param_groups + self.critic.param_groups

    @property
    def state(self):
        # KL回退只写Actor状态；Critic保留第一次step，不随Actor重试重复推进。
        return ChainMap(self.actor.state, self.critic.state)

    def zero_grad(self, *, set_to_none=True):
        self.actor.zero_grad(set_to_none=set_to_none)
        self.critic.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.actor.step()
        self.critic.step()

    def state_dict(self):
        return dict(
            version=self.VERSION, actor=self.actor.state_dict(), critic=self.critic.state_dict()
        )

    def load_state_dict(self, value):
        if value.get("version") != self.VERSION:
            raise ValueError("新接口需要两个独立Adam，不能加载旧单Adam状态")
        self.actor.load_state_dict(value["actor"])
        self.critic.load_state_dict(value["critic"])
