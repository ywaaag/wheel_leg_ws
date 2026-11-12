"""Actor更新后KL回退；拒绝的Actor参数与Adam动量一起恢复。"""

from __future__ import annotations

import copy
import math

import torch


def bounded_actor_step(optimizer, actor, measure_kl, *, limit, max_backtracks):
    parameters = list(actor.parameters())
    ids = {id(p) for p in parameters}
    groups = [g for g in optimizer.param_groups if any(id(p) in ids for p in g["params"])]
    if len(groups) != 1 or any(id(p) not in ids for p in groups[0]["params"]):
        raise ValueError("Actor必须独占一个optimizer参数组")
    group = groups[0]
    learning_rate = group["lr"]
    values = [p.detach().clone() for p in parameters]
    moments = {p: copy.deepcopy(optimizer.state[p]) for p in parameters if p in optimizer.state}

    def restore():
        with torch.no_grad():
            for p, value in zip(parameters, values, strict=False):
                p.copy_(value)
        for p in parameters:
            if p in moments:
                optimizer.state[p] = copy.deepcopy(moments[p])
            else:
                optimizer.state.pop(p, None)

    trials = []
    try:
        for index in range(max_backtracks + 1):
            if index:
                restore()
                # Critic只在第一步推进一次；Actor重试不能重复推进Critic动量。
                for other in optimizer.param_groups:
                    if other is not group:
                        for p in other["params"]:
                            p.grad = None
            group["lr"] = learning_rate * 0.5**index
            optimizer.step()
            with torch.no_grad():
                kl = float(measure_kl())
            trials.append(kl)
            if math.isfinite(kl) and kl <= limit:
                return dict(
                    accepted=True,
                    attempts=index + 1,
                    backtracks=index,
                    post_kl=kl,
                    effective_actor_lr=group["lr"],
                    trials=trials,
                )
        restore()
        return dict(
            accepted=False,
            attempts=len(trials),
            backtracks=max_backtracks,
            post_kl=float(measure_kl()),
            effective_actor_lr=0.0,
            trials=trials,
        )
    except BaseException:
        restore()
        raise
    finally:
        group["lr"] = learning_rate
