"""Dynamic single owner: prefix Motion -> Recovery -> confirming Motion.

Owner selection/expert query must never mutate physical/causal execution state.
"""


def recovery_owner(initialized, episode_steps, prefix_s, handover_active):
    return initialized & (episode_steps * 0.01 >= prefix_s) & ~handover_active
