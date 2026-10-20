"""Full-episode teacher demonstrations with bounded attempts and causal checks.

Equivalent to old clean full-episode collection (rollin_schedule=[0]).
Historical scripts_v5/collect.py is untouched.
"""
from collect_recovery import collect as collect_continuation


def collect(task, policy, cfg):
    if cfg.get('rollin_schedule', [0]) != [0]:
        raise ValueError('Full demonstrations must start at reset; choose collector=recovery for roll-ins')
    effective = dict(cfg, rollin_method='teacher', takeover_at_episode_start=True,
                     takeover_schedule=[dict(name='episode_start', after_push_end_steps=-6)])
    return collect_continuation(task, policy, effective)
