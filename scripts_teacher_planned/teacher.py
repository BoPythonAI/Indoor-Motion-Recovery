"""Independent planned-reference comparison, retaining each original controller."""


def train_or_evaluate(agent, cfg):
    if cfg.get('mode') == 'evaluate' and cfg.get('method') == 'base':
        baseline = dict(cfg)
        for key in ('checkpoint', 'resume', 'student_checkpoint'):
            baseline.pop(key, None)
        baseline['feature_version'] = 'v3_terrain_reference'
        from v5_task import train_or_evaluate as run_base
        return run_base(agent, baseline)
    from warm_runtime import train_or_evaluate as run_surface
    return run_surface(agent, cfg)
