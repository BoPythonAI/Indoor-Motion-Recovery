"""Use the teacher's clean-continuation collector, including actual student roll-in."""
import json
from pathlib import Path
from contracts import FEATURE_VERSION, ACTION_TARGET_VERSION, file_sha256


def run_collection(task, teacher, config):
    from collect_recovery import collect
    from collection_sampling import BalancedCases
    from controller import make_student
    from dataset import validate_episode
    config = dict(config)
    if task.feature_version != FEATURE_VERSION:
        raise ValueError('Surface collection requires current terrain and foot-clearance features')
    config.update(action_target_version=ACTION_TARGET_VERSION,
                  teacher_checkpoint_sha256=file_sha256(config['checkpoint']), frozen_prior=teacher.prior.identity)
    kind, method = config.get('collector', 'full'), config.get('rollin_method', 'teacher')
    if kind == 'full':
        if config.get('rollin_schedule', [0]) != [0]:
            raise ValueError('Use collector=recovery for a noisy or student roll-in')
        config.update(rollin_method='teacher', takeover_at_episode_start=True,
                      takeover_schedule=[dict(name='episode_start', after_push_end_steps=-6)])
        method = 'teacher'
    elif kind != 'recovery':
        raise ValueError('collector must be full or recovery')
    if method not in ('teacher', 'noisyteacher', 'bc', 'diffusion'):
        raise ValueError('Unsupported roll-in policy')
    rollin = make_student(task, config, prior=teacher.prior, checkpoint_key='rollin_student_checkpoint') if method in ('bc', 'diffusion') else None
    # collect() queries the clean teacher on the actual current state and stores
    # only record_start:successful_end, after the noise/student handover.
    original_reset, sampler = task.reset, None
    if config.get('balanced_collection'):
        sampler = BalancedCases(task.rows, config.get('collection_seed', config.get('seed', 17)),
            config.get('collection_strength_schedule', [0., .45, .75, .45, .75]),
            config.get('collection_directions', [0, 1, 2, 3]), config.get('collection_phases', [.3, .6]))
        if not config.get('fixed_attempts') or config['fixed_attempts'] % sampler.cycle_size:
            raise ValueError('Balanced collection requires complete fixed attempt cycles')

        def reset(ids=None, cases=None):
            if ids is not None or cases is not None:
                raise ValueError('Balanced collection resets complete batches only')
            return original_reset(cases=sampler.next_cases(task.n))
        task.reset = reset
    try:
        result = collect(task, teacher, config, rollin_policy=rollin)
    finally:
        task.reset = original_reset
        if sampler is not None:
            output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
            (output / 'sampling_report.json').write_text(json.dumps(sampler.report(), indent=2))
    # This catches a stale teacher collector before any dataset is used in training.
    import torch
    for path in sorted(Path(config['output']).glob('episode_*.pt')):
        validate_episode(torch.load(path, map_location='cpu', weights_only=False))
    return result
