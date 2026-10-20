# Official frozen actor helpers. No gates.
import dataclasses
import enum
import inspect
import json
from pathlib import Path
import torch

def _canonical(value):
    """Compare configuration semantics, including tensors and bound context paths."""
    if isinstance(value, torch.Tensor):
        return dict(dtype=str(value.dtype), shape=list(value.shape), values=value.cpu().tolist())
    if isinstance(value, enum.Enum):
        return dict(enum=f'{type(value).__module__}.{type(value).__qualname__}', value=value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if inspect.isfunction(value) or inspect.isclass(value):
        value = inspect.unwrap(value)
        return f'{value.__module__}.{value.__qualname__}'
    if type(value).__name__ == 'MdpComponent':
        return {k: _canonical(getattr(value, k)) for k in
                ['compute_func', 'dynamic_vars', 'static_params']}
    if type(value).__name__ == 'FieldPath':
        return {k: _canonical(getattr(value, k)) for k in ['name', '_parent_path']}
    if dataclasses.is_dataclass(value):
        return {f.name: _canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
    raise TypeError(f'Unknown configuration type: {type(value)}')

def _diff(left, right, path=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append(path + '.' + key + ': missing')
            else:
                result.extend(_diff(left[key], right[key], path + '.' + key))
        return result
    return [] if left == right else [f'{path}: runtime={left!r}, terrain={right!r}']

def check_runtime_contract(env, terrain_configs):
    """Fail on observation/actuation incompatibility before any blended rollout."""
    expected_env = terrain_configs['env']
    pairs = {
        'observation_components': (env.config.observation_components, expected_env.observation_components),
        'control_components': (env.config.control_components, expected_env.control_components),
        'action_config': (env.config.action_config, expected_env.action_config),
        'num_state_history_steps': (env.config.num_state_history_steps, expected_env.num_state_history_steps),
        'robot_kinematic_info': (env.robot_config.kinematic_info, terrain_configs['robot'].kinematic_info),
        'robot_control': (env.robot_config.control, terrain_configs['robot'].control),
        'robot_anchor_body_index': (env.robot_config.anchor_body_index, terrain_configs['robot'].anchor_body_index),
        'simulator_timing': ([env.simulator.config.sim.fps, env.simulator.config.sim.decimation],
                             [terrain_configs['simulator'].sim.fps, terrain_configs['simulator'].sim.decimation]),
        'simulator_pd_target_max_accel': (env.simulator.config.pd_target_max_accel,
                                          terrain_configs['simulator'].pd_target_max_accel),
        'terrain_sampling': ([env.terrain.config.num_samples_per_axis, env.terrain.config.sample_width],
                             [terrain_configs['terrain'].num_samples_per_axis, terrain_configs['terrain'].sample_width]),
    }
    differences = []
    for name, (runtime, saved) in pairs.items():
        differences.extend(_diff(_canonical(runtime), _canonical(saved), name))
    report = dict(compatible=not differences, checked=list(pairs), differences=differences)
    if differences:
        raise ValueError('Hybrid tracker contract mismatch: ' + json.dumps(report))
    return report

def load_official_actor(checkpoint, device):
    """Use the original ProtoMotions actor/config/normalizer implementation."""
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.agents.utils.normalization import materialize_lazy_running_stats_from_state_dict
    checkpoint = Path(checkpoint)
    configs = torch.load(checkpoint.parent / 'resolved_configs_inference.pt',
                         map_location='cpu', weights_only=False)
    actor_config = configs['agent'].model.actor
    actor = get_class(actor_config._target_)(config=actor_config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    actor_state = {key[len('_actor.'):]: value for key, value in state.items()
                   if key.startswith('_actor.')}
    if not actor_state:
        raise ValueError('Checkpoint contains no official _actor state')
    materialize_lazy_running_stats_from_state_dict(actor, actor_state)
    actor.load_state_dict(actor_state, strict=True)
    actor.to(device).eval().requires_grad_(False)
    # Loading/eval must preserve every parameter and normalizer exactly.
    loaded = actor.state_dict()
    assert set(loaded) == set(actor_state)
    for key in actor_state:
        assert torch.equal(loaded[key].cpu(), actor_state[key]), f'Actor state changed: {key}'
    return actor, configs
