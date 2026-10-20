"""Warm teacher checkpoint/factory shared by training, evaluation and collection."""
import hashlib
import json
import random
import numpy as np
import torch
from torch.distributions import Normal

from warm_policy import WarmResidualPolicy
from warm_prior import FrozenTerrainPrior
from motion_reference import resolve_motion_reference, file_sha256
from surface_contract import POLICY_KIND


def read_checkpoint(path, device='cpu'):
    from pathlib import Path
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get('policy_kind') != POLICY_KIND:
        raise ValueError('Checkpoint is not a terrain-prior warm correction teacher')
    state['_loaded_checkpoint_path'] = str(Path(path).resolve())
    state['_loaded_checkpoint_sha256'] = file_sha256(path)
    return state


def checkpoint_provenance(config, checkpoint):
    """Identity of the correction weights actually selected for this controller."""
    from pathlib import Path
    if checkpoint is None:
        if config.get('checkpoint'):
            raise ValueError('A configured correction checkpoint was not loaded')
        return dict(correction_state='untrained_zero_head', teacher_checkpoint_path=None,
                    teacher_checkpoint_sha256=None, teacher_checkpoint_update=0)
    if not config.get('checkpoint'):
        raise ValueError('Loaded correction weights require their actual config checkpoint path')
    path = Path(config['checkpoint']).resolve()
    actual = file_sha256(path)
    if checkpoint.get('_loaded_checkpoint_path', str(path)) != str(path):
        raise ValueError('Configured correction checkpoint differs from the loaded checkpoint')
    if checkpoint.get('_loaded_checkpoint_sha256', actual) != actual:
        raise ValueError('Correction checkpoint changed after it was loaded')
    return dict(correction_state='checkpoint', teacher_checkpoint_path=str(path),
                teacher_checkpoint_sha256=actual, teacher_checkpoint_update=int(checkpoint['update']))


def build_teacher_provenance(config, checkpoint, prior_identity, model_config, feature_version):
    """Pure identity builder shared with offline collection-plan preparation."""
    reference = resolve_motion_reference(config, require_version=False)
    provenance = dict(**prior_identity, **reference, feature_version=feature_version,
        reference_placement=config['reference_placement'], reward_regularizer=config['reward_regularizer'],
        correction_model=model_config)
    provenance['kind'] = POLICY_KIND
    for key in ('geometry_version','surface_mjcf_sha256','surface_mjcf_path','proximity_reward','migration'):
        if key in config:provenance[key]=config[key]
    provenance.update(checkpoint_provenance(config, checkpoint))
    identity = POLICY_KIND+':'+hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    return reference, provenance, identity


class WarmTeacher:
    def __init__(self, task, model, prior, prior_only=False):
        self.task, self.model, self.prior = task, model, prior
        self.feature_version = task.feature_version
        self.prior_only = bool(prior_only)
        self.last_prior = None

    def reset(self, ids=None):
        self.last_prior = None
        self.task._pending_prior = None

    def __call__(self, history):
        prior = self.prior.current()
        self.task.set_prior(prior)
        self.last_prior = prior.delta.detach().clone()
        dist, value = self.model(history, self.last_prior)
        if self.prior_only:
            dist = Normal(self.last_prior, dist.stddev)
        return dist, value


def make_warm_teacher(task, config, checkpoint=None):
    prior = FrozenTerrainPrior(task.agent, config)
    size = task.history.flatten(1).shape[-1]
    if checkpoint:
        if checkpoint['feature_version'] != task.feature_version or checkpoint['model_config']['obs_dim'] != size:
            raise ValueError('Warm checkpoint feature contract does not match this task')
        for key in ('flat_sha256','terrain_sha256'):
            if checkpoint['frozen_prior'][key] != prior.identity[key]:
                raise ValueError('Frozen actor checkpoint changed since warm teacher training')
        saved_version = checkpoint['config'].get('stop_reference_version')
        if saved_version != config.get('stop_reference_version'):
            raise ValueError('Warm teacher and evaluation use different motion reference versions')
        for key in ('geometry_version','surface_mjcf_sha256'):
            if checkpoint['config'].get(key)!=config.get(key):
                raise ValueError('Surface checkpoint geometry differs from the current task: '+key)
        if checkpoint.get('migration'):
            config['migration']=checkpoint['migration']
        model = WarmResidualPolicy(**checkpoint['model_config']).to(task.device)
        model.load_state_dict(checkpoint['policy'], strict=True)
    else:
        model = WarmResidualPolicy(size, width=config.get('width',256),
            correction_limit=config.get('correction_limit',.08), initial_std=config.get('initial_std',.012),
            min_std=config.get('min_std',.006), max_std=config.get('max_std',.04)).to(task.device)
    reference, provenance, identity = build_teacher_provenance(
        config, checkpoint, prior.identity, model.model_config, task.feature_version)
    config.update(reference)
    config['teacher_provenance'] = provenance
    config['teacher_identity'] = identity
    return WarmTeacher(task, model, prior, prior_only=config.get('method')=='terrain_prior')


def save_checkpoint(path, controller, optimizers, update, config, elapsed):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(policy_kind=POLICY_KIND, policy=controller.model.state_dict(),
        model_config=controller.model.model_config, feature_version=controller.feature_version,
        frozen_prior=controller.prior.identity, update=update, config=config, training_seconds=elapsed,
        optimizers={k:o.state_dict() for k,o in optimizers.items()} if optimizers else None,
        rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    state.update(geometry_version=config['geometry_version'],surface_mjcf_sha256=config['surface_mjcf_sha256'],
                 migration=config.get('migration'))
    tmp = path.with_suffix('.tmp')
    torch.save(state,tmp)
    tmp.replace(path)


def restore_rng(state):
    rng = state['rng']
    torch.set_rng_state(rng['torch'].cpu())
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])
