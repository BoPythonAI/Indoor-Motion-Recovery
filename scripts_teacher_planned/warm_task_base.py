"""V5 dynamic task with terrain features and correction-only effort penalty."""
import torch

from v5_task import RecoveryTask as V5RecoveryTask
from recovery_runtime import compose_action
from warm_policy import replace_prior_penalty


class WarmRecoveryTask(V5RecoveryTask):
    feature_version = 'v3_terrain_reference__terrain_correction_v1'

    def __init__(self, agent, config):
        expected = config.get('feature_version')
        if expected and expected != self.feature_version:
            raise ValueError('Warm correction network requires its explicit terrain feature version')
        if config.get('task_owned_gate') or config.get('momentum_observer'):
            raise ValueError('Warm prior has no observer or switching gate')
        super().__init__(agent, config)
        self._pending_prior = None
        self.cfg['reward_regularizer'] = '0.5 * mean((actual_total - composed_frozen_terrain_total)^2)'

    @torch.no_grad()
    def reset(self, ids=None, cases=None):
        self._pending_prior = None
        return super().reset(ids, cases)

    @torch.no_grad()
    def features(self):
        basic = super().features()
        terrain = self.env.get_obs()['terrain']
        if terrain.shape != (self.n, 256) or not torch.isfinite(terrain).all():
            raise ValueError('Warm trainable actor/critic both require current terrain256')
        return torch.cat([basic, terrain], -1)

    @torch.no_grad()
    def set_prior(self, prior):
        _, total, _ = compose_action(prior.flat_raw, prior.delta)
        self._pending_prior = dict(delta=prior.delta.detach().clone(), total=total.detach().clone(),
                                   flat=prior.flat_raw.tanh().detach().clone(), progress=self.env.progress_buf.clone())

    @torch.no_grad()
    def step(self, full_residual_sample):
        prior, self._pending_prior = self._pending_prior, None
        if prior is None or not torch.equal(prior['progress'], self.env.progress_buf):
            raise RuntimeError('Set a fresh current-state terrain prior before every physics step')
        history, reward, done, info = super().step(full_residual_sample)
        if not torch.allclose(info['base_normalized'], prior['flat'], atol=1e-6, rtol=0):
            raise RuntimeError('Flat baseline changed between frozen prior query and task actuation')
        reward, actual_correction = replace_prior_penalty(
            reward, info['executed_residual'], info['executed_total'], prior['total'])
        info.update(prior_delta=prior['delta'], prior_total=prior['total'], actual_correction=actual_correction,
                    executed_correction=actual_correction,
                    action_target_version='executed_total_minus_composed_terrain_prior_v1')
        return history, reward, done, info
